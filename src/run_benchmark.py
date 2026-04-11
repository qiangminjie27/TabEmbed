"""
Tabula-8B Embedding Benchmark Evaluation Script

Evaluate embedding model performance on classification and retrieval tasks:
1. Classification task: embedding + logistic regression
2. Retrieval task: vector retrieval using faiss

Supported features:
- Multi-GPU parallel encoding
- Configurable max_seq_length
- flat and ivf faiss index types

Usage:
python run_benchmark.py \
    --benchmark_dir /path/to/benchmark \
    --model_name_or_path sentence-transformers/all-MiniLM-L6-v2 \
    --output_dir /path/to/results \
    --tasks classification retrieval \
    --max_seq_length 512 \
    --batch_size 64
"""

import os
import json
import logging
import argparse
import time
import random
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Tuple, Optional
from collections import defaultdict

import numpy as np
import torch
from tqdm import tqdm


# =============================================================================
# Set Random Seed for Reproducibility
# =============================================================================

def set_seed(seed: int = 42):
    """Set all random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # Ensure CUDA operation determinism
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# Set seed at import time
set_seed(42)

# Sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder

# Sentence Transformers
from sentence_transformers import SentenceTransformer

# Faiss
import faiss


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    force=True
)
logger = logging.getLogger(__name__)


# =============================================================================
# Configuration Classes
# =============================================================================

@dataclass
class EvalConfig:
    """Evaluation configuration"""
    benchmark_dir: str  # Benchmark directory
    model_name_or_path: str  # Model name or path
    output_dir: str  # Output directory
    
    # Task selection
    tasks: List[str] = field(default_factory=lambda: ["classification", "retrieval"])
    
    # Encoding parameters
    max_seq_length: int = 512  # Maximum sequence length
    batch_size: int = 64  # Encoding batch size
    
    # Multi-GPU settings
    num_gpus: int = 1  # Number of GPUs (>1 enables multi-GPU parallelism)
    
    # Retrieval task parameters
    retrieval_index_type: str = "flat"  # Faiss index type: flat, ivf
    retrieval_top_k: List[int] = field(default_factory=lambda: [1, 5, 10, 20, 50, 100])
    
    # Dataset filtering
    benchmarks: Optional[List[str]] = None  # List of benchmarks to evaluate
    datasets: Optional[List[str]] = None  # List of datasets to evaluate


# =============================================================================
# Data Loading
# =============================================================================

def load_jsonl(filepath: str) -> List[Dict]:
    """Load JSONL file"""
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line.strip()))
    return data


def load_classification_data(
    benchmark_dir: str,
    benchmarks: Optional[List[str]] = None,
    datasets: Optional[List[str]] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Load classification task data
    
    Supports two directory structures:
    1. classification/{benchmark}/{dataset}/  (openml_cc18, openml_ctr23, unipredict)
    2. classification/{benchmark}/{sub_benchmark}/{dataset}/  (grinsztajn)
    
    Returns:
        {unique_dataset_key: {'train': [...], 'test': [...], 'metadata': {...}}}
    """
    clf_dir = os.path.join(benchmark_dir, "classification")
    
    if not os.path.exists(clf_dir):
        logger.warning(f"Classification task directory not found: {clf_dir}")
        return {}
    
    data = {}
    
    def try_load_dataset(ds_dir: str, bm_name: str, sub_bm: str = "") -> Optional[Tuple[str, Dict]]:
        """Try to load dataset, returns (unique_key, data) or None"""
        train_path = os.path.join(ds_dir, "train.jsonl")
        test_path = os.path.join(ds_dir, "test.jsonl")
        metadata_path = os.path.join(ds_dir, "metadata.json")
        
        if not os.path.exists(train_path) or not os.path.exists(test_path):
            return None
        
        ds_name = os.path.basename(ds_dir)
        
        if datasets and ds_name not in datasets:
            return None
        
        train_data = load_jsonl(train_path)
        test_data = load_jsonl(test_path)
        
        metadata = {}
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
        
        # Generate unique key: for datasets with sub_benchmark, use sub_benchmark/dataset_name
        if sub_bm:
            unique_key = f"{sub_bm}/{ds_name}"
        else:
            unique_key = ds_name
        
        return unique_key, {
            'train': train_data,
            'test': test_data,
            'metadata': metadata,
            'benchmark': bm_name,
            'sub_benchmark': sub_bm,
        }
    
    # Iterate all benchmark directories
    for bm_name in os.listdir(clf_dir):
        bm_dir = os.path.join(clf_dir, bm_name)
        if not os.path.isdir(bm_dir):
            continue
        
        if benchmarks and bm_name not in benchmarks:
            continue
        
        # Iterate contents under benchmark
        for item_name in os.listdir(bm_dir):
            item_dir = os.path.join(bm_dir, item_name)
            if not os.path.isdir(item_dir):
                continue
            
            # Check if this is a dataset directory (contains train.jsonl)
            if os.path.exists(os.path.join(item_dir, "train.jsonl")):
                # Directly a dataset directory
                result = try_load_dataset(item_dir, bm_name)
                if result:
                    unique_key, ds_data = result
                    data[unique_key] = ds_data
            else:
                # Possibly a sub_benchmark directory (e.g. grinsztajn/cat_clf/)
                for ds_name in os.listdir(item_dir):
                    ds_dir = os.path.join(item_dir, ds_name)
                    if os.path.isdir(ds_dir):
                        result = try_load_dataset(ds_dir, bm_name, sub_bm=item_name)
                        if result:
                            unique_key, ds_data = result
                            data[unique_key] = ds_data
    
    logger.info(f"Loaded {len(data)} classification datasets")
    return data


def load_retrieval_data(benchmark_dir: str) -> Tuple[List[Dict], List[Dict]]:
    """
    Load retrieval task data
    
    Returns:
        queries: Query list
        corpus: Corpus
    """
    ret_dir = os.path.join(benchmark_dir, "retrieval")
    
    queries = []
    query_path = os.path.join(ret_dir, "queries.jsonl")
    if os.path.exists(query_path):
        queries = load_jsonl(query_path)
    
    corpus = []
    corpus_path = os.path.join(ret_dir, "corpus.jsonl")
    if os.path.exists(corpus_path):
        corpus = load_jsonl(corpus_path)
    
    logger.info(f"Loaded {len(queries)} queries, {len(corpus)} corpus entries")
    return queries, corpus


# =============================================================================
# Embedding Encoder
# =============================================================================

class EmbeddingEncoder:
    """Multi-GPU Embedding Encoder"""
    
    def __init__(
        self,
        model_name_or_path: str,
        max_seq_length: int = 512,
        num_gpus: int = 1,
    ):
        self.model_name_or_path = model_name_or_path
        self.max_seq_length = max_seq_length
        self.num_gpus = num_gpus
        
        # Detect available GPUs
        self.available_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
        self.actual_num_gpus = min(num_gpus, self.available_gpus) if self.available_gpus > 0 else 0
        
        logger.info(f"Loading model: {model_name_or_path}")
        logger.info(f"Max sequence length: {max_seq_length}")
        logger.info(f"Available GPUs: {self.available_gpus}, Using: {self.actual_num_gpus}")
        
        # Determine device
        if self.actual_num_gpus > 0:
            device = "cuda:0"
        else:
            device = "cpu"
        
        # Load model
        self.model = SentenceTransformer(model_name_or_path, device=device, trust_remote_code=True)
        self.model.max_seq_length = max_seq_length
        
        # Get embedding dimension
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        
        # If model doesn't return dimension, obtain it by actual encoding
        if self.embedding_dim is None:
            logger.info("Model did not return embedding dimension, obtaining via actual encoding...")
            test_embedding = self.model.encode(["test"], convert_to_numpy=True)
            self.embedding_dim = test_embedding.shape[1]

        logger.info(f"Embedding dimension: {self.embedding_dim}")
    
    def _get_target_devices(self) -> Optional[List[str]]:
        """Get multi-GPU target device list"""
        if self.actual_num_gpus > 1:
            return [f"cuda:{i}" for i in range(self.actual_num_gpus)]
        return None
    
    def encode(
        self,
        texts: List[str],
        batch_size: int = 64,
        show_progress: bool = True,
    ) -> np.ndarray:
        """
        Encode text list, supports multi-GPU parallel encoding
        """
        if not texts:
            return np.array([])
        
        # Get multi-GPU device list
        target_devices = self._get_target_devices()
        
        if target_devices:
            # Multi-GPU parallel encoding (new API: pass device list via device parameter)
            logger.info(f"Using multi-GPU parallel encoding: {target_devices}")
            embeddings = self.model.encode(
                texts,
                convert_to_numpy=True,
                show_progress_bar=show_progress,
                batch_size=batch_size,
                device=target_devices,
                normalize_embeddings=True,
            )
        else:
            # Single GPU or CPU encoding
            embeddings = self.model.encode(
                texts,
                convert_to_numpy=True,
                show_progress_bar=show_progress,
                batch_size=batch_size,
                normalize_embeddings=True,
            )
        
        return embeddings.astype(np.float32)


# =============================================================================
# Classification Task Evaluation
# =============================================================================

def evaluate_classification(
    encoder: EmbeddingEncoder,
    data: Dict[str, Dict[str, Any]],
    config: EvalConfig,
) -> Dict[str, Any]:
    """
    Evaluate classification task
    
    Pipeline:
    1. Collect texts from all datasets and encode in one batch
    2. Split embeddings by dataset
    3. Train logistic regression and evaluate for each dataset
    """
    logger.info("=" * 60)
    logger.info("Evaluating Classification Task")
    logger.info("=" * 60)
    
    results = {
        'task': 'classification',
        'model': config.model_name_or_path,
        'datasets': {},
        'summary': {},
    }
    
    # =========================================================================
    # Step 1: Collect all texts and encode in one batch
    # =========================================================================
    all_texts = []
    dataset_indices = {}  # {ds_name: {'train': (start, end), 'test': (start, end)}}
    
    logger.info("Collecting texts from all datasets...")
    for ds_name, ds_data in data.items():
        train_samples = ds_data['train']
        test_samples = ds_data['test']
        
        train_texts = [s['text'] for s in train_samples]
        test_texts = [s['text'] for s in test_samples]
        
        # Record index ranges
        train_start = len(all_texts)
        all_texts.extend(train_texts)
        train_end = len(all_texts)
        
        test_start = len(all_texts)
        all_texts.extend(test_texts)
        test_end = len(all_texts)
        
        dataset_indices[ds_name] = {
            'train': (train_start, train_end),
            'test': (test_start, test_end),
        }
    
    logger.info(f"Total {len(all_texts)} texts from {len(data)} datasets")
    
    # Encode all texts in one batch
    logger.info("Encoding all texts...")
    all_embeddings = encoder.encode(all_texts, batch_size=config.batch_size, show_progress=True)
    logger.info(f"Encoding complete, embeddings shape: {all_embeddings.shape}")
    
    # =========================================================================
    # Step 2: Evaluate classification per dataset
    # =========================================================================
    all_accuracies = []
    all_f1_scores = []
    benchmark_results = defaultdict(list)
    
    logger.info("Starting per-dataset evaluation...")
    for ds_name, ds_data in tqdm(data.items(), desc="Evaluating classification datasets"):
        train_samples = ds_data['train']
        test_samples = ds_data['test']
        benchmark = ds_data.get('benchmark', 'unknown')
        
        # Extract labels
        train_labels = [s['label'] for s in train_samples]
        test_labels = [s['label'] for s in test_samples]
        
        # Encode labels
        label_encoder = LabelEncoder()
        label_encoder.fit(train_labels)
        train_y = label_encoder.transform(train_labels)
        test_y = label_encoder.transform(test_labels)
        
        num_classes = len(label_encoder.classes_)
        
        # Split embeddings
        train_start, train_end = dataset_indices[ds_name]['train']
        test_start, test_end = dataset_indices[ds_name]['test']
        
        train_embeddings = all_embeddings[train_start:train_end]
        test_embeddings = all_embeddings[test_start:test_end]
        
        # Train logistic regression
        clf = LogisticRegression(max_iter=1000, n_jobs=-1, random_state=42)
        clf.fit(train_embeddings, train_y)
        
        # Predict
        train_pred = clf.predict(train_embeddings)
        test_pred = clf.predict(test_embeddings)
        
        # Compute metrics
        train_acc = accuracy_score(train_y, train_pred)
        test_acc = accuracy_score(test_y, test_pred)
        
        # Use macro F1 uniformly
        train_f1 = f1_score(train_y, train_pred, average='macro')
        test_f1 = f1_score(test_y, test_pred, average='macro')
        
        ds_result = {
            'benchmark': benchmark,
            'dataset': ds_name,
            'num_classes': num_classes,
            'train_samples': len(train_labels),
            'test_samples': len(test_labels),
            'train_accuracy': float(train_acc),
            'test_accuracy': float(test_acc),
            'train_f1': float(train_f1),
            'test_f1': float(test_f1),
        }
        
        results['datasets'][ds_name] = ds_result
        all_accuracies.append(test_acc)
        all_f1_scores.append(test_f1)
        benchmark_results[benchmark].append(ds_result)
    
    # Aggregate statistics
    if all_accuracies:
        results['summary']['overall'] = {
            'num_datasets': len(all_accuracies),
            'mean_accuracy': float(np.mean(all_accuracies)),
            'std_accuracy': float(np.std(all_accuracies)),
            'mean_f1': float(np.mean(all_f1_scores)),
            'std_f1': float(np.std(all_f1_scores)),
        }
        
        # Aggregate by benchmark
        results['summary']['by_benchmark'] = {}
        for bm_name, bm_results in benchmark_results.items():
            accs = [r['test_accuracy'] for r in bm_results]
            f1s = [r['test_f1'] for r in bm_results]
            results['summary']['by_benchmark'][bm_name] = {
                'num_datasets': len(accs),
                'mean_accuracy': float(np.mean(accs)),
                'std_accuracy': float(np.std(accs)),
                'mean_f1': float(np.mean(f1s)),
                'std_f1': float(np.std(f1s)),
            }
    
    # Print summary
    logger.info("=" * 60)
    logger.info("Classification Task Summary")
    logger.info("=" * 60)
    
    if 'overall' in results['summary']:
        summary = results['summary']['overall']
        logger.info(f"Overall: {summary['num_datasets']} datasets")
        logger.info(f"  Mean Accuracy: {summary['mean_accuracy']:.4f} +/- {summary['std_accuracy']:.4f}")
        logger.info(f"  Mean F1: {summary['mean_f1']:.4f} +/- {summary['std_f1']:.4f}")
        
        for bm_name, bm_summary in results['summary']['by_benchmark'].items():
            logger.info(f"{bm_name}: {bm_summary['num_datasets']} datasets")
            logger.info(f"  Mean Accuracy: {bm_summary['mean_accuracy']:.4f} +/- {bm_summary['std_accuracy']:.4f}")
            logger.info(f"  Mean F1: {bm_summary['mean_f1']:.4f} +/- {bm_summary['std_f1']:.4f}")
    
    return results


# =============================================================================
# Retrieval Task Evaluation
# =============================================================================

class FaissIndex:
    """Faiss Index Wrapper"""
    
    def __init__(
        self,
        embedding_dim: int,
        index_type: str = "flat",
        use_gpu: bool = False,
    ):
        self.embedding_dim = embedding_dim
        self.index_type = index_type
        self.use_gpu = use_gpu
        self.index = None

    def build(self, embeddings: np.ndarray) -> None:
        """Build index"""
        embeddings = embeddings.astype(np.float32)
        n_samples = embeddings.shape[0]
        
        logger.info(f"Building {self.index_type.upper()} index (n={n_samples}, dim={self.embedding_dim})...")
        
        if self.index_type == "flat":
            # Flat (brute-force search) index
            self.index = faiss.IndexFlatIP(self.embedding_dim)
            self.index.add(embeddings)
            
        elif self.index_type == "ivf":
            # IVF index: auto-configure parameters based on data size
            nlist = min(int(np.sqrt(n_samples)), 4096)
            nlist = max(nlist, 1)  # At least 1 cluster
            
            quantizer = faiss.IndexFlatIP(self.embedding_dim)
            self.index = faiss.IndexIVFFlat(quantizer, self.embedding_dim, nlist, faiss.METRIC_INNER_PRODUCT)
            
            # Train index
            logger.info(f"  Training IVF index (nlist={nlist})...")
            self.index.train(embeddings)
            
            # Set nprobe: number of clusters checked during search
            nprobe = max(8, min(nlist // 5, 128))
            self.index.nprobe = nprobe
            logger.info(f"  Set nprobe={nprobe} (nlist={nlist}, search coverage={nprobe/nlist*100:.1f}%)")
            
            # Add vectors
            self.index.add(embeddings)
        
        else:
            raise ValueError(f"Unsupported index type: {self.index_type}")
        
        # GPU acceleration
        if self.use_gpu:
            try:
                # Check if multiple GPUs are available
                ngpus = faiss.get_num_gpus()
                if ngpus > 0:
                    logger.info(f"Moving index to {ngpus} GPUs...")
                    # Use all available GPUs
                    self.index = faiss.index_cpu_to_all_gpus(self.index)
                else:
                    logger.warning("No GPU detected, using CPU for retrieval")
            except Exception as e:
                logger.warning(f"Cannot use GPU for Faiss acceleration: {e}")

        logger.info(f"Index built, containing {self.index.ntotal} vectors")
    
    def search(self, queries: np.ndarray, top_k: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search nearest neighbors
        
        Returns:
            scores: (n_queries, top_k) similarity scores
            indices: (n_queries, top_k) neighbor indices
        """
        queries = queries.astype(np.float32)
        scores, indices = self.index.search(queries, top_k)
        return scores, indices


def compute_retrieval_metrics(
    queries: List[Dict],
    retrieved_indices: np.ndarray,
    top_k_list: List[int],
) -> Dict[str, Any]:
    """
    Compute retrieval evaluation metrics
    
    Metrics:
    - MAP@K
    - MRR@K
    - Recall@K
    - Precision@K
    - NDCG@K
    """
    metrics = {}
    
    # Group statistics by query type
    query_types = sorted(list(set(q.get('query_type', 'unknown') for q in queries)))
    # Group statistics by number of conditions
    num_conditions_set = sorted(list(set(q.get('num_conditions', 0) for q in queries)))
    
    # =========================================================================
    # Compute Recall@K, Precision@K, NDCG@K, MRR@K, MAP@K
    # =========================================================================
    for k in top_k_list:
        recalls = []
        precisions = []
        ndcgs = []
        mrrs = []
        maps = []
        
        # Group by type
        recalls_by_type = defaultdict(list)
        precisions_by_type = defaultdict(list)
        ndcgs_by_type = defaultdict(list)
        mrrs_by_type = defaultdict(list)
        maps_by_type = defaultdict(list)
        
        # Group by number of conditions
        recalls_by_num_conditions = defaultdict(list)
        precisions_by_num_conditions = defaultdict(list)
        ndcgs_by_num_conditions = defaultdict(list)
        mrrs_by_num_conditions = defaultdict(list)
        maps_by_num_conditions = defaultdict(list)

        # Cross-group by type and number of conditions
        recalls_by_type_and_conditions = defaultdict(list)
        precisions_by_type_and_conditions = defaultdict(list)
        ndcgs_by_type_and_conditions = defaultdict(list)
        mrrs_by_type_and_conditions = defaultdict(list)
        maps_by_type_and_conditions = defaultdict(list)

        for i, query in enumerate(queries):
            gt_indices = set(query.get('matching_indices', []))
            if not gt_indices:
                continue
            
            # Retrieved results (top K)
            retrieved = retrieved_indices[i, :k].tolist()
            retrieved_set = set(retrieved)
            
            # Recall@K: number of relevant docs retrieved / total relevant docs
            hits = len(gt_indices & retrieved_set)
            recall = hits / len(gt_indices)
            recalls.append(recall)
            
            # Precision@K: number of relevant docs retrieved / K
            precision = hits / k
            precisions.append(precision)
            
            # MRR@K: reciprocal rank of first relevant document
            rr = 0.0
            for rank, doc_idx in enumerate(retrieved, 1):
                if doc_idx in gt_indices:
                    rr = 1.0 / rank
                    break
            mrrs.append(rr)

            # MAP@K: Average Precision @ K
            ap = 0.0
            num_relevant_found = 0
            for rank, doc_idx in enumerate(retrieved, 1):
                if doc_idx in gt_indices:
                    num_relevant_found += 1
                    ap += num_relevant_found / rank
            ap /= len(gt_indices)
            maps.append(ap)
            
            # NDCG@K: Normalized Discounted Cumulative Gain
            dcg = 0.0
            for rank, doc_idx in enumerate(retrieved, 1):
                if doc_idx in gt_indices:
                    dcg += 1.0 / np.log2(rank + 1)
            
            # Ideal DCG (all relevant docs at the top)
            ideal_dcg = 0.0
            for rank in range(1, min(len(gt_indices), k) + 1):
                ideal_dcg += 1.0 / np.log2(rank + 1)
            
            ndcg = dcg / ideal_dcg if ideal_dcg > 0 else 0.0
            ndcgs.append(ndcg)
            
            # Statistics by type
            query_type = query.get('query_type', 'unknown')
            recalls_by_type[query_type].append(recall)
            precisions_by_type[query_type].append(precision)
            ndcgs_by_type[query_type].append(ndcg)
            mrrs_by_type[query_type].append(rr)
            maps_by_type[query_type].append(ap)
            
            # Statistics by number of conditions
            num_conditions = query.get('num_conditions', 0)
            recalls_by_num_conditions[num_conditions].append(recall)
            precisions_by_num_conditions[num_conditions].append(precision)
            ndcgs_by_num_conditions[num_conditions].append(ndcg)
            mrrs_by_num_conditions[num_conditions].append(rr)
            maps_by_num_conditions[num_conditions].append(ap)
            
            # Cross statistics by type and number of conditions
            type_cond_key = (query_type, num_conditions)
            recalls_by_type_and_conditions[type_cond_key].append(recall)
            precisions_by_type_and_conditions[type_cond_key].append(precision)
            ndcgs_by_type_and_conditions[type_cond_key].append(ndcg)
            mrrs_by_type_and_conditions[type_cond_key].append(rr)
            maps_by_type_and_conditions[type_cond_key].append(ap)

        # Aggregate @K metrics
        if recalls:
            metrics[f'recall@{k}'] = {
                'overall': float(np.mean(recalls)),
                'std': float(np.std(recalls)),
            }
            metrics[f'precision@{k}'] = {
                'overall': float(np.mean(precisions)),
                'std': float(np.std(precisions)),
            }
            metrics[f'ndcg@{k}'] = {
                'overall': float(np.mean(ndcgs)),
                'std': float(np.std(ndcgs)),
            }
            metrics[f'mrr@{k}'] = {
                'overall': float(np.mean(mrrs)),
                'std': float(np.std(mrrs)),
            }
            metrics[f'map@{k}'] = {
                'overall': float(np.mean(maps)),
                'std': float(np.std(maps)),
            }
            
            # By type
            for qt in query_types:
                if recalls_by_type[qt]:
                    metrics[f'recall@{k}'][f'by_type/{qt}'] = float(np.mean(recalls_by_type[qt]))
                    metrics[f'precision@{k}'][f'by_type/{qt}'] = float(np.mean(precisions_by_type[qt]))
                    metrics[f'ndcg@{k}'][f'by_type/{qt}'] = float(np.mean(ndcgs_by_type[qt]))
                    metrics[f'mrr@{k}'][f'by_type/{qt}'] = float(np.mean(mrrs_by_type[qt]))
                    metrics[f'map@{k}'][f'by_type/{qt}'] = float(np.mean(maps_by_type[qt]))
            
            # By number of conditions
            for nc in num_conditions_set:
                if recalls_by_num_conditions[nc]:
                    metrics[f'recall@{k}'][f'by_num_conditions/{nc}'] = float(np.mean(recalls_by_num_conditions[nc]))
                    metrics[f'precision@{k}'][f'by_num_conditions/{nc}'] = float(np.mean(precisions_by_num_conditions[nc]))
                    metrics[f'ndcg@{k}'][f'by_num_conditions/{nc}'] = float(np.mean(ndcgs_by_num_conditions[nc]))
                    metrics[f'mrr@{k}'][f'by_num_conditions/{nc}'] = float(np.mean(mrrs_by_num_conditions[nc]))
                    metrics[f'map@{k}'][f'by_num_conditions/{nc}'] = float(np.mean(maps_by_num_conditions[nc]))
            
            # Cross statistics by type and number of conditions
            for qt in query_types:
                for nc in num_conditions_set:
                    type_cond_key = (qt, nc)
                    if recalls_by_type_and_conditions[type_cond_key]:
                        metrics[f'recall@{k}'][f'by_type_conditions/{qt}/{nc}'] = float(np.mean(recalls_by_type_and_conditions[type_cond_key]))
                        metrics[f'precision@{k}'][f'by_type_conditions/{qt}/{nc}'] = float(np.mean(precisions_by_type_and_conditions[type_cond_key]))
                        metrics[f'ndcg@{k}'][f'by_type_conditions/{qt}/{nc}'] = float(np.mean(ndcgs_by_type_and_conditions[type_cond_key]))
                        metrics[f'mrr@{k}'][f'by_type_conditions/{qt}/{nc}'] = float(np.mean(mrrs_by_type_and_conditions[type_cond_key]))
                        metrics[f'map@{k}'][f'by_type_conditions/{qt}/{nc}'] = float(np.mean(maps_by_type_and_conditions[type_cond_key]))
    
    return metrics


def evaluate_retrieval(
    encoder: EmbeddingEncoder,
    queries: List[Dict],
    corpus: List[Dict],
    config: EvalConfig,
) -> Dict[str, Any]:
    """
    Evaluate retrieval task
    
    Uses MAP, MRR, Recall, Precision, NDCG as evaluation metrics
    """
    logger.info("=" * 60)
    logger.info("Evaluating Retrieval Task")
    logger.info("=" * 60)
    
    if not queries or not corpus:
        logger.warning("Queries or corpus is empty, skipping retrieval evaluation")
        return {'task': 'retrieval', 'error': 'empty data'}
    
    results = {
        'task': 'retrieval',
        'model': config.model_name_or_path,
        'index_type': config.retrieval_index_type,
        'corpus_size': len(corpus),
        'num_queries': len(queries),
        'metrics': {},
    }
    
    # Encode corpus
    logger.info(f"Encoding corpus ({len(corpus)} entries)...")
    corpus_texts = [c['text'] for c in corpus]
    corpus_embeddings = encoder.encode(corpus_texts, batch_size=config.batch_size)
    
    # Encode queries
    logger.info(f"Encoding queries ({len(queries)} entries)...")
    query_texts = [q['query_text'] for q in queries]
    query_embeddings = encoder.encode(query_texts, batch_size=config.batch_size)
    
    # Build index
    faiss_index = FaissIndex(
        embedding_dim=encoder.embedding_dim,
        index_type=config.retrieval_index_type,
        use_gpu=config.num_gpus > 0,
    )
    faiss_index.build(corpus_embeddings)
    
    # Search
    max_k = max(config.retrieval_top_k)
    logger.info(f"Performing retrieval (top_k={max_k})...")
    
    start_time = time.time()
    scores, retrieved_indices = faiss_index.search(query_embeddings, max_k)
    search_time = time.time() - start_time
    
    results['search_time_seconds'] = search_time
    results['queries_per_second'] = len(queries) / search_time
    
    logger.info(f"Retrieval complete: {search_time:.2f}s, {results['queries_per_second']:.1f} queries/s")
    
    # Compute metrics
    logger.info("Computing retrieval metrics...")
    metrics = compute_retrieval_metrics(queries, retrieved_indices, config.retrieval_top_k)
    results['metrics'] = metrics
    
    # Print summary
    logger.info("=" * 60)
    logger.info("Retrieval Task Summary")
    logger.info("=" * 60)
    logger.info(f"Corpus size: {len(corpus)}")
    logger.info(f"Number of queries: {len(queries)}")
    logger.info(f"Index type: {config.retrieval_index_type}")
    logger.info(f"Retrieval speed: {results['queries_per_second']:.1f} queries/s")
    
    # Print metrics for each K value
    for k in config.retrieval_top_k:
        logger.info(f" @{k}:")
        for metric_name in ['map', 'mrr', 'recall', 'precision', 'ndcg']:
            metric_key = f'{metric_name}@{k}'
            if metric_key in results['metrics']:
                overall = results['metrics'][metric_key]['overall']
                logger.info(f"    {metric_name.upper()}: {overall:.4f}")
    
    # Detailed output by type
    query_types = set(q.get('query_type', 'unknown') for q in queries)
    
    # Print Recall@10 (or first K value) by type
    if config.retrieval_top_k:
        target_k = 10 if 10 in config.retrieval_top_k else config.retrieval_top_k[0]
        
        logger.info(f"Metrics by query type (Top-K={target_k}):")
        
        for metric in ['map', 'mrr', 'recall', 'precision', 'ndcg']:
            key = f'{metric}@{target_k}'
            if key in results['metrics']:
                logger.info(f"  {metric.upper()}:")
                for qt in sorted(query_types):
                    type_key = f'by_type/{qt}'
                    if type_key in results['metrics'][key]:
                        logger.info(f"    {qt}: {results['metrics'][key][type_key]:.4f}")
    
    return results


# =============================================================================
# Main Evaluator
# =============================================================================

class BenchmarkEvaluator:
    """Benchmark Evaluator"""
    
    def __init__(self, config: EvalConfig):
        self.config = config
        self.encoder = None
    
    def run(self) -> Dict[str, Any]:
        """Run evaluation"""
        # Initialize encoder
        self.encoder = EmbeddingEncoder(
            model_name_or_path=self.config.model_name_or_path,
            max_seq_length=self.config.max_seq_length,
            num_gpus=self.config.num_gpus,
        )
        
        # Create output subdirectory: model_name/timestamp
        model_name = os.path.basename(self.config.model_name_or_path.rstrip('/\\'))
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        run_output_dir = os.path.join(self.config.output_dir, model_name, timestamp)
        os.makedirs(run_output_dir, exist_ok=True)
        logger.info(f"Results will be saved to: {run_output_dir}")
        
        # Save config for this run
        config_output_path = os.path.join(run_output_dir, "config.json")
        with open(config_output_path, 'w', encoding='utf-8') as f:
            json.dump(asdict(self.config), f, indent=2, ensure_ascii=False)
        logger.info(f"Config saved to: {config_output_path}")
        
        results = {
            'model': self.config.model_name_or_path,
            'max_seq_length': self.config.max_seq_length,
            'embedding_dim': self.encoder.embedding_dim,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'output_dir': run_output_dir,
        }
        
        # Classification task
        if "classification" in self.config.tasks:
            clf_data = load_classification_data(
                self.config.benchmark_dir,
                benchmarks=self.config.benchmarks,
                datasets=self.config.datasets,
            )
            
            if clf_data:
                clf_results = evaluate_classification(self.encoder, clf_data, self.config)
                results['classification'] = clf_results
                
                # Save classification results
                clf_output_path = os.path.join(run_output_dir, "classification_results.json")
                with open(clf_output_path, 'w', encoding='utf-8') as f:
                    json.dump(clf_results, f, indent=2, ensure_ascii=False)
                logger.info(f"Classification results saved to: {clf_output_path}")
        
        # Retrieval task
        if "retrieval" in self.config.tasks:
            queries, corpus = load_retrieval_data(self.config.benchmark_dir)
            
            if queries and corpus:
                ret_results = evaluate_retrieval(self.encoder, queries, corpus, self.config)
                results['retrieval'] = ret_results
                
                # Save retrieval results
                ret_output_path = os.path.join(run_output_dir, "retrieval_results.json")
                with open(ret_output_path, 'w', encoding='utf-8') as f:
                    json.dump(ret_results, f, indent=2, ensure_ascii=False)
                logger.info(f"Retrieval results saved to: {ret_output_path}")
        
        # Save full results
        full_output_path = os.path.join(run_output_dir, "full_results.json")
        with open(full_output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info(f"Full results saved to: {full_output_path}")


# =============================================================================
# CLI Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Evaluate Embedding Model Performance on Tabula Benchmark")
    
    # Basic arguments
    parser.add_argument("--benchmark_dir", "-b", type=str, required=True,
                        help="Benchmark directory (output of build_benchmark.py)")
    parser.add_argument("--model_name_or_path", "-m", type=str, required=True,
                        help="Embedding model name or path")
    parser.add_argument("--output_dir", "-o", type=str, required=True,
                        help="Results output directory")
    
    # Task selection
    parser.add_argument("--tasks", type=str, nargs='+', 
                        default=["classification", "retrieval"],
                        choices=["classification", "retrieval"],
                        help="Tasks to evaluate (default: classification retrieval)")
    
    # Encoding arguments
    parser.add_argument("--max_seq_length", type=int, default=1024,
                        help="Maximum sequence length (default: 1024)")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Encoding batch size (default: 64)")
    
    # Multi-GPU settings
    parser.add_argument("--num_gpus", type=int, default=1,
                        help="Number of GPUs (default: 1, >1 enables multi-GPU parallelism)")
    
    # Dataset filtering
    parser.add_argument("--benchmarks", type=str, nargs='+', default=None,
                        help="List of benchmarks to evaluate")
    parser.add_argument("--datasets", type=str, nargs='+', default=None,
                        help="List of datasets to evaluate")
    
    # Retrieval task arguments
    parser.add_argument("--retrieval_index_type", type=str, default="flat",
                        choices=["flat", "ivf"],
                        help="Faiss index type (default: flat)")
    parser.add_argument("--retrieval_top_k", type=int, nargs='+', 
                        default=[1, 5, 10, 20, 50, 100],
                        help="Retrieval Top-K value list (default: 1 5 10 20 50 100)")
    
    args = parser.parse_args()
    
    config = EvalConfig(
        benchmark_dir=args.benchmark_dir,
        model_name_or_path=args.model_name_or_path,
        output_dir=args.output_dir,
        tasks=args.tasks,
        max_seq_length=args.max_seq_length,
        batch_size=args.batch_size,
        num_gpus=args.num_gpus,
        benchmarks=args.benchmarks,
        datasets=args.datasets,
        retrieval_index_type=args.retrieval_index_type,
        retrieval_top_k=args.retrieval_top_k,
    )
    
    evaluator = BenchmarkEvaluator(config)
    evaluator.run()
    
    logger.info("=" * 60)
    logger.info("Evaluation complete!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
