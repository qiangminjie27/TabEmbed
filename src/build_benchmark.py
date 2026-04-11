"""
Tabula-8B Benchmark Construction Script

Builds an embedding model evaluation benchmark based on the tabula-8b-eval-suite dataset.
Evaluation tasks include:
1. Classification: evaluated independently per dataset (logistic regression)
2. Retrieval: generates query conditions and retrieves matching rows

Construction rules based on process_T4.py:
- Numeric precision: defaults to 2 decimal places
- Serialization format: "The {feature} is {value}. ..."
- Length limit: 4096 * 3.5 = 14336 characters
- Query construction: supports numeric/categorical/mixed types

Usage:
python build_benchmark.py \
    --input_dir /path/to/tabula-8b-eval-suite \
    --output_dir /path/to/output \
    --num_workers 4
"""

import os
import re
import json
import glob
import yaml
import logging
import argparse
import random
import hashlib
from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple, Optional, Set
from multiprocessing import Pool
import numpy as np
import pandas as pd
from tqdm import tqdm


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    force=True
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# =============================================================================
# Configuration Classes
# =============================================================================

@dataclass
class SerializerConfig:
    """Serializer configuration"""
    numeric_precision: int = 2  # Numeric precision (decimal places)
    model_max_len_tokens: int = 4096  # Model max token length
    appx_chars_per_token: float = 3.5  # Approximate characters per token
    
    @property
    def max_chars(self) -> int:
        """Calculate maximum character count"""
        return int(self.model_max_len_tokens * self.appx_chars_per_token)


@dataclass
class RetrievalConfig:
    """Retrieval task configuration"""
    num_queries_per_type: int = 1000  # Number of queries per type
    min_matches: int = 5  # Minimum matches for retrieval queries
    max_conditions: int = 3  # Maximum number of conditions
    numeric_precision: int = 2  # Numeric precision (for query condition generation)
    max_samples_per_dataset: Optional[int] = 1000  # Max samples per dataset for retrieval


@dataclass
class ClassificationConfig:
    """Classification task configuration"""
    max_samples_per_dataset: Optional[int] = None  # Max samples per dataset (None = unlimited)
    train_ratio: float = 0.8  # Train split ratio
    max_label_ratio: float = 0.1  # Max ratio of num_classes/num_samples (above this is treated as regression)
    max_num_labels: int = 50  # Maximum number of classes


@dataclass
class BenchmarkConfig:
    """Benchmark configuration"""
    input_dir: str = ""
    output_dir: str = ""
    random_seed: int = 42
    num_workers: int = 4
    chunk_size: int = 1000
    serializer_config: SerializerConfig = field(default_factory=SerializerConfig)
    retrieval_config: RetrievalConfig = field(default_factory=RetrievalConfig)
    classification_config: ClassificationConfig = field(default_factory=ClassificationConfig)
    # Dataset filtering
    benchmarks: Optional[List[str]] = None  # List of benchmarks to process, None = all
    min_samples: int = 100  # Minimum sample count (filters out too-small datasets)


# =============================================================================
# Utility Functions
# =============================================================================

def is_numeric_value(val: Any) -> bool:
    """Check whether a value is numeric"""
    if pd.isnull(val):
        return False
    if isinstance(val, (int, float, np.integer, np.floating)):
        return True
    if isinstance(val, str):
        return bool(re.match(r"^-?\d+(\.\d+)?$", str(val).strip()))
    return False


def preprocess_value(val: Any, precision: int = 2) -> Any:
    """Preprocess a value"""
    if pd.isnull(val):
        return "unknown"
    if isinstance(val, (int, np.integer)):
        return int(val)
    if isinstance(val, (float, np.floating)):
        return round(float(val), precision)
    if isinstance(val, str):
        val = val.strip()
        if val.endswith("."):
            val = val[:-1]
        return val
    return val


def serialize_row(row: Dict[str, Any], precision: int = 2) -> str:
    """Serialize row data to text format: "The {feature} is {value}. ..." """
    parts = []
    for key, value in row.items():
        processed_value = preprocess_value(value, precision)
        parts.append(f"The {key} is {processed_value}.")
    return " ".join(parts)


def set_random_seed(seed: int):
    """Set random seed"""
    random.seed(seed)
    np.random.seed(seed)
    logger.info(f"Random seed set to: {seed}")


def get_original_linux_path(dataset_info: 'DatasetInfo', config: 'BenchmarkConfig') -> str:
    """
    Reconstruct the original Linux path used to generate the canonical dataset.
    This ensures that the MD5 hash (and therefore the random seed) is identical
    on Windows and Linux, aligning with the already fixed Linux datasets.
    """
    try:
        rel_path = os.path.relpath(dataset_info.csv_path, config.input_dir)
    except ValueError:
        rel_path = dataset_info.csv_path
    rel_path = rel_path.replace('\\', '/')
    linux_input_dir = "/ossfs/workspace/TabularEmbedding_Evalsuite/data/tabula-8b-eval-suite"
    return f"{linux_input_dir}/{rel_path}"


def dataset_file_seed(dataset_info: 'DatasetInfo', config: 'BenchmarkConfig') -> int:
    """Compute a deterministic per-dataset seed using the original Linux path."""
    linux_path = get_original_linux_path(dataset_info, config)
    key_hash = int(hashlib.md5(linux_path.encode()).hexdigest(), 16) % (2**31)
    return config.random_seed + key_hash


# =============================================================================
# Dataset Loading
# =============================================================================

@dataclass
class DatasetInfo:
    """Dataset information"""
    name: str  # Dataset name
    benchmark: str  # Parent benchmark (grinsztajn, openml_cc18, openml_ctr23, unipredict)
    task_type: str  # Task type (clf/reg)
    data_type: str  # Data type (cat/num/mixed)
    csv_path: str  # CSV file path
    yaml_path: str  # YAML config path
    feature_path: str  # feature_list.jsonl path
    target_column: str  # Target column name
    label_values: List[str]  # Label value list
    features: List[Dict]  # Feature list
    sub_benchmark: str = ""  # Sub-benchmark (e.g. grinsztajn's cat_clf, num_reg, etc.)
    

def discover_datasets(input_dir: str, benchmarks: Optional[List[str]] = None) -> List[DatasetInfo]:
    """
    Discover all datasets
    
    Dataset structure:
    - grinsztajn/{cat_clf,cat_reg,num_clf,num_reg}/{dataset_name}/
    - openml_cc18/{dataset_name}/
    - openml_ctr23/{dataset_name}/
    - unipredict/{user_name}/{dataset_name}/
    """
    datasets = []
    
    # Define benchmark directory structure
    # (pattern, task_type, data_type, sub_benchmark)
    benchmark_patterns = {
        'grinsztajn': [
            ('grinsztajn/cat_clf/*', 'clf', 'cat', 'cat_clf'),
            ('grinsztajn/cat_reg/*', 'reg', 'cat', 'cat_reg'),
            ('grinsztajn/num_clf/*', 'clf', 'num', 'num_clf'),
            ('grinsztajn/num_reg/*', 'reg', 'num', 'num_reg'),
        ],
        'openml_cc18': [
            ('openml_cc18/*', 'clf', 'mixed', ''),
        ],
        'openml_ctr23': [
            ('openml_ctr23/*', 'reg', 'mixed', ''),
        ],
        'unipredict': [
            ('unipredict/*/*', 'clf', 'mixed', ''),  # unipredict/{user}/{dataset}
        ],
    }
    
    # Filter benchmarks to process
    if benchmarks:
        benchmark_patterns = {k: v for k, v in benchmark_patterns.items() if k in benchmarks}
    
    for benchmark_name, patterns in benchmark_patterns.items():
        for pattern, task_type, data_type, sub_benchmark in patterns:
            full_pattern = os.path.join(input_dir, pattern)
            dataset_dirs = sorted(glob.glob(full_pattern))
            
            for dataset_dir in dataset_dirs:
                if not os.path.isdir(dataset_dir):
                    continue
                
                # Find CSV files
                csv_files = sorted(glob.glob(os.path.join(dataset_dir, "*.csv")))
                if not csv_files:
                    continue
                csv_path = csv_files[0]
                
                # Get dataset name
                dataset_name = os.path.splitext(os.path.basename(csv_path))[0]
                
                # Find corresponding YAML file
                yaml_path = os.path.join(dataset_dir, f"{dataset_name}.yaml")
                if not os.path.exists(yaml_path):
                    # Try alternative naming
                    yaml_files = sorted(glob.glob(os.path.join(dataset_dir, "*.yaml")))
                    if yaml_files:
                        yaml_path = yaml_files[0]
                    else:
                        logger.warning(f"YAML config not found: {dataset_dir}")
                        continue
                
                # Find feature_list.jsonl
                feature_path = os.path.join(dataset_dir, "feature_list.jsonl")
                if not os.path.exists(feature_path):
                    logger.warning(f"feature_list.jsonl not found: {dataset_dir}")
                    continue
                
                # Read YAML config
                try:
                    with open(yaml_path, 'r', encoding='utf-8') as f:
                        yaml_config = yaml.safe_load(f)
                except Exception as e:
                    logger.warning(f"Failed to read YAML {yaml_path}: {e}")
                    continue
                
                # Read feature list
                try:
                    features = []
                    with open(feature_path, 'r', encoding='utf-8') as f:
                        for line in f:
                            features.append(json.loads(line.strip()))
                except Exception as e:
                    logger.warning(f"Failed to read feature_list.jsonl {feature_path}: {e}")
                    continue
                
                # Find target column
                target_column = None
                for feat in features:
                    if feat.get('is_target', False):
                        target_column = feat['name']
                        break
                
                if not target_column:
                    logger.warning(f"Target column not found: {dataset_dir}")
                    continue
                
                # Get label values
                label_values = yaml_config.get('label_values', [])
                
                dataset_info = DatasetInfo(
                    name=dataset_name,
                    benchmark=benchmark_name,
                    task_type=task_type,
                    data_type=data_type,
                    csv_path=csv_path,
                    yaml_path=yaml_path,
                    feature_path=feature_path,
                    target_column=target_column,
                    label_values=label_values,
                    features=features,
                    sub_benchmark=sub_benchmark,
                )
                datasets.append(dataset_info)
    
    logger.info(f"Found {len(datasets)} datasets")
    
    # Statistics by benchmark
    benchmark_counts = {}
    for ds in datasets:
        benchmark_counts[ds.benchmark] = benchmark_counts.get(ds.benchmark, 0) + 1
    for benchmark, count in sorted(benchmark_counts.items()):
        logger.info(f"  - {benchmark}: {count} datasets")
    
    return datasets


def load_dataset(
    dataset_info: DatasetInfo, 
    config: BenchmarkConfig,
    max_samples: Optional[int] = None,
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    Load dataset and return DataFrame with column type mapping
    
    Args:
        dataset_info: Dataset information
        config: Configuration
        max_samples: Maximum number of samples (None = unlimited)
    
    Returns:
        df: Data DataFrame
        column_types: Column type mapping {'col_name': 'numeric'/'categorical'}
    """
    df = pd.read_csv(dataset_info.csv_path)
    
    # Determine column types based on feature_list.jsonl
    column_types = {}
    feature_kinds = {f['name']: f.get('kind', 'category') for f in dataset_info.features}
    
    for col in df.columns:
        if col == dataset_info.target_column:
            continue
        kind = feature_kinds.get(col, 'category')
        if kind in ['float', 'int']:
            column_types[col] = 'numeric'
        else:
            column_types[col] = 'categorical'
    
    # Limit sample count (only when specified)
    if max_samples and len(df) > max_samples:
        df = df.sample(n=max_samples, random_state=config.random_seed)
    
    return df, column_types


# =============================================================================
# Sample Building
# =============================================================================

def build_samples(
    dataset_info: DatasetInfo,
    config: BenchmarkConfig,
    max_samples: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Build samples for a single dataset (shared by classification and retrieval tasks)
    
    Args:
        dataset_info: Dataset information
        config: Configuration
        max_samples: Maximum number of samples (None = unlimited)
    
    Each sample contains:
    - text: Serialized row text
    - label: Target column value
    - dataset: Dataset name
    - benchmark: Parent benchmark
    - row_data: Raw row data (for retrieval task condition matching)
    - column_types: Column type mapping
    """
    df, column_types = load_dataset(dataset_info, config, max_samples=max_samples)
    
    if len(df) < config.min_samples:
        logger.warning(f"Dataset {dataset_info.name} has insufficient samples ({len(df)} < {config.min_samples}), skipping")
        return []
    
    samples = []
    precision = config.serializer_config.numeric_precision
    max_chars = config.serializer_config.max_chars
    
    target_col = dataset_info.target_column
    feature_cols = [col for col in df.columns if col != target_col]
    
    for idx, row in df.iterrows():
        # Extract features
        row_dict = {col: row[col] for col in feature_cols}
        label = str(row[target_col])
        
        # Serialize
        text = serialize_row(row_dict, precision)
        
        # Check length
        if len(text) > max_chars:
            continue
        
        sample = {
            'text': text,
            'label': label,
            'dataset': dataset_info.name,
            'benchmark': dataset_info.benchmark,
            'task_type': dataset_info.task_type,
            'row_data': row_dict,
            'column_types': column_types,
        }
        samples.append(sample)
    
    logger.info(f"Dataset {dataset_info.name}: {len(samples)}/{len(df)} samples")
    
    return samples


# =============================================================================
# Multiprocessing Support
# =============================================================================

# Global variables for multiprocessing
_global_config: Optional[BenchmarkConfig] = None
_global_max_samples: Optional[int] = None


def init_worker(config: BenchmarkConfig, max_samples: Optional[int]):
    """Initialize worker process"""
    global _global_config, _global_max_samples
    _global_config = config
    _global_max_samples = max_samples


def process_dataset_wrapper(dataset_info: DatasetInfo) -> List[Dict]:
    """Multiprocessing wrapper function"""
    try:
        # Set deterministic seed using the original Linux path
        file_seed = dataset_file_seed(dataset_info, _global_config)
        random.seed(file_seed)
        np.random.seed(file_seed)
        
        return build_samples(dataset_info, _global_config, max_samples=_global_max_samples)
    except Exception as e:
        logger.warning(f"Error processing dataset {dataset_info.name}: {e}")
        return []


def process_datasets_parallel(
    datasets: List[DatasetInfo],
    config: BenchmarkConfig,
    max_samples: Optional[int],
    num_workers: int,
    desc: str = "Processing datasets",
) -> List[List[Dict]]:
    """
    Process multiple datasets in parallel
    
    Args:
        datasets: List of datasets
        config: Configuration
        max_samples: Max samples per dataset
        num_workers: Number of worker processes
        desc: Progress bar description
    
    Returns:
        List of sample lists, one per dataset
    """
    if num_workers > 1:
        logger.info(f"Using {num_workers} processes for parallel processing...")
        with Pool(
            num_workers,
            initializer=init_worker,
            initargs=(config, max_samples)
        ) as pool:
            results = list(tqdm(
                pool.imap(process_dataset_wrapper, datasets),
                total=len(datasets),
                desc=desc
            ))
        return results
    else:
        # Single-process fallback
        results = []
        for dataset_info in tqdm(datasets, desc=desc):
            file_seed = dataset_file_seed(dataset_info, config)
            random.seed(file_seed)
            np.random.seed(file_seed)
            
            samples = build_samples(dataset_info, config, max_samples=max_samples)
            results.append(samples)
        return results


# =============================================================================
# Retrieval Task Dataset Construction
# =============================================================================

class RetrievalQueryBuilder:
    """Retrieval query builder"""
    
    def __init__(self, config: RetrievalConfig, random_seed: int = 42):
        self.config = config
        self.random_seed = random_seed
        
        # Indices (for fast matching)
        self._categorical_indices: Dict[str, Dict[str, Set[int]]] = {}
        self._numeric_arrays: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        self._n_samples: int = 0
    
    def _build_column_indices(self, samples: List[Dict]) -> None:
        """Pre-build column indices to accelerate condition matching"""
        logger.info("Pre-building column indices to accelerate condition matching...")
        
        categorical_indices = {}
        numeric_data = {}
        
        for idx, sample in enumerate(tqdm(samples, desc="Building column indices")):
            row_data = sample.get('row_data', {})
            column_types = sample.get('column_types', {})
            
            for col, col_type in column_types.items():
                val = row_data.get(col)
                if val is None or pd.isna(val):
                    continue
                
                if col_type == 'numeric':
                    try:
                        if col not in numeric_data:
                            numeric_data[col] = []
                        numeric_data[col].append((idx, float(val)))
                    except (ValueError, TypeError):
                        pass
                else:
                    str_val = str(val)
                    if col not in categorical_indices:
                        categorical_indices[col] = {}
                    if str_val not in categorical_indices[col]:
                        categorical_indices[col][str_val] = set()
                    categorical_indices[col][str_val].add(idx)
        
        # Convert to numpy arrays
        numeric_arrays = {}
        for col, data in numeric_data.items():
            if data:
                indices = np.array([d[0] for d in data], dtype=np.int32)
                values = np.array([d[1] for d in data], dtype=np.float32)
                numeric_arrays[col] = (indices, values)
        
        self._categorical_indices = categorical_indices
        self._numeric_arrays = numeric_arrays
        self._n_samples = len(samples)
        
        logger.info(f"Index built: {len(categorical_indices)} categorical columns, {len(numeric_arrays)} numeric columns")
    
    def _generate_numeric_condition(self, column: str, value: Any) -> Optional[Dict]:
        """Generate a numeric condition"""
        if pd.isna(value):
            return None
        
        try:
            float_value = float(value)
        except (ValueError, TypeError):
            return None
        
        operators = ['==', '>', '<']
        operator = random.choice(operators)
        precision = self.config.numeric_precision
        
        if operator == '==':
            return {'field': column, 'operator': operator, 'value': round(float_value, precision), 'type': 'numeric'}
        elif operator == '>':
            delta = abs(float_value) * random.uniform(0.05, 0.5) + 0.01
            threshold = float_value - delta
            return {'field': column, 'operator': operator, 'value': round(threshold, precision), 'type': 'numeric'}
        else:  # <
            delta = abs(float_value) * random.uniform(0.05, 0.5) + 0.01
            threshold = float_value + delta
            return {'field': column, 'operator': operator, 'value': round(threshold, precision), 'type': 'numeric'}
    
    def _generate_categorical_condition(self, column: str, value: Any) -> Optional[Dict]:
        """Generate a categorical condition"""
        if pd.isna(value):
            return None
        return {'field': column, 'operator': '==', 'value': str(value), 'type': 'categorical'}
    
    def _find_matches_fast(self, conditions: List[Dict]) -> List[int]:
        """Find matches quickly using pre-built indices"""
        result_set = None
        
        for cond in conditions:
            field = cond['field']
            operator = cond['operator']
            value = cond['value']
            cond_type = cond['type']
            
            matching = None
            
            if cond_type == 'numeric' and field in self._numeric_arrays:
                indices, values = self._numeric_arrays[field]
                
                if operator == '==':
                    mask = np.abs(values - value) < 1e-6
                elif operator == '>':
                    mask = values > value
                elif operator == '<':
                    mask = values < value
                else:
                    continue
                
                matching = set(indices[mask].tolist())
                
            elif cond_type == 'categorical' and field in self._categorical_indices:
                str_value = str(value)
                matching = self._categorical_indices[field].get(str_value, set())
            
            if matching is None:
                return []
            
            if result_set is None:
                result_set = matching.copy() if isinstance(matching, set) else set(matching)
            else:
                result_set &= matching
            
            if not result_set:
                return []
        
        return sorted(result_set) if result_set else []
    
    def _generate_query_text(self, conditions: List[Dict]) -> str:
        """Generate natural language query text"""
        parts = []
        op_map = {'==': 'is', '>': 'greater than', '<': 'less than'}
        
        for cond in conditions:
            field = cond['field']
            operator = cond['operator']
            value = cond['value']
            if isinstance(value, float):
                rounded = round(value, self.config.numeric_precision)
                if rounded == int(rounded):
                    value_str = str(int(rounded))
                else:
                    value_str = str(rounded)
            else:
                value_str = str(value)
            parts.append(f"{field} {op_map[operator]} {value_str}")
        
        return "find records where " + " and ".join(parts)
    
    def _prepare_seed_pools(self, samples: List[Dict]) -> Dict[str, List[int]]:
        """Prepare seed pools"""
        seed_pools = {'numeric': [], 'categorical': [], 'mixed': []}
        
        for idx, sample in enumerate(samples):
            column_types = sample.get('column_types', {})
            if not column_types:
                continue
            
            has_numeric = any(t == 'numeric' for t in column_types.values())
            has_categorical = any(t == 'categorical' for t in column_types.values())
            
            if has_numeric and has_categorical:
                seed_pools['mixed'].append(idx)
            elif has_numeric:
                seed_pools['numeric'].append(idx)
            elif has_categorical:
                seed_pools['categorical'].append(idx)
        
        return seed_pools
    
    def build_queries(self, samples: List[Dict]) -> Dict[str, List[Dict]]:
        """
        Generate retrieval queries
        
        For each query type (numeric/categorical/mixed), queries are evenly distributed
        across different condition counts.
        E.g.: num_per_type=3000, max_conditions=3 -> ~1000 queries per condition count
        
        Returns:
            queries: {'numeric': [...], 'categorical': [...], 'mixed': [...]}
        """
        self._build_column_indices(samples)
        
        seed_pools = self._prepare_seed_pools(samples)
        logger.info(f"Seed pools ready:")
        logger.info(f"  - Numeric only: {len(seed_pools['numeric'])} samples")
        logger.info(f"  - Categorical only: {len(seed_pools['categorical'])} samples")
        logger.info(f"  - Mixed: {len(seed_pools['mixed'])} samples")
        
        queries = {'numeric': [], 'categorical': [], 'mixed': []}
        
        # Expand seed pools: mixed type can be used to generate numeric and categorical queries
        valid_seed_pools = {
            'numeric': seed_pools['numeric'] + seed_pools['mixed'],
            'categorical': seed_pools['categorical'] + seed_pools['mixed'],
            'mixed': seed_pools['mixed']
        }
        
        max_conds = self.config.max_conditions
        
        for query_type in ['numeric', 'categorical', 'mixed']:
            current_pool = valid_seed_pools[query_type]
            if not current_pool:
                logger.warning(f"Cannot generate '{query_type}' queries, seed pool is empty")
                continue
            
            # Determine valid condition count range for this type
            # mixed type requires at least 2 conditions (1 numeric + 1 categorical)
            min_conds = 2 if query_type == 'mixed' else 1
            valid_cond_counts = list(range(min_conds, max_conds + 1))
            
            if not valid_cond_counts:
                logger.warning(f"Cannot generate '{query_type}' queries, max_conditions={max_conds} is insufficient")
                continue
            
            # Evenly distribute across condition counts
            num_per_cond = self.config.num_queries_per_type // len(valid_cond_counts)
            remainder = self.config.num_queries_per_type % len(valid_cond_counts)
            
            logger.info(f"Generating {query_type} queries (target: {self.config.num_queries_per_type}, "
                       f"condition count {min_conds}-{max_conds}, ~{num_per_cond} per count)...")
            
            total_generated = 0
            condition_stats = {}
            
            for cond_idx, num_conditions in enumerate(valid_cond_counts):
                # Allocate count: earlier condition counts get one extra (handle remainder)
                target_count = num_per_cond + (1 if cond_idx < remainder else 0)
                
                generated = 0
                attempts = 0
                max_attempts = target_count * 50  # Give multi-condition queries more attempts
                
                pbar = tqdm(total=target_count, desc=f"{query_type} ({num_conditions} conditions)")
                
                while generated < target_count and attempts < max_attempts:
                    attempts += 1
                    
                    seed_idx = random.choice(current_pool)
                    seed_sample = samples[seed_idx]
                    row_data = seed_sample.get('row_data', {})
                    column_types = seed_sample.get('column_types', {})
                    
                    # Get available numeric and categorical columns
                    numeric_cols = [c for c, t in column_types.items() 
                                   if t == 'numeric' and c in row_data and pd.notna(row_data[c])]
                    categorical_cols = [c for c, t in column_types.items() 
                                       if t == 'categorical' and c in row_data and pd.notna(row_data[c])]
                    
                    conditions = []
                    
                    try:
                        if query_type == 'numeric' and numeric_cols:
                            if len(numeric_cols) < num_conditions:
                                continue
                            for col in random.sample(numeric_cols, num_conditions):
                                if cond := self._generate_numeric_condition(col, row_data[col]):
                                    conditions.append(cond)
                        
                        elif query_type == 'categorical' and categorical_cols:
                            if len(categorical_cols) < num_conditions:
                                continue
                            for col in random.sample(categorical_cols, num_conditions):
                                if cond := self._generate_categorical_condition(col, row_data[col]):
                                    conditions.append(cond)
                        
                        elif query_type == 'mixed' and numeric_cols and categorical_cols:
                            # mixed type: randomly allocate numeric and categorical counts, at least 1 each
                            num_numeric = random.randint(1, num_conditions - 1)
                            num_categorical = num_conditions - num_numeric
                            
                            if len(numeric_cols) < num_numeric or len(categorical_cols) < num_categorical:
                                continue
                            
                            for col in random.sample(numeric_cols, num_numeric):
                                if cond := self._generate_numeric_condition(col, row_data[col]):
                                    conditions.append(cond)
                            for col in random.sample(categorical_cols, num_categorical):
                                if cond := self._generate_categorical_condition(col, row_data[col]):
                                    conditions.append(cond)
                        
                        # Ensure correct number of conditions were generated
                        if len(conditions) != num_conditions:
                            continue
                        
                        # Find matching samples (accelerated by pre-built indices)
                        matching_indices = self._find_matches_fast(conditions)
                        
                        if len(matching_indices) >= self.config.min_matches:
                            query_data = {
                                'query_id': f"retrieval_{query_type}_{total_generated + 1:06d}",
                                'query_text': self._generate_query_text(conditions),
                                'query_type': query_type,
                                'conditions': conditions,
                                'num_conditions': len(conditions),
                                'matching_indices': matching_indices,
                                'num_matches': len(matching_indices)
                            }
                            queries[query_type].append(query_data)
                            generated += 1
                            total_generated += 1
                            pbar.update(1)
                    
                    except Exception:
                        continue
                
                pbar.close()
                condition_stats[num_conditions] = generated
                
                if generated < target_count:
                    logger.warning(f"  {num_conditions}-condition queries: target {target_count}, "
                                  f"actual {generated} ({attempts} attempts)")
            
            # Output statistics for this type
            stats_str = ", ".join([f"{k}-cond:{v}" for k, v in sorted(condition_stats.items())])
            logger.info(f"Successfully generated {total_generated} {query_type} queries ({stats_str})")
        
        return queries


def build_retrieval_benchmark(
    samples: List[Dict],
    config: BenchmarkConfig,
) -> List[Dict]:
    """
    Build retrieval task benchmark
    
    Returns:
        queries: Query list (for evaluation)
    """
    logger.info("=" * 60)
    logger.info("Building Retrieval Task Benchmark")
    logger.info("=" * 60)
    
    if not samples:
        logger.warning("No samples, skipping retrieval task construction")
        return []
    
    # Check for row_data
    if 'row_data' not in samples[0]:
        logger.error("Samples missing row_data field")
        return []
    
    random.seed(config.random_seed)
    np.random.seed(config.random_seed)
    
    builder = RetrievalQueryBuilder(config.retrieval_config, config.random_seed)
    queries = builder.build_queries(samples)
    
    # Merge all queries
    all_queries = []
    for query_type in ['numeric', 'categorical', 'mixed']:
        all_queries.extend(queries[query_type])
    
    if not all_queries:
        logger.warning("No valid queries generated")
        return []
    
    logger.info(f"Total queries: {len(all_queries)}")
    
    # Build retrieval samples
    result = []
    for query in all_queries:
        result.append({
            'task': 'retrieval',
            'query_id': query['query_id'],
            'query_text': query['query_text'],
            'query_type': query['query_type'],
            'conditions': query['conditions'],
            'num_conditions': query['num_conditions'],
            'matching_indices': query['matching_indices'],  # Ground Truth
            'num_matches': query['num_matches'],
        })
    
    logger.info(f"Retrieval task: {len(result)} queries")
    
    return result


# =============================================================================
# Main Processor
# =============================================================================

class BenchmarkBuilder:
    """Benchmark builder"""
    
    def __init__(self, config: BenchmarkConfig):
        self.config = config
    
    def build(self) -> None:
        """Build the complete benchmark"""
        set_random_seed(self.config.random_seed)
        
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        # Discover all datasets
        datasets = discover_datasets(self.config.input_dir, self.config.benchmarks)
        
        if not datasets:
            logger.error("No datasets found")
            return
        
        # Classification task: organized by dataset
        self._build_classification_benchmark(datasets)
        
        # Retrieval task: merge all datasets
        self._build_retrieval_benchmark(datasets)
        
        # Save configuration
        self._save_config()
        
        logger.info("Benchmark construction complete!")
    
    def _build_classification_benchmark(self, datasets: List[DatasetInfo]) -> None:
        """Build classification task benchmark"""
        logger.info("=" * 60)
        logger.info("Building Classification Task Benchmark")
        logger.info("=" * 60)
        
        clf_dir = os.path.join(self.config.output_dir, "classification")
        os.makedirs(clf_dir, exist_ok=True)
        
        # Process datasets in parallel using multiprocessing
        max_samples = self.config.classification_config.max_samples_per_dataset
        all_results = process_datasets_parallel(
            datasets, self.config, max_samples, 
            self.config.num_workers, desc="Processing classification datasets"
        )
        
        # Statistics
        stats = {
            'total_datasets': 0,
            'total_samples': 0,
            'by_benchmark': {},
        }
        
        # Save results (sequential processing to ensure output consistency)
        for dataset_info, samples in zip(datasets, all_results):
            if not samples:
                continue
            
            # Set deterministic seed (for train/test splitting)
            file_seed = dataset_file_seed(dataset_info, self.config)
            random.seed(file_seed)
            np.random.seed(file_seed)
            
            # Check number of classes
            labels = [s['label'] for s in samples]
            unique_labels = list(set(labels))
            num_labels = len(unique_labels)
            num_samples = len(samples)
            label_ratio = num_labels / num_samples
            
            if num_labels < 2:
                logger.warning(f"Dataset {dataset_info.name} has only {num_labels} classes, skipping")
                continue
            
            # Check if too many classes (may be a regression problem)
            max_label_ratio = self.config.classification_config.max_label_ratio
            max_num_labels = self.config.classification_config.max_num_labels
            
            if num_labels > max_num_labels:
                logger.warning(f"Dataset {dataset_info.name} has too many classes ({num_labels} > {max_num_labels}), skipping")
                continue
            
            if label_ratio > max_label_ratio:
                logger.warning(f"Dataset {dataset_info.name} has too high label/sample ratio "
                             f"({num_labels}/{num_samples} = {label_ratio:.2%} > {max_label_ratio:.0%}), possibly a regression problem, skipping")
                continue
            
            # Use stratified sampling to split train/test, ensuring each class has samples
            train_ratio = self.config.classification_config.train_ratio
            
            # Group by class
            samples_by_label = {}
            for s in samples:
                label = s['label']
                if label not in samples_by_label:
                    samples_by_label[label] = []
                samples_by_label[label].append(s)
            
            # Check if each class has enough samples for splitting (at least 2)
            valid_split = True
            for label, label_samples in samples_by_label.items():
                if len(label_samples) < 2:
                    logger.warning(f"Class '{label}' in dataset {dataset_info.name} has only {len(label_samples)} samples, skipping")
                    valid_split = False
                    break
            
            if not valid_split:
                continue
            
            # Stratified sampling: split each class proportionally
            train_samples = []
            test_samples = []
            
            for label, label_samples in samples_by_label.items():
                random.shuffle(label_samples)
                split_idx = max(1, int(len(label_samples) * train_ratio))  # At least 1 for training
                # Ensure test set has at least 1 sample
                if split_idx >= len(label_samples):
                    split_idx = len(label_samples) - 1
                train_samples.extend(label_samples[:split_idx])
                test_samples.extend(label_samples[split_idx:])
            
            # Final shuffle
            random.shuffle(train_samples)
            random.shuffle(test_samples)
            
            # Save dataset
            # Use sub_benchmark to distinguish same-named datasets (e.g. grinsztajn/cat_clf/albert vs grinsztajn/num_clf/albert)
            if dataset_info.sub_benchmark:
                dataset_dir = os.path.join(clf_dir, dataset_info.benchmark, dataset_info.sub_benchmark, dataset_info.name)
            else:
                dataset_dir = os.path.join(clf_dir, dataset_info.benchmark, dataset_info.name)
            os.makedirs(dataset_dir, exist_ok=True)
            
            # Save as JSONL format (convenient for line-by-line reading)
            self._save_jsonl(train_samples, os.path.join(dataset_dir, "train.jsonl"))
            self._save_jsonl(test_samples, os.path.join(dataset_dir, "test.jsonl"))

            # Save as CSV format (convenient for tree models to read directly)
            self._save_csv(train_samples, os.path.join(dataset_dir, "train.csv"), dataset_info.target_column)
            self._save_csv(test_samples, os.path.join(dataset_dir, "test.csv"), dataset_info.target_column)
            
            # Save metadata (using actually detected class info)
            # Count class distribution in train and test sets
            train_label_counts = {}
            for s in train_samples:
                train_label_counts[s['label']] = train_label_counts.get(s['label'], 0) + 1
            test_label_counts = {}
            for s in test_samples:
                test_label_counts[s['label']] = test_label_counts.get(s['label'], 0) + 1
            
            metadata = {
                'dataset': dataset_info.name,
                'benchmark': dataset_info.benchmark,
                'sub_benchmark': dataset_info.sub_benchmark,
                'task_type': dataset_info.task_type,
                'data_type': dataset_info.data_type,
                'target_column': dataset_info.target_column,
                'label_values': unique_labels,  # Use actually detected classes
                'num_labels': len(unique_labels),
                'train_samples': len(train_samples),
                'test_samples': len(test_samples),
                'train_label_distribution': train_label_counts,
                'test_label_distribution': test_label_counts,
            }
            with open(os.path.join(dataset_dir, "metadata.json"), 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            
            # Update statistics
            stats['total_datasets'] += 1
            stats['total_samples'] += len(samples)
            
            if dataset_info.benchmark not in stats['by_benchmark']:
                stats['by_benchmark'][dataset_info.benchmark] = {'datasets': 0, 'samples': 0}
            stats['by_benchmark'][dataset_info.benchmark]['datasets'] += 1
            stats['by_benchmark'][dataset_info.benchmark]['samples'] += len(samples)
        
        # Save statistics
        with open(os.path.join(clf_dir, "stats.json"), 'w') as f:
            json.dump(stats, f, indent=2)
        
        logger.info(f"Classification task: {stats['total_datasets']} datasets, {stats['total_samples']} samples")
        for benchmark, info in stats['by_benchmark'].items():
            logger.info(f"  - {benchmark}: {info['datasets']} datasets, {info['samples']} samples")
    
    def _build_retrieval_benchmark(self, datasets: List[DatasetInfo]) -> None:
        """Build retrieval task benchmark"""
        logger.info("=" * 60)
        logger.info("Building Retrieval Task Benchmark")
        logger.info("=" * 60)
        
        ret_dir = os.path.join(self.config.output_dir, "retrieval")
        os.makedirs(ret_dir, exist_ok=True)
        
        # Max samples per dataset
        max_samples_per_dataset = self.config.retrieval_config.max_samples_per_dataset
        if max_samples_per_dataset:
            logger.info(f"Retrieval task: max {max_samples_per_dataset} samples per dataset")
        
        # Process datasets in parallel using multiprocessing
        all_results = process_datasets_parallel(
            datasets, self.config, max_samples_per_dataset,
            self.config.num_workers, desc="Loading retrieval datasets"
        )
        
        # Merge samples from all datasets
        all_samples = []
        dataset_sample_counts = {}
        
        for dataset_info, samples in zip(datasets, all_results):
            dataset_sample_counts[dataset_info.name] = len(samples)
            all_samples.extend(samples)
        
        logger.info(f"Total {len(all_samples)} samples for retrieval task (from {len(dataset_sample_counts)} valid datasets)")
        
        if not all_samples:
            logger.warning("No samples, skipping retrieval task")
            return
        
        # Build retrieval benchmark
        random.seed(self.config.random_seed)
        np.random.seed(self.config.random_seed)
        
        queries = build_retrieval_benchmark(all_samples, self.config)
        
        if queries:
            # Retain matching_indices as Ground Truth for retrieval task
            self._save_jsonl(queries, os.path.join(ret_dir, "queries.jsonl"), keep_ground_truth=True)
        
        # Save corpus (all texts)
        corpus = [{'idx': i, 'text': s['text'], 'label': s['label'], 
                   'dataset': s['dataset'], 'benchmark': s['benchmark']} 
                  for i, s in enumerate(all_samples)]
        self._save_jsonl(corpus, os.path.join(ret_dir, "corpus.jsonl"))
        
        # Save statistics
        stats = {
            'corpus_size': len(all_samples),
            'num_queries': len(queries),
            'query_type_distribution': {}
        }
        
        for item in queries:
            qt = item['query_type']
            stats['query_type_distribution'][qt] = \
                stats['query_type_distribution'].get(qt, 0) + 1
        
        with open(os.path.join(ret_dir, "stats.json"), 'w') as f:
            json.dump(stats, f, indent=2)
        
        logger.info(f"Retrieval task: {len(queries)} queries")
    
    def _save_jsonl(self, data: List[Dict], filepath: str, keep_ground_truth: bool = False) -> None:
        """
        Save as JSONL format
        
        Args:
            data: Data list
            filepath: Output file path
            keep_ground_truth: Whether to keep matching_indices (Ground Truth for retrieval)
        """
        # Large fields to remove
        exclude_keys = ['row_data', 'column_types']
        if not keep_ground_truth:
            exclude_keys.append('matching_indices')
        
        with open(filepath, 'w', encoding='utf-8') as f:
            for item in data:
                item_clean = {k: v for k, v in item.items() if k not in exclude_keys}
                f.write(json.dumps(item_clean, ensure_ascii=False) + '\n')

    def _save_csv(self, data: List[Dict], filepath: str, target_column: str) -> None:
        """
        Save as CSV format (for tree model evaluation)
        
        Args:
            data: Data list (each sample contains row_data and label)
            filepath: Output file path
            target_column: Target column name
        """
        if not data:
            return
        
        precision = self.config.serializer_config.numeric_precision
        
        def round_float(v):
            """Apply precision only to floats, keep other values as-is"""
            if isinstance(v, (float, np.floating)):
                # Keep NaN as-is
                if pd.isna(v):
                    return v
                rounded = round(float(v), precision)
                # Convert to int if it's a whole number
                if rounded == int(rounded):
                    return int(rounded)
                return rounded
            return v
        
        rows = []
        for item in data:
            row_data = item.get('row_data', {})
            label = item.get('label')
            
            # Create a row: features + target column, apply precision only to floats
            row = {k: round_float(v) for k, v in row_data.items()}
            row[target_column] = label
            rows.append(row)
        
        df = pd.DataFrame(rows)
        df.to_csv(filepath, index=False, encoding='utf-8')

    def _save_config(self) -> None:
        """Save configuration"""
        config_info = {
            'random_seed': self.config.random_seed,
            'min_samples': self.config.min_samples,
            'serializer': {
                'numeric_precision': self.config.serializer_config.numeric_precision,
                'model_max_len_tokens': self.config.serializer_config.model_max_len_tokens,
                'appx_chars_per_token': self.config.serializer_config.appx_chars_per_token,
                'max_chars': self.config.serializer_config.max_chars,
            },
            'classification': {
                'max_samples_per_dataset': self.config.classification_config.max_samples_per_dataset,
                'train_ratio': self.config.classification_config.train_ratio,
                'max_label_ratio': self.config.classification_config.max_label_ratio,
                'max_num_labels': self.config.classification_config.max_num_labels,
            },
            'retrieval': {
                'max_samples_per_dataset': self.config.retrieval_config.max_samples_per_dataset,
                'num_queries_per_type': self.config.retrieval_config.num_queries_per_type,
                'min_matches': self.config.retrieval_config.min_matches,
                'max_conditions': self.config.retrieval_config.max_conditions,
            },
        }
        
        config_path = os.path.join(self.config.output_dir, "config.json")
        with open(config_path, 'w') as f:
            json.dump(config_info, f, indent=2)
        
        logger.info(f"Config saved to {config_path}")


# =============================================================================
# Data Loading Utilities
# =============================================================================

def load_classification_benchmark(
    benchmark_dir: str,
    dataset: Optional[str] = None,
    benchmark: Optional[str] = None,
    split: str = "train",
) -> Dict[str, List[Dict]]:
    """
    Load classification task benchmark
    
    Args:
        benchmark_dir: Benchmark output directory
        dataset: Specific dataset name (None = all)
        benchmark: Specific benchmark name (None = all)
        split: "train" or "test"
    
    Returns:
        data: {dataset_name: [samples]}
    """
    clf_dir = os.path.join(benchmark_dir, "classification")
    data = {}
    
    if benchmark:
        benchmark_dirs = [os.path.join(clf_dir, benchmark)]
    else:
        benchmark_dirs = glob.glob(os.path.join(clf_dir, "*"))
    
    for bm_dir in benchmark_dirs:
        if not os.path.isdir(bm_dir):
            continue
        
        if dataset:
            dataset_dirs = [os.path.join(bm_dir, dataset)]
        else:
            dataset_dirs = glob.glob(os.path.join(bm_dir, "*"))
        
        for ds_dir in dataset_dirs:
            if not os.path.isdir(ds_dir):
                continue
            
            ds_name = os.path.basename(ds_dir)
            filepath = os.path.join(ds_dir, f"{split}.jsonl")
            
            if not os.path.exists(filepath):
                continue
            
            samples = []
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    samples.append(json.loads(line.strip()))
            
            data[ds_name] = samples
    
    return data


def load_retrieval_benchmark(
    benchmark_dir: str,
    load_corpus: bool = True,
) -> Tuple[List[Dict], Optional[List[Dict]]]:
    """
    Load retrieval task benchmark
    
    Returns:
        queries: Query list
        corpus: Corpus (if load_corpus=True)
    """
    ret_dir = os.path.join(benchmark_dir, "retrieval")
    
    queries = []
    query_path = os.path.join(ret_dir, "queries.jsonl")
    if os.path.exists(query_path):
        with open(query_path, 'r', encoding='utf-8') as f:
            for line in f:
                queries.append(json.loads(line.strip()))
    
    corpus = None
    if load_corpus:
        corpus_path = os.path.join(ret_dir, "corpus.jsonl")
        if os.path.exists(corpus_path):
            corpus = []
            with open(corpus_path, 'r', encoding='utf-8') as f:
                for line in f:
                    corpus.append(json.loads(line.strip()))
    
    return queries, corpus


# =============================================================================
# CLI Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Build Tabula-8B Embedding Benchmark")
    
    # Basic arguments
    parser.add_argument("--input_dir", "-i", type=str, required=True,
                        help="Input directory (tabula-8b-eval-suite)")
    parser.add_argument("--output_dir", "-o", type=str, required=True,
                        help="Output directory")
    
    # Data processing arguments
    parser.add_argument("--random_seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of parallel processes (default: 4)")
    
    # Dataset filtering
    parser.add_argument("--benchmarks", type=str, nargs='+', default=None,
                        help="List of benchmarks to process (default: all)")
    parser.add_argument("--min_samples", type=int, default=100,
                        help="Minimum samples per dataset (default: 100)")
    
    # Serialization arguments
    parser.add_argument("--numeric_precision", type=int, default=2,
                        help="Numeric precision (default: 2)")
    parser.add_argument("--model_max_len_tokens", type=int, default=4096,
                        help="Model max token length (default: 4096)")
    parser.add_argument("--appx_chars_per_token", type=float, default=3.5,
                        help="Approximate characters per token (default: 3.5)")
    
    # Classification task arguments
    parser.add_argument("--classification_max_samples_per_dataset", type=int, default=None,
                        help="Max samples per dataset for classification (default: None, unlimited)")
    parser.add_argument("--classification_train_ratio", type=float, default=0.7,
                        help="Classification train ratio (default: 0.8)")
    parser.add_argument("--classification_max_label_ratio", type=float, default=0.1,
                        help="Max label/sample ratio for classification, exceeding means regression (default: 0.1)")
    parser.add_argument("--classification_max_num_labels", type=int, default=50,
                        help="Max number of labels for classification (default: 50)")
    
    # Retrieval task arguments
    parser.add_argument("--retrieval_max_samples_per_dataset", type=int, default=1000,
                        help="Max samples per dataset for retrieval (default: 1000)")
    parser.add_argument("--num_queries_per_type", type=int, default=1000,
                        help="Number of queries per type (default: 1000)")
    parser.add_argument("--min_matches", type=int, default=5,
                        help="Min matches for retrieval queries (default: 5)")
    parser.add_argument("--max_conditions", type=int, default=3,
                        help="Max conditions per query (default: 3)")
    
    args = parser.parse_args()
    
    serializer_config = SerializerConfig(
        numeric_precision=args.numeric_precision,
        model_max_len_tokens=args.model_max_len_tokens,
        appx_chars_per_token=args.appx_chars_per_token,
    )
    
    classification_config = ClassificationConfig(
        max_samples_per_dataset=args.classification_max_samples_per_dataset,
        train_ratio=args.classification_train_ratio,
        max_label_ratio=args.classification_max_label_ratio,
        max_num_labels=args.classification_max_num_labels,
    )
    
    retrieval_config = RetrievalConfig(
        num_queries_per_type=args.num_queries_per_type,
        min_matches=args.min_matches,
        max_conditions=args.max_conditions,
        numeric_precision=args.numeric_precision,
        max_samples_per_dataset=args.retrieval_max_samples_per_dataset,
    )
    
    config = BenchmarkConfig(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        random_seed=args.random_seed,
        num_workers=args.num_workers,
        serializer_config=serializer_config,
        retrieval_config=retrieval_config,
        classification_config=classification_config,
        benchmarks=args.benchmarks,
        min_samples=args.min_samples,
    )
    
    builder = BenchmarkBuilder(config)
    builder.build()


if __name__ == "__main__":
    main()

