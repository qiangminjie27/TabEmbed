python src/build_benchmark.py \
    -i /path/to/tabula-8b-eval-suite \
    -o /path/to/benchmark \
    --num_workers 64 \
    --classification_max_samples_per_dataset 10000 \
    --classification_train_ratio 0.9 \
    --retrieval_max_samples_per_dataset 10000 \
    --num_queries_per_type 10000