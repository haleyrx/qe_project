# Model parameters
SEED = 777
RESULT_FILE = "en-de_result.tsv"
METRICS_FILE = "en-de_metrics.txt"
BEST_MODEL_FILE = "en-de_best_model.bin"
MODEL_TYPE = "xlmroberta"
MODEL_NAME = "xlm-roberta-large"

args = {
    'output_dir': './results/',
    'best_model_dir': './best_models/',

    'max_seq_length': 128,
    'train_batch_size': 8,
    'gradient_accumulation_steps': 1,
    'eval_batch_size': 8,
    'num_train_epochs': 3,
    'weight_decay': 0,
    'learning_rate': 1e-5,
    'adam_epsilon': 1e-8,
    'warmup_ratio': 0.1,
    'warmup_steps': 0,
    'max_grad_norm': 1.0,
    "manual_seed": SEED,

}