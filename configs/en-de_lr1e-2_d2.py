SEED = 0
RESULT_FILE = 'en-de_result.tsv'
METRICS_FILE = 'en-de_metrics.txt'
BEST_MODEL_FILE = 'en-de_best_model.bin'
MODEL_TYPE = 'xlmroberta'
MODEL_NAME = 'xlm-roberta-large'
TRAIN_DATA = 'en-de/train.ende.df.short.tsv'
DEV_DATA = 'en-de/dev.ende.df.short.tsv'
TEST_DATA = 'en-de/test20.ende.df.short.tsv'
DATA_DIR = './data/'
OUTPUT_DIR = './results/'
BEST_MODEL_DIR = './best_models/'

# Model parameters
args = {
    'max_seq_length': 128,
    'train_batch_size': 8,
    'gradient_accumulation_steps': 1,
    'eval_batch_size': 8,
    'num_train_epochs': 3,
    'weight_decay': 0,
    'learning_rate': 1e-2,
    'adam_epsilon': 1e-8,
    'warmup_ratio': 0.1,
    'warmup_steps': 0,
    'max_grad_norm': 1.0,
    'dropout': 0.2
}