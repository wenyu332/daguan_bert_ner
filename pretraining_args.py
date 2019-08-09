# -----------ARGS---------------------
pretrain_train_path = "./data/train.txt"
pretrain_dev_path = "./data/dev.txt"
pretrain_test_path = "./data/test.txt"
submit_test_path = "./data/normal_daguan_test.txt"


max_seq_length = 200
do_train = True
do_lower_case = True
train_batch_size = 16
eval_batch_size = 16
learning_rate = 2e-4
num_train_epochs = 100
warmup_proportion = 0.1
use_cuda = True
local_rank = -1
seed = 42
gradient_accumulation_steps = 1
bert_config_json = "./bert_config.json"
vocab_file = "./bert_vocab.txt"
output_dir = "outputs"
masked_lm_prob = 0.15
max_predictions_per_seq = 20
weight='ckpts/bert_weight.bin'