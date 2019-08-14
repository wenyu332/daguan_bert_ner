# -----------ARGS---------------------
pretrain_train_path = "/home/hing/bert/Pretraining_Bert_From_Scratch/lm_smallBert/data/pretrain_train.txt"
pretrain_dev_path = "/home/hing/bert/Pretraining_Bert_From_Scratch/lm_smallBert/data/pretrain_dev.txt"

max_seq_length = 200
do_train = True
do_lower_case = True
train_batch_size = 20
eval_batch_size = 16
learning_rate = 2e-5
num_train_epochs = 100
warmup_proportion = 0.1
no_cuda = False
local_rank = -1
seed = 42
gradient_accumulation_steps = 1
fp16 = False
loss_scale = 0.
bert_config_json = "/home/hing/bert/Pretraining_Bert_From_Scratch/lm_smallBert/bert_config.json"
vocab_file = "/home/hing/bert/Pretraining_Bert_From_Scratch/lm_smallBert/bert_vocab.txt"
output_dir = "outputs"
masked_lm_prob = 0.15
max_predictions_per_seq = 20