
def create_masked_lm_predictions(tokens, masked_lm_prob, max_predictions_per_seq, vocab_list):
    """Creates the predictions for the masked LM objective. This is mostly copied from the Google BERT repo, but
    with several refactors to clean it up and remove a lot of unnecessary variables."""
    cand_indices = []
    for (i, token) in enumerate(tokens):
        if token == "[CLS]" or token == "[SEP]":
            continue
        cand_indices.append(i)

    num_to_mask = min(max_predictions_per_seq,
                      max(1, int(round(len(tokens) * masked_lm_prob))))
    # print(num_to_mask)
    # print("tokens", len(tokens))
    # print("cand", len(cand_indices))
    shuffle(cand_indices)
    mask_indices = sorted(sample(cand_indices, num_to_mask))
    masked_token_labels = []
    for index in mask_indices:
        # 80% of the time, replace with [MASK]
        if random() < 0.8:
            masked_token = "[MASK]"
        else:
            # 10% of the time, keep original
            if random() < 0.5:
                masked_token = tokens[index]
            # 10% of the time, replace with random word
            else:
                masked_token = choice(vocab_list)
        masked_token_labels.append(tokens[index])
        # Once we've saved the true label for that token, we can overwrite it with the masked version
        tokens[index] = masked_token

    return tokens, mask_indices, masked_token_labels

class InputFeatures(object):
    """A single set of features of data."""
    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
def create_examples(data_path, max_seq_length, masked_lm_prob, max_predictions_per_seq, vocab_list):
    """Creates examples for the training and dev sets."""
    examples = []
    max_num_tokens = max_seq_length - 2
    fr = open(data_path, "r")
    for (i, line) in tqdm(enumerate(fr), desc="Creating Example"):
        tokens_a = line.strip("\n").split()[:max_num_tokens]
        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0 for _ in range(len(tokens_a) + 2)]
        # remove too short sample
        if len(tokens_a) < 5:
            continue
        tokens, masked_lm_positions, masked_lm_labels = create_masked_lm_predictions(
            tokens, masked_lm_prob, max_predictions_per_seq, vocab_list)
        example = {
            "tokens": tokens,
            "segment_ids": segment_ids,
            "masked_lm_positions": masked_lm_positions,
            "masked_lm_labels": masked_lm_labels}
        examples.append(example)
    fr.close()
    return examples


def convert_examples_to_features(examples, max_seq_length, tokenizer):
    features = []
    for i, example in tqdm(enumerate(examples), desc="Converting Feature"):
        tokens = example["tokens"]
        segment_ids = example["segment_ids"]
        masked_lm_positions = example["masked_lm_positions"]
        masked_lm_labels = example["masked_lm_labels"]
        assert len(tokens) == len(segment_ids) <= max_seq_length  # The preprocessed data should be already truncated
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        masked_label_ids = tokenizer.convert_tokens_to_ids(masked_lm_labels)

        input_array = np.zeros(max_seq_length, dtype=np.int)
        input_array[:len(input_ids)] = input_ids

        mask_array = np.zeros(max_seq_length, dtype=np.bool)
        mask_array[:len(input_ids)] = 1

        segment_array = np.zeros(max_seq_length, dtype=np.bool)
        segment_array[:len(segment_ids)] = segment_ids

        lm_label_array = np.full(max_seq_length, dtype=np.int, fill_value=-1)
        lm_label_array[masked_lm_positions] = masked_label_ids

        feature = InputFeatures(input_ids=input_array,
                                 input_mask=mask_array,
                                 segment_ids=segment_array,
                                 label_id=lm_label_array)
        features.append(feature)
        # if i < 10:
        #     logger.info("input_ids: %s\ninput_mask:%s\nsegment_ids:%s\nlabel_id:%s" %(input_array, mask_array, segment_array, lm_label_array))
    return features
# from __future__ import absolute_import, division, print_function

import pretraining_args as args
import csv
import logging
import os
import random
random.seed(args.seed)
import sys
from glob import glob
import numpy as np
import torch

from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
from random import random, randrange, randint, shuffle, choice, sample
from torch.nn import CrossEntropyLoss, MSELoss
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import matthews_corrcoef, f1_score

from pytorch_pretrained_bert.file_utils import  WEIGHTS_NAME, CONFIG_NAME
from pytorch_pretrained_bert.modeling import BertForMaskedLM, BertConfig
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.optimization import BertAdam
print(args.bert_config_json)
vocab_list = []
with open(args.vocab_file, 'r') as fr:
    for line in fr:
        vocab_list.append(line.strip("\n"))
tokenizer = BertTokenizer(vocab_file=args.vocab_file)

model = BertForMaskedLM(config=BertConfig.from_json_file(args.bert_config_json))
model.load_state_dict(torch.load('/home/hing/bert/Pretraining_Bert_From_Scratch/lm_smallBert/outputs/60000_pytorch_model.bin'))
for k,v in model.named_parameters():
    print(k,v)
pretrain_=BertForMaskedLM(args.bert_config_json)
eval_examples = create_examples(data_path=args.pretrain_dev_path,
                                         max_seq_length=args.max_seq_length,
                                         masked_lm_prob=args.masked_lm_prob,
                                         max_predictions_per_seq=args.max_predictions_per_seq,
                                         vocab_list=vocab_list)
eval_features = convert_examples_to_features(
                eval_examples, args.max_seq_length, tokenizer)
all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
# Run prediction for full data
eval_sampler = SequentialSampler(eval_data)
eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)
model.eval()
eval_loss = 0
nb_eval_steps = 0
# device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
device = torch.device("cpu")

model.to(device)
for input_ids, input_mask, segment_ids, label_ids in tqdm(eval_dataloader, desc="Evaluating"):
    input_ids = input_ids.to(device)
    input_mask = input_mask.to(device)
    segment_ids = segment_ids.to(device)
    label_ids = label_ids.to(device)

    with torch.no_grad():
        loss = model(input_ids, segment_ids, input_mask, label_ids)
print(loss)