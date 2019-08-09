# coding=utf-8
import torch
import torch.nn as nn
from torch.autograd import Variable
from config import Config
from model import BERT_LSTM_CRF
import torch.optim as optim
from utils import load_vocab, read_corpus, load_model, save_model,get_ner_fmeasure
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import pretraining_args as args
from pytorch_pretrained_bert.optimization import BertAdam
def train(**kwargs):
    config = Config()
    config.update(**kwargs)
    print('当前设置为:\n', config)
    if args.use_cuda:
        torch.cuda.set_device(config.gpu)
    print('loading corpus')
    vocab = load_vocab(args.vocab_file)
    label_dic = load_vocab(config.label_file)
    index2label={v:k for k,v in label_dic.items()}
    tagset_size = len(label_dic)
    train_data,_ = read_corpus(args.pretrain_train_path, max_length=args.max_seq_length, label_dic=label_dic, vocab=vocab)
    dev_data,dev_len = read_corpus(args.pretrain_dev_path, max_length=args.max_seq_length, label_dic=label_dic, vocab=vocab)
    num_train_optimization_steps = int(
        len(train_data) / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs
    if args.local_rank != -1:
        num_train_optimization_steps = num_train_optimization_steps // torch.distributed.get_world_size()

    train_ids = torch.LongTensor([temp.input_id for temp in train_data])
    train_masks = torch.LongTensor([temp.input_mask for temp in train_data])
    train_tags = torch.LongTensor([temp.label_id for temp in train_data])

    train_dataset = TensorDataset(train_ids, train_masks, train_tags)
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=args.train_batch_size)

    dev_ids = torch.LongTensor([temp.input_id for temp in dev_data])
    dev_masks = torch.LongTensor([temp.input_mask for temp in dev_data])
    dev_tags = torch.LongTensor([temp.label_id for temp in dev_data])
    dev_dataset = TensorDataset(dev_ids, dev_masks, dev_tags)
    dev_loader = DataLoader(dev_dataset, shuffle=False, batch_size=args.eval_batch_size)
    model = BERT_LSTM_CRF(args, tagset_size, config.bert_embedding, config.rnn_hidden, config.rnn_layer, dropout_ratio=config.dropout_ratio, dropout1=config.dropout1, use_cuda=config.use_cuda)

    if config.use_cuda:
        model=model.cuda()
    if config.load_model:
        if config.flag=='submit':
            assert config.load_path is not None
            test_data, test_len = read_corpus(args.submit_test_path, max_length=args.max_seq_length, label_dic=label_dic,
                                              vocab=vocab,flag='submit')
            test_ids = torch.LongTensor([temp.input_id for temp in test_data])
            test_masks = torch.LongTensor([temp.input_mask for temp in test_data])
            test_dataset = TensorDataset(test_ids, test_masks)
            test_loader = DataLoader(test_dataset, shuffle=False, batch_size=args.eval_batch_size)
            model = load_model(model, name=None)
            test(model, test_loader, config, index2label, test_len)
            # dev(model, test_loader, None, config)
        if config.flag=='test':
            assert config.load_path is not None
            test_data, test_len = read_corpus(args.pretrain_test_path, max_length=args.max_seq_length, label_dic=label_dic,
                                              vocab=vocab)
            test_ids = torch.LongTensor([temp.input_id for temp in test_data])
            test_masks = torch.LongTensor([temp.input_mask for temp in test_data])
            test_tags = torch.LongTensor([temp.label_id for temp in test_data])
            test_dataset = TensorDataset(test_ids, test_masks, test_tags)
            test_loader = DataLoader(test_dataset, shuffle=False, batch_size=args.eval_batch_size)
            model = load_model(model, name=None)
            # test(model, test_loader, config, index2label, test_len)
            dev(model, test_loader, 0, config, index2label, dev_len)

    else:
    # print(model)
        model.train()
        bert_param_optimizer = list(model.word_embeds.named_parameters())
        lstm_param_optimizer = list(model.lstm.named_parameters())
        liner_param_optimizer = list(model.liner.named_parameters())
        crf_param_optimizer = list(model.crf.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in bert_param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01,'lr':config.lr},
            {'params': [p for n, p in bert_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,'lr':config.lr},
            {'params': [p for n, p in lstm_param_optimizer if not any(nd in n for nd in no_decay)],'weight_decay': 0.001, 'lr': config.lr*5},
            {'params': [p for n, p in lstm_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,'lr': config.lr*5},
            {'params': [p for n, p in liner_param_optimizer if not any(nd in n for nd in no_decay)],'weight_decay': 0.001, 'lr': config.lr * 2},
            {'params': [p for n, p in liner_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0, 'lr':config.lr * 2},
            {'params': [p for n, p in crf_param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.001, 'lr': config.lr * 3},
            {'params': [p for n, p in crf_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,'lr': config.lr * 3},

        ]
        # print(optimizer_grouped_parameters)
        optimizer = BertAdam(optimizer_grouped_parameters,
                         # lr=config.lr,
                         warmup=args.warmup_proportion,
                         t_total=num_train_optimization_steps)
        # optimizer = optimizer(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
        eval_f1 = 0.0
        for epoch in range(config.base_epoch):
            print(optimizer.get_lr())
            step = 0
            for i, batch in enumerate(train_loader):
                step += 1
                model.zero_grad()
                inputs, masks, tags = batch
                inputs, masks, tags = Variable(inputs), Variable(masks), Variable(tags)
                if config.use_cuda:
                    inputs, masks, tags = inputs.cuda(), masks.cuda(), tags.cuda()

                feats = model(inputs, masks)
                loss = model.loss(feats, masks,tags)
                loss.backward()
                optimizer.step()
                if step % 50 == 0:
                    print('step: {} |  epoch: {}|  loss: {}'.format(step, epoch, loss.item()))
            f_measure = dev(model, dev_loader, epoch, config,index2label,dev_len)
            if eval_f1 < f_measure:
                eval_f1=f_measure
                save_model(model,epoch,f_measure)


def dev(model, dev_loader, epoch, config,index2label,dev_lens):
    model.eval()
    eval_loss = 0
    trues = []
    preds = []
    length = 0
    for i, batch in enumerate(dev_loader):
        inputs, masks, tags = batch
        length += inputs.size(0)
        inputs, masks, tags = Variable(inputs), Variable(masks), Variable(tags)
        if config.use_cuda:
            inputs, masks, tags = inputs.cuda(), masks.cuda(), tags.cuda()
        feats = model(inputs, masks)
        path_score, best_path = model.crf(feats, masks.byte())
        loss = model.loss(feats, masks, tags)
        eval_loss += loss.item()
        nums=len(best_path)
        for i in range(nums):
            pred_result=[]
            true_result=[]
            for index in list(best_path[i].cpu().numpy()):
                pred_result.append(index2label[index])
            for index in list(tags[i].cpu().numpy()):
                true_result.append(index2label[index])
            preds.append(pred_result)
            trues.append(true_result)
    # print(len(dev_lens))
    pred_tag_lists = [preds[i][1:dev_lens[i]+1] for i in range(len(dev_lens))]
    tag_lists = [trues[i][1:dev_lens[i]+1] for i in range(len(dev_lens))]
    accuracy, precision, recall, f_measure=get_ner_fmeasure(tag_lists,pred_tag_lists)
    def calculate_category_f1():
        print(pred_tag_lists[:25])
        print(tag_lists[:25])
        labels=[v for k,v in index2label.items()]
        truth_label_count={}
        predict_label_count = {}
        label_count={}
        count=0
        for pred,true in zip(preds,trues):
            for i,t in enumerate(true):
                if t=='<eos>' and pred[i]=='<eos>':
                    count=count+1
                    break
                else:
                    if t not in ['<pad>', 'o', '<start>']:
                        if t==pred[i]:
                            if t not in label_count:
                                label_count[t]=1
                            else:
                                label_count[t] +=1
                        if t not in truth_label_count:
                            truth_label_count[t]=1
                        else:
                            truth_label_count[t]+=1
                        if pred[i] not in predict_label_count:
                            predict_label_count[pred[i]]=1
                        else:
                            predict_label_count[pred[i]] += 1
        precision={}
        recall={}
        f1={}
        # print(label_count.keys())
        # print(predict_label_count.keys())
        # print(truth_label_count.keys())
        for label in labels:
            if label in label_count:
                precision[label]=label_count[label]/predict_label_count[label]
                recall[label]=label_count[label]/truth_label_count[label]
                f1[label]=2*precision[label]*recall[label]/(precision[label]+recall[label])

        # print(sum(precision.values())/len(truth_label_count))
        # print(sum(recall.values())/len(truth_label_count))

        print(precision)
        print(recall)
        print(f1)
    # print(truth_label_count)
    print('eval  epoch: {}|  loss: {}'.format(epoch, eval_loss/length))
    model.train()
    return f_measure
def test(model, test_loader, config,index2label,dev_lens):
    model.eval()
    preds = []
    length = 0
    for i, batch in enumerate(test_loader):
        inputs, masks = batch
        length += inputs.size(0)
        inputs, masks = Variable(inputs), Variable(masks)
        if config.use_cuda:
            inputs, masks = inputs.cuda(), masks.cuda()
        feats = model(inputs, masks)
        path_score, best_path = model.crf(feats, masks.byte())
        nums=len(best_path)
        for i in range(nums):
            pred_result=[]
            for index in list(best_path[i].cpu().numpy()):
                pred_result.append(index2label[index])
            preds.append(pred_result)
    # print(len(dev_lens))
    pred_tag_lists = [preds[i][1:dev_lens[i]+1] for i in range(len(dev_lens))]
    f=open('predict.txt','w')
    for pred_tag_list in pred_tag_lists:
        f.write(' '.join(pred_tag_list) + '\n')
    f.close()

if __name__ == '__main__':
    train()










