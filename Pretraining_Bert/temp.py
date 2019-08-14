def create_bert_datas():
    f=open('/home/hing/Desktop/named_entity_recognition/language_model/corpus_no_space.txt','r')
    datas=f.readlines()
    f.close()
    length=[]
    f = open('pretrain_train.txt', 'w')
    for data in datas:
        length.append(len(data.strip().split()))
    f.close()
    length=sorted(length)
    count=0
    for len_ in length:
        if len_<=156:
            count+=1
    print(count)
    print(len(length))
    print(count/len(length))
    # f=open('vocab.txt','w')
    # vocabs=set()
    # for data in datas:
    #     words=data.strip().split()
    #     for word in words:
    #         vocabs.add(word)
    # for vocab in vocabs:
    #     f.write(vocab+'\n')
    # f.close()
    # # f = open('/home/hing/Desktop/named_entity_recognition/normal_daguan_test.txt', 'r')
    # # datas = f.readlines()
    # # f.close()
    # f = open('pretrain_dev.txt', 'w')
    # for data in datas[-100000:]:
    #     f.write(data)
    # f.close()
# create_bert_datas()
# f = open('/home/hing/bert/Pretraining_Bert_From_Scratch/lm_smallBert/data/pretrain_dev.txt',)
# datas=f.readlines()
# f.close()
# print(len(datas))
# f = open('/home/hing/bert/Pretraining_Bert_From_Scratch/lm_smallBert/data/pretrain_train.txt',)
# datas=f.readlines()
# f.close()
# print(len(datas))
from pytorch_pretrained_bert.modeling import BertForMaskedLM, BertConfig
import torch
import lm_smallBert.pretraining_args as args
from pytorch_pretrained_bert.modeling import BertModel
def model_transfer():
    model = BertForMaskedLM(config=BertConfig.from_json_file(args.bert_config_json))
    # print('language_model',model.state_dict()['bert.embeddings.word_embeddings.weight'])
    # print('language_model',model.state_dict()['bert.embeddings.LayerNorm.weight'])
    # print('language_model',model.state_dict()['bert.encoder.layer.0.attention.self.key.weight'])
    model = model.bert
    # print('bert_model',model.state_dict()['embeddings.word_embeddings.weight'])
    # print('bert_model',model.state_dict()['embeddings.LayerNorm.weight'])
    # print('bert_model',model.state_dict()['encoder.layer.0.attention.self.key.weight'])
    model_dict = model.state_dict()
    lm_dict = torch.load('./lm_smallBert/outputs/1.41_150000_step')
    for k,v in lm_dict.items():
        print(k,v)
    # print('lm_dict',lm_dict['bert.embeddings.word_embeddings.weight'])
    # print('lm_dict',lm_dict['bert.embeddings.LayerNorm.weight'])
    # print('lm_dict',lm_dict['bert.encoder.layer.0.attention.self.key.weight'])
    pretrained_dict = {k[5:]: v for k, v in lm_dict.items() if k[5:] in model_dict.keys()}
    # print('pretrained_dict',pretrained_dict)
    model.load_state_dict(pretrained_dict)
    torch.save(model.state_dict(),'1.41_bert_weight.bin')
model_transfer()
# bert_model=BertModel(config=BertConfig.from_json_file(args.bert_config_json))
# # bert_model_weight=torch.load('bert_weight.bin')
# # print(bert_model)
# # bert_model.load_state_dict(bert_model_weight)
# for k,v in bert_model.named_parameters():
#     print(k,v)
# lm_dict = torch.load('./lm_smallBert/outputs/60000_pytorch_model.bin')
# for k, v in lm_dict.items():
#     print(k, v)