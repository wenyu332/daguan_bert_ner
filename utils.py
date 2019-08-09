# coding=utf-8
import torch
import os
import datetime
import unicodedata


class InputFeatures(object):
    def __init__(self, input_id, label_id, input_mask):
        self.input_id = input_id
        self.label_id = label_id
        self.input_mask = input_mask


def load_vocab(vocab_file):
    """Loads a vocabulary file into a dictionary."""
    vocab = {}
    index = 0
    with open(vocab_file, "r", encoding="utf-8") as reader:
        while True:
            token = reader.readline()
            if not token:
                break
            token = token.strip()
            vocab[token] = index
            index += 1
    return vocab


def read_corpus(path, max_length, label_dic, vocab,flag=None):
    """
    :param path:数据文件路径
    :param max_length: 最大长度
    :param label_dic: 标签字典
    :return:
    """
    file = open(path, encoding='utf-8')
    content = file.readlines()
    file.close()
    result = []
    lengths = []
    if flag=='submit':
        for line in content:
            tokens = line.strip().split(' ')
            length=len(tokens)
            if len(tokens) > max_length-2:
                length=max_length-2
                tokens = tokens[0:(max_length-2)]
            lengths.append(length)
            tokens_f =['[CLS]'] + tokens + ['[SEP]']
            input_ids = [int(vocab[i]) if i in vocab else int(vocab['[UNK]']) for i in tokens_f]
            input_mask = [1] * len(input_ids)
            while len(input_ids) < max_length:
                input_ids.append(0)
                input_mask.append(0)
            assert len(input_ids) == max_length
            assert len(input_mask) == max_length
            feature = InputFeatures(input_id=input_ids, input_mask=input_mask, label_id=None)
            result.append(feature)
        return result,lengths
    else:
        for line in content:
            text, label = line.strip().split('|||')
            tokens = text.split()
            label = label.split()
            length=len(tokens)
            if len(tokens) > max_length-2:
                length=max_length-2
                tokens = tokens[0:(max_length-2)]
                label = label[0:(max_length-2)]
            lengths.append(length)
            tokens_f =['[CLS]'] + tokens + ['[SEP]']
            label_f = ["<start>"] + label + ['<eos>']
            input_ids = [int(vocab[i]) if i in vocab else int(vocab['[UNK]']) for i in tokens_f]
            label_ids = [label_dic[i] for i in label_f]
            input_mask = [1] * len(input_ids)
            while len(input_ids) < max_length:
                input_ids.append(0)
                input_mask.append(0)
                label_ids.append(label_dic['<pad>'])
            assert len(input_ids) == max_length
            assert len(input_mask) == max_length
            assert len(label_ids) == max_length
            feature = InputFeatures(input_id=input_ids, input_mask=input_mask, label_id=label_ids)
            result.append(feature)
        return result,lengths


def save_model(model, epoch,eval_f1, path='result', **kwargs):
    """
    默认保留所有模型
    :param model: 模型
    :param path: 保存路径
    :param loss: 校验损失
    :param last_loss: 最佳epoch损失
    :param kwargs: every_epoch or best_epoch
    :return:
    """
    if not os.path.exists(path):
        os.mkdir(path)
    if kwargs.get('name', None) is None:
        cur_time = datetime.datetime.now().strftime('%Y-%m-%d#%H:%M:%S')
        name ='epoch_{}'.format(epoch)+'_'+str(eval_f1)[2:6]
        full_name = os.path.join(path, name)
        torch.save(model.state_dict(), full_name)
        print('Saved model at epoch {} successfully'.format(epoch))
        with open('{}/checkpoint'.format(path), 'w') as file:
            file.write(name)
            print('Write to checkpoint')


def load_model(model, path='result/8918', **kwargs):
    if kwargs.get('name', None) is None:
        with open('{}/checkpoint'.format(path)) as file:
            content = file.read().strip()
            name = os.path.join(path, content)
    else:
        name=kwargs['name']
        name = os.path.join(path,name)
    model.load_state_dict(torch.load(name, map_location=lambda storage, loc: storage))
    print('load model {} successfully'.format(name))
    return model

def data_split():
    def train_count():
        #16941 17000 0.9965294117647059
        f=open('./data/normal_daguan_train.txt')
        datas=f.readlines()
        f.close()
        count=0
        for data in datas:
            words=data.split('|||')[0].split(' ')
            if len(words)<=256:
                count+=1
        print(count,len(datas),count/len(datas))
    def test_count():
        #2989 3000 0.9963333333333333
        f=open('./data/normal_daguan_test.txt')
        datas=f.readlines()
        f.close()
        count=0
        for data in datas:
            words=data.split(' ')
            if len(words)<=256:
                count+=1
        print(count,len(datas),count/len(datas))

    def split():
        # 16941 17000 0.9965294117647059
        f = open('./data/normal_daguan_train.txt')
        datas = f.readlines()
        f.close()
        f = open('./data/train.txt','w')
        for data in datas[:14000]:
            f.write(data)
        f.close()
        f = open('./data/dev.txt', 'w')
        for data in datas[14000:15500]:
            f.write(data)
        f.close()
        f = open('./data/test.txt', 'w')
        for data in datas[15500:]:
            f.write(data)
        f.close()
    train_count()
    test_count()
    # split()
data_split()
def get_ner_fmeasure(golden_lists, predict_lists, label_type="BMES"):
    sent_num = len(golden_lists)
    golden_full = []
    predict_full = []
    right_full = []
    right_tag = 0
    all_tag = 0
    for idx in range(0,sent_num):
        # word_list = sentence_lists[idx]
        golden_list = golden_lists[idx]
        predict_list = predict_lists[idx]
        for idy in range(len(golden_list)):
            if golden_list[idy] == predict_list[idy]:
                right_tag += 1
        all_tag += len(golden_list)
        if label_type == "BMES":
            gold_matrix = get_ner_BMES(golden_list)
            pred_matrix = get_ner_BMES(predict_list)
        else:
            gold_matrix = get_ner_BIO(golden_list)
            pred_matrix = get_ner_BIO(predict_list)
        # print "gold", gold_matrix
        # print "pred", pred_matrix
        right_ner = list(set(gold_matrix).intersection(set(pred_matrix)))
        golden_full += gold_matrix
        predict_full += pred_matrix
        right_full += right_ner
    right_num = len(right_full)
    golden_num = len(golden_full)
    predict_num = len(predict_full)
    if predict_num == 0:
        precision = -1
    else:
        precision =  (right_num+0.0)/predict_num
    if golden_num == 0:
        recall = -1
    else:
        recall = (right_num+0.0)/golden_num
    if (precision == -1) or (recall == -1) or (precision+recall) <= 0.:
        f_measure = -1
    else:
        f_measure = 2*precision*recall/(precision+recall)
    accuracy = (right_tag+0.0)/all_tag
    # print "Accuracy: ", right_tag,"/",all_tag,"=",accuracy
    print ("gold_num = ", golden_num, " pred_num = ", predict_num, " right_num = ", right_num)
    # print('accuracy=',accuracy,' precision=',precision,' recall=',recall,' f_measure=',f_measure)
    print('acc=',round(accuracy,4),' p=',round(precision,4),' r=',round(recall,4),' f1=',round(f_measure,4))
    return accuracy, precision, recall, f_measure


def get_ner_BMES(label_list):
    # list_len = len(word_list)
    # assert(list_len == len(label_list)), "word list size unmatch with label list"
    list_len = len(label_list)
    begin_label = 'B-'
    end_label = 'E-'
    single_label = 'S-'
    whole_tag = ''
    index_tag = ''
    tag_list = []
    stand_matrix = []
    for i in range(0, list_len):
        # wordlabel = word_list[i]
        current_label = label_list[i].upper()
        if begin_label in current_label:
            if index_tag != '':
                tag_list.append(whole_tag + ',' + str(i - 1))
            whole_tag = current_label.replace(begin_label, "", 1) + '[' + str(i)
            index_tag = current_label.replace(begin_label, "", 1)

        elif single_label in current_label:
            if index_tag != '':
                tag_list.append(whole_tag + ',' + str(i - 1))
            whole_tag = current_label.replace(single_label, "", 1) + '[' + str(i)
            tag_list.append(whole_tag)
            whole_tag = ""
            index_tag = ""
        elif end_label in current_label:
            if index_tag != '':
                tag_list.append(whole_tag + ',' + str(i))
            whole_tag = ''
            index_tag = ''
        else:
            continue
    if (whole_tag != '') & (index_tag != ''):
        tag_list.append(whole_tag)
    tag_list_len = len(tag_list)

    for i in range(0, tag_list_len):
        if len(tag_list[i]) > 0:
            tag_list[i] = tag_list[i] + ']'
            insert_list = reverse_style(tag_list[i])
            stand_matrix.append(insert_list)
    # print stand_matrix
    return stand_matrix
def get_ner_BIO(label_list):
    # list_len = len(word_list)
    # assert(list_len == len(label_list)), "word list size unmatch with label list"
    list_len = len(label_list)
    begin_label = 'B-'
    inside_label = 'I-'
    whole_tag = ''
    index_tag = ''
    tag_list = []
    stand_matrix = []
    for i in range(0, list_len):
        # wordlabel = word_list[i]
        current_label = label_list[i].upper()
        if begin_label in current_label:
            if index_tag == '':
                whole_tag = current_label.replace(begin_label,"",1) +'[' +str(i)
                index_tag = current_label.replace(begin_label,"",1)
            else:
                tag_list.append(whole_tag + ',' + str(i-1))
                whole_tag = current_label.replace(begin_label,"",1)  + '[' + str(i)
                index_tag = current_label.replace(begin_label,"",1)

        elif inside_label in current_label:
            if current_label.replace(inside_label,"",1) == index_tag:
                whole_tag = whole_tag
            else:
                if (whole_tag != '')&(index_tag != ''):
                    tag_list.append(whole_tag +',' + str(i-1))
                whole_tag = ''
                index_tag = ''
        else:
            if (whole_tag != '')&(index_tag != ''):
                tag_list.append(whole_tag +',' + str(i-1))
            whole_tag = ''
            index_tag = ''

    if (whole_tag != '')&(index_tag != ''):
        tag_list.append(whole_tag)
    tag_list_len = len(tag_list)

    for i in range(0, tag_list_len):
        if  len(tag_list[i]) > 0:
            tag_list[i] = tag_list[i]+ ']'
            insert_list = reverse_style(tag_list[i])
            stand_matrix.append(insert_list)
    return stand_matrix
def reverse_style(input_string):
    target_position = input_string.index('[')
    input_len = len(input_string)
    output_string = input_string[target_position:input_len] + input_string[0:target_position]
    return output_string