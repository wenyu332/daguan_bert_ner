# daguan_bert_ner
bert的模型传到:https://pan.baidu.com/s/149fOvlaIkMrdtvsW3KMsXQ
下载之后放到'./ckpts'目录下即可进行训练
预训练模型的数据我也上传到：https://pan.baidu.com/s/1fMkZSiKXkRjdXAesYwDBzg，
且提供了两个辅助脚本，在temp.py中，create_bert_datas（）方法为创建不同的训练数据，model_transfer（）方法为删除语言模型训练的部分权重而只保留命名实体识别所需的部分模型权重
