# daguan_bert_ner
bert的模型传到:https://pan.baidu.com/s/1anf4K_nv-LhTeMnV69oBCw
下载之后放到'./ckpts'目录下即可进行训练


语言模型的代码在Pretraining_Bert文件中，
预训练模型的数据我也上传到：https://pan.baidu.com/s/19mnQ9lDvQhBniBU5SBVN9Q ，下载之后将其放到lm_bert目录下即可
temp.py代码提供了两个辅助脚本：
  （1）create_bert_datas（）方法为创建不同的训练数据，
  （2）model_transfer（）方法为删除语言模型训练的部分权重而只保留命名实体识别所需的部分模型权重
将转换之后的模型放到'./ckpts'目录下即可按照自己训练的权重进行实体识别训练
