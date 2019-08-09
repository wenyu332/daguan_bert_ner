# coding=utf-8


class Config(object):
    def __init__(self):
        self.label_file = './data/tag.txt'
        self.vocab = 'bert_vocab.txt'
        self.use_cuda = True
        self.gpu = 0
        self.rnn_hidden = 128
        self.bert_embedding = 384
        self.dropout1 = 0.1
        self.dropout_ratio = 0.1
        self.rnn_layer = 1
        self.lr = 2e-5
        self.checkpoint = 'result/'
        self.optim = 'Adam'
        self.load_model = False
        self.load_path = 'result'
        self.base_epoch = 100
        self.flag='submit'
    def update(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __str__(self):

        return '\n'.join(['%s:%s' % item for item in self.__dict__.items()])


if __name__ == '__main__':

    con = Config()
    con.update(gpu=1)
    # print(con.gpu)
    # print(con)
