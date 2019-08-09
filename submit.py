def submitFormat():
    f=open('predict.txt')
    labels=f.readlines()
    f.close()
    f=open('./data/daguan_test.txt')
    texts=f.readlines()
    f.close()
    f=open('submit.txt','w',encoding='utf8')
    for i,label in enumerate(labels):

        label=label.strip().split(' ')
        text=texts[i].strip().split('_')
        if len(label)!=len(text):
            print(i,len(label),len(text))
        length=len(label)
        start=0
        end=0
        result=[]
        print(i,label)
        while length>0:
            if 'M' in label[end]:
                label[end]=label[end][:2]+'B'
                continue
            if label[end] == 'o':
                while label[end] == 'o':
                    end+=1
                    length-=1
                    if length==0:
                        break
                result.append('_'.join(text[start:end])+'/o')
                start=end
                if start==len(label):
                    break
            if 'S' in label[end]:
                # print(text[start:end] +'/'+label[end][0])
                result.append('_'.join(text[start:end+1]) +'/'+label[end][0])
                end+=1
                length-=1
                start=end
                if length == 0:
                    break
            if 'B' in label[end]:
                while 'E' not in label[end]:
                    end+=1
                    length-=1
                    if length==0:
                        break
                end += 1
                length -= 1
                result.append('_'.join(text[start:end])+'/'+label[start][0])
                start=end
                if start==len(label):
                    break
            # if '' in label[end]:
        f.write('  '.join(result)+'\n')
    f.close()
submitFormat()
# import torch
# model=torch.load('./ckpts/bert_weight.bin')
# for k,v in model.items():
#     print(k,v)