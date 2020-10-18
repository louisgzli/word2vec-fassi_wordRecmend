import faiss
import pandas as pd
import numpy as np
import json
index = faiss.read_index("./index.ivf")
embedding = pd.read_csv("./word_embedding.txt",sep="\s",skiprows = 1)


with open('./jsonFile.json', 'r') as f:
    kw2id = json.load(fp=f)
id2kw = {}
for kw,idx in kw2id.items():

    id2kw[int(idx)] = kw
#



k = 10
test_index = [3,4,5,6]
x = np.random.randint(1,20000,size=100)
x = x.tolist()
test_index.extend(x)
temp_index = np.array(test_index)-1
emb = embedding.iloc[temp_index,1:]
emb = emb.values
emb = emb.copy(order="C")#转 C-contiguous
emb = emb.astype(np.float32) #embedding 使用32位浮点数
print(emb.flags) #可以打印numpy状态
D,I = index.search(emb,k)
def show_key_expand(test_intdex,expansion,id2k):
    '''
    :param test_intdex: 测试的词的index(相对于embedding 文件的位移)
    :param expansion: 二维数组，对应了
    :param id2k:id和keyword 的映射
    :return:
    '''

    for i_index,i_items in enumerate(expansion):
        print("keyword: >>> ",id2k[test_index[i_index]],"\n")
        print("expansion word: ",end=">>> ")
        print("\n")
        for j in i_items:
            print(id2k[j],end=" ")

show_key_expand(test_index,I,id2kw)



