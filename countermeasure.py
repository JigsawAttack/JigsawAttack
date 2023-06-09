import numpy as np
import random
from tqdm import tqdm
def padding_seal(real_query_d,n=4):
    #print("padding")
    if n==0 or n==1:
        return real_query_d
    query_number = len(real_query_d)
    
    padded_query_d = np.zeros((len(real_query_d),0))
    for i in range(query_number):
        s = int(np.sum(real_query_d[i]))
        power = 0
        while(1):
            if s > (n**power):
                power += 1
            else:
                padding_number = (n**power)-s
                break
        
        if padding_number > len(padded_query_d[0]):
            pad_doc = np.zeros((query_number, padding_number - len(padded_query_d[0])))
            padded_query_d = np.hstack((padded_query_d,pad_doc))
            temp = np.ones((1,padding_number))
            padded_query_d[i] = temp
        else:
            temp = np.zeros(len(padded_query_d[0]))
            index = np.where(temp==0)[0]
            
            pad_index = np.random.choice(index,padding_number,replace=False)
            temp[pad_index] = 1
            padded_query_d[i] = padded_query_d[i] + temp
    
    
    padded_query_d = np.hstack((real_query_d,padded_query_d))
    
    return padded_query_d

def padding_linear(real_query_d,n=500):
    # real doc as padding docs
    # print("padding")
    if n==0 :
        return real_query_d
    query_number = len(real_query_d)
    padded_query_d = real_query_d.copy()
    for i in range(query_number):
        s = int(np.sum(real_query_d[i]))
        power = 0
        while(1):
            if s > (n*power):
                power += 1
            else:
                padding_number = (n*power)-s
                break
        
        if (s + padding_number) > len(padded_query_d[0]):
            pad_doc = np.zeros((query_number, padding_number + s - len(padded_query_d[0])))
            padded_query_d = np.hstack((padded_query_d,pad_doc))
            temp = np.ones((1,padding_number + s))
            padded_query_d[i] = temp
        else:
            temp = np.zeros(len(padded_query_d[0]))
            index = np.where(temp==0)[0]
            
            pad_index = np.random.choice(index,padding_number,replace=False)
            temp[pad_index] = 1
            padded_query_d[i] = padded_query_d[i] + temp
    return padded_query_d


def padding_linear_2(real_query_d,n=500):
# generated doc as padding docs
    
    if n==0 :
        return real_query_d
    query_number = len(real_query_d)
    
    padded_query_d = np.zeros((len(real_query_d),0))
    for i in range(query_number):
        s = int(np.sum(real_query_d[i]))
        power = 0
        while(1):
            if s > (n*power):
                power += 1
            else:
                padding_number = (n*power)-s
                break
        if padding_number > len(padded_query_d[0]):
            pad_doc = np.zeros((query_number, padding_number - len(padded_query_d[0])))
            padded_query_d = np.hstack((padded_query_d,pad_doc))
            temp = np.ones((1,padding_number))
            padded_query_d[i] = temp
        else:
            temp = np.zeros(len(padded_query_d[0]))
            index = np.where(temp==0)[0]
            
            pad_index = np.random.choice(index,padding_number,replace=False)
            temp[pad_index] = 1
            padded_query_d[i] = padded_query_d[i] + temp
    #print(np.sum(padded_query_d)/len(padded_query_d),len(padded_query_d),np.sum(padded_query_d))
    padded_query_d = np.hstack((real_query_d,padded_query_d))
    return padded_query_d

def padding_cluster(real_query_d,knum_in_cluster=2):
    # generated doc as padding docs
    if knum_in_cluster == 1:
        return real_query_d
    v = np.sum(real_query_d,axis=1)
    query_number = len(real_query_d)
    index = [i for i in range(query_number)]
    id_v = list(zip(index,v))
    id_v = sorted(id_v,key=lambda x:x[1])
    i = 0
    padding_number = np.zeros((query_number,1))
    while i < query_number:
        if i < (query_number//knum_in_cluster)*knum_in_cluster:
            padding_number[i] = id_v[i-i%knum_in_cluster+knum_in_cluster-1][1] - id_v[i][1]
        else:
            padding_number[i] = id_v[-1][1] - id_v[i][1]
        i = i+1
    padded_query_d = np.zeros((len(real_query_d),0))
    for i in range(query_number):
        id = id_v[i][0]
        number = int(padding_number[i])
        if number > len(padded_query_d[0]):
            pad_doc = np.zeros((query_number, number - len(padded_query_d[0])))
            padded_query_d = np.hstack((padded_query_d,pad_doc))
            temp = np.ones((1,number))
            padded_query_d[id] = temp
        else:
            temp = np.zeros(len(padded_query_d[0]))
            index = np.where(temp==0)[0]
            pad_index = np.random.choice(index,number,replace=False)
            temp[pad_index] = 1
            padded_query_d[id] = padded_query_d[id] + temp
    padded_query_d = np.hstack((real_query_d,padded_query_d))
    # for i in range(query_number):
    #     print(i,id_v[i][1],np.sum(padded_query_d,axis=1)[id_v[i][0]])
    return padded_query_d



def obfuscate(real_query_d, p=0.8703, q=0.004416, m = 6):
    if q == 0:
        return real_query_d
    temp = real_query_d
    for i in range(m-1):
        temp = np.hstack((temp,real_query_d))
    for i in range(len(temp)):
        for j in range(len(temp[0])):
            if temp[i][j]==1:
                if random.random()>p:
                    temp[i][j] = 0
            else:
                if random.random()<q:
                    temp[i][j] = 1
    return temp


### apply different defenses in four quadrants
def apply_defense(real_query_d,defense):
    if defense["method"] == "P":
        return padding_linear(real_query_d,defense["k"])
    elif defense["method"] == "o":
        return obfuscate(real_query_d,defense["p"],defense["q"],defense["m"])
    else:
        return real_query_d
def mix_defenses(real_query_d,high_f,high_v,real_F,real_V,defense=[None,None,None,None]):
    freq_line = sorted(real_F,reverse=True)[int(len(real_F)* high_f)]
    vol_line = sorted(real_V,reverse=True)[int(len(real_V)* high_v)]
    divided_query_d = [[],[],[],[]]
    for i in range(len(real_V)):
        if real_V[i]>=vol_line:
            if real_F[i]>=freq_line:
                divided_query_d[3].append(real_query_d[i])
            else:
                divided_query_d[2].append(real_query_d[i])
        else:
            if real_F[i]>=freq_line:
                divided_query_d[1].append(real_query_d[i])
            else:
                divided_query_d[0].append(real_query_d[i])
    temp = []
    for i in range(4):
        temp.append(apply_defense(np.array(divided_query_d[i]),defense[i]))
    return np.vstack((temp[0],temp[1],temp[2],temp[3]))