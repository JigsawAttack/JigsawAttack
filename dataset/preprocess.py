import pickle
import time
from tqdm import tqdm

import numpy as np
kws_extraction = "sorted"
kws_universe_size = 1000

print("read kws dict:")
time1 = time.time()
with open("dataset/wiki_kws_dict.pkl","rb") as f:
    kws_dict = pickle.load(f)
print(time.time() - time1)

kws_count = []
kws_list = list(kws_dict.keys())
for k in kws_list:
    kws_count.append([k, kws_dict[k]["count"]])
if kws_extraction == "sorted":
    kws_count.sort(reverse=True,key = lambda x: x[1])
    kws_list = kws_count[:kws_universe_size]
    kws_list = [tmp[0] for tmp in kws_list]

kws_dict_sorted = {}
for kws in kws_list:
    kws_dict_sorted[kws] = kws_dict[kws]
with open("kws_dict_"+str(kws_universe_size)+"_sorted.pkl", "wb") as f:
    pickle.dump(kws_dict_sorted,f)

print("read doc")
time1 = time.time()
with open("dataset/wiki_doc_0.pkl","rb") as f:
    doc = pickle.load(f)
print(time.time() - time1)
print("doc number:",len(doc))
print("kws number:",len(kws_list))
doc_kwsid = np.zeros((len(doc),len(kws_list)),bool)

for i in tqdm(range(len(doc))):
    for j in range(len(kws_list)):
        if kws_list[j] in doc[i]:
            doc_kwsid[i][j]=1

time1 = time.time()
with open("dataset/kws_list_and_doc_kws_new_"+str(kws_universe_size)+"_0.pkl", "wb") as f:
    pickle.dump([kws_list,doc_kwsid],f)
print(time.time() - time1)
