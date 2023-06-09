from run_single_attack import *
import pickle
import os
from tqdm import tqdm




def test_alpha(test_times,datasets,kws_uni_size):
    if not os.path.exists("./results"):
        os.makedirs("./results")
    if not os.path.exists("./results/test_alpha"):
        os.makedirs("./results/test_alpha")

    Alpha = [i*0.05 for i in range(0,21)]
    for dataset in datasets:
        for i in tqdm(range(test_times)):
            Result = []
            for alpha in Alpha:
                print("Test alpha:",alpha)
                attack_params = {"alg":"Ours","alpha":alpha,"step":1,"baseRec":1000,\
                    "beta":None,"no_F":None,"confRec":None,"refinespeed":None}
                result = run_single_attack(kws_uni_size,kws_uni_size,"sorted",5000,30,0,dataset,\
                    {"alg":None},attack_params)
                Result.append(result)
            
            with open("results/test_alpha/"+dataset+"_3000_"+str(kws_uni_size)+"_"+str(test_times)+"_"+str(i)+".pkl", "wb") as tf:
                pickle.dump([Alpha,Result],tf)
    return 0
test_alpha(10,["wiki"],3000)
