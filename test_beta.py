from run_single_attack import *
import pickle
import os
from tqdm import tqdm
from cal_acc import calculate_acc_weighted

def test_beta(test_times,kws_uni_size,refspeed,datasets = ["enron","lucene"],query_number_per_week=2000,weeks=50):
    if not os.path.exists("./results"):
        os.makedirs("./results")
    if not os.path.exists("./results/test_beta"):
        os.makedirs("./results/test_beta")
    Beta = [i*0.2 for i in range(6)]
    
    for dataset in datasets:
        Acc_with_freq = []
        Acc_without_freq = []
        for _ in tqdm(range(test_times)):
            acc_with_freq = []
            acc_without_freq = []
            for beta in Beta:
                print(dataset,"alpha:0.3","beta:",beta)
                attack_params_with_freq = {"alg":"Ours","alpha":0.3,"step":3,\
                        "baseRec":100,"confRec":50,\
                        "beta":beta,"no_F":False,"refinespeed":refspeed}
                attack_params_without_freq = {"alg":"Ours","alpha":1,"step":3,\
                        "baseRec":100,"confRec":20,\
                        "beta":beta,"no_F":True,"refinespeed":refspeed}
                result = run_single_attack(kws_uni_size,kws_uni_size,"sorted",query_number_per_week,weeks,0,dataset,\
                        {"alg":None},attack_params_with_freq)
                data_for_acc_cal = result["data_for_acc_cal"]
                tdid_2_kwid = result["results"][2]
                correct_count,acc,correct_id,wrong_id=\
                    calculate_acc_weighted(data_for_acc_cal,tdid_2_kwid)
                acc_with_freq.append(acc)
                print("Acc with frequency:",acc)

                result = run_single_attack(kws_uni_size,kws_uni_size,"sorted",query_number_per_week,weeks,0,dataset,\
                        {"alg":None},attack_params_without_freq)
                data_for_acc_cal = result["data_for_acc_cal"]
                tdid_2_kwid = result["results"][2]
                correct_count,acc,correct_id,wrong_id=\
                    calculate_acc_weighted(data_for_acc_cal,tdid_2_kwid)
                acc_without_freq.append(acc)
                print("Acc without frequency:",acc)
            Acc_with_freq.append(acc_with_freq)
            Acc_without_freq.append(acc_without_freq)
        with open("results/test_beta/"+dataset+"_"+str(kws_uni_size)+".pkl", "wb") as tf:
            pickle.dump([Beta,Acc_with_freq,Acc_without_freq],tf)

test_beta(30,1000,10,datasets=["enron","lucene"])
test_beta(30,3000,10,datasets=["wiki"],query_number_per_week=5000,weeks=30)
                
