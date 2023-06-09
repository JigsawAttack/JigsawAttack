from run_single_attack import *
import pickle
import os
from tqdm import tqdm
from cal_acc import calculate_acc_weighted

#datasets = ["enron","lucene"]

def test_base_conf_rec(test_times,datasets,kws_uni_size):
    if not os.path.exists("./results"):
        os.makedirs("./results")
    if not os.path.exists("./results/test_base_conf_rec"):
        os.makedirs("./results/test_base_conf_rec")
    
    Base_rec = [25,50,100,200,400]
    Conf_rec_rate = [1.0,0.5,0.2]
    for dataset in datasets:
        Result_all = []
        for _ in tqdm(range(test_times)):
            Result_base = []
            for base_rec in Base_rec:
                Result_conf = []
                for conf_rec_rate in Conf_rec_rate:
                    print("run test base conf rec",dataset,"baserec:",base_rec,"confrecrate:",conf_rec_rate)
                    conf_rec = int(conf_rec_rate*base_rec)
                    attack_params = {"alg":"Ours","alpha":0.3,"step":2,\
                        "baseRec":base_rec,"confRec":conf_rec,\
                        "beta":None,"no_F":None,"refinespeed":None}
                    result = run_single_attack(kws_uni_size,kws_uni_size,"sorted",5000,30,0,dataset,\
                        {"alg":None},attack_params)
                    data_for_acc_cal = result["data_for_acc_cal"]
                    tdid_2_kwid = result["results"][1]
                    correct_count,acc,correct_id,wrong_id=calculate_acc_weighted(data_for_acc_cal,tdid_2_kwid)
                    print(correct_count/acc/(5000*30),acc,len(correct_id))
                    recovery_rate = correct_count/acc/(5000*30)

                    Result_conf.append((acc,recovery_rate,len(correct_id)))
                Result_base.append(Result_conf)
            Result_all.append(Result_base)
        Result_all = np.array(Result_all)
        Result_all = np.average(Result_all,axis=0)
        
        with open("results/test_base_conf_rec/"+dataset+"_"+str(kws_uni_size)+".pkl", "wb") as tf:
            pickle.dump([Base_rec,Conf_rec_rate,Result_all],tf)

def show(datasets,kws_uni_size):
    for dataset in datasets:
        with open("results/test_base_conf_rec/"+dataset+"_"+str(kws_uni_size)+".pkl", "rb") as tf:
            Base_rec,Conf_rec_rate,Result_all = pickle.load(tf)
        print(dataset)
        for i in range(len(Base_rec)):
            print("&",Base_rec[i],end= "  ")
            for j in range(len(Conf_rec_rate)):
                
                print("&$","%.2f"%(Result_all[i][j][0]*100),"\\%/",end=" ")
                print("%.2f"%(Result_all[i][j][1]*100),"\\%/",end=" ")
                print("%.1f"%(Result_all[i][j][2]),end="$")
            print("\\\\")
test_base_conf_rec(10,["wiki"],3000)
show(["wiki"],3000)
