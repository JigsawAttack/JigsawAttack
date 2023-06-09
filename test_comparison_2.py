import pickle
import time
from tqdm import tqdm
from cal_acc import calculate_acc_weighted
from run_single_attack import *
import os

def comparison_2(test_times,dataset,\
        Query_number_per_week = [2500],\
        Kws_uni_size = [500],observed_time=50,eta=50,kws_extraction="sorted",refspeed=10,Sim_data_rate=[1,0.5,0.1]):
    if not os.path.exists("./results"):
        os.makedirs("./results")
    if not os.path.exists("./results/test_comparison_2"):
        os.makedirs("./results/test_comparison_2")
    Known_query_number_in_RSA = 10

    print("Test our attack")
    if not os.path.exists("./results/test_comparison_2/Ours"):
        os.makedirs("./results/test_comparison_2/Ours")
    for i in tqdm(range(test_times)):
        Result = []
        for query_number_per_week in Query_number_per_week:
            result = []
            for kws_uni_size in Kws_uni_size:
                if kws_uni_size > 2000:
                    refspeed = 50
                else:
                    refspeed = 10
                if query_number_per_week <500 :
                    attack_params = {"alg":"Ours","alpha":0.3,"beta":0.9,"step":3,\
                    "baseRec":25,"confRec":20,\
                    "no_F":False,"refinespeed":refspeed}
                else:
                    
                    attack_params = {"alg":"Ours","alpha":0.3,"beta":0.9,"step":3,\
                        "baseRec":45,"confRec":35,\
                        "no_F":False,"refinespeed":refspeed}
                r = run_single_attack(kws_uni_size,kws_uni_size,kws_extraction,\
                    query_number_per_week,observed_time,eta,dataset,\
                    {"alg":None},attack_params)
                result.append(r)
            Result.append(result)         
        with open("results/test_comparison_2/Ours/"+dataset+"_"+str(i)+".pkl", "wb") as f:
            pickle.dump(Result,f)

    print("Test sap attack")
    if not os.path.exists("./results/test_comparison_2/SAP"):
        os.makedirs("./results/test_comparison_2/SAP")
    for i in tqdm(range(test_times)):
        Result = []
        for query_number_per_week in Query_number_per_week:
            result = []
            for kws_uni_size in Kws_uni_size:
                attack_params = {"alg":"Sap","alpha":0.5}
                r = run_single_attack(kws_uni_size,kws_uni_size,kws_extraction,\
                    query_number_per_week,observed_time,eta,dataset,\
                    {"alg":None},attack_params)
                result.append(r)
            Result.append(result)         
        with open("results/test_comparison_2/SAP/"+dataset+"_"+str(i)+".pkl", "wb") as f:
            pickle.dump(Result,f)

    print("Test RSA attack")
    if not os.path.exists("./results/test_comparison_2/RSA"):
        os.makedirs("./results/test_comparison_2/RSA")
    for i in tqdm(range(test_times)):
        Result = []
        for query_number_per_week in Query_number_per_week:
            result = []
            for kws_uni_size in Kws_uni_size:
                if kws_uni_size > 2000:
                    refspeed = 50
                else:
                    refspeed = 10
                attack_params = {"alg":"RSA","known_query_number":Known_query_number_in_RSA,"refinespeed":refspeed}
                r = run_single_attack(kws_uni_size,kws_uni_size,kws_extraction,\
                    query_number_per_week,observed_time,eta,dataset,\
                    {"alg":None},attack_params)
                result.append(r)
            Result.append(result)         
        with open("results/test_comparison_2/RSA/"+dataset+"_"+str(i)+".pkl", "wb") as f:
            pickle.dump(Result,f)

    print("Test IHOP attack")
    if not os.path.exists("./results/test_comparison_2/IHOP"):
        os.makedirs("./results/test_comparison_2/IHOP")
    for i in tqdm(range(test_times)):
        Result = []
        for query_number_per_week in Query_number_per_week:
            result = []
            for kws_uni_size in Kws_uni_size:
                attack_params = {"alg":"IHOP","niters":500,"pfree":0.25,"no_F":False}
                r = run_single_attack(kws_uni_size,kws_uni_size,kws_extraction,\
                    query_number_per_week,observed_time,eta,dataset,\
                    {"alg":None},attack_params)
                result.append(r)
            Result.append(result)         
        with open("results/test_comparison_2/IHOP/"+dataset+"_"+str(i)+".pkl", "wb") as f:
            pickle.dump(Result,f)
    

    # print("Test Graphm attack")
    # if not os.path.exists("./results/test_comparison_2/Graphm"):
    #     os.makedirs("./results/test_comparison_2/Graphm")
    # for i in tqdm(range(test_times)):
    #     Result = []
    #     for query_number_per_week in Query_number_per_week:
    #         result = []
    #         for kws_uni_size in Kws_uni_size:
    #             if kws_uni_size>1000:
    #                 continue
    #             Alpha = [i*0.1 for i in range(11)]
    #             for alpha in Alpha:
    #                 attack_params = {"alg":"Graphm","alpha":alpha,"match_alg":"PATH"}
    #                 r = run_single_attack(kws_uni_size,kws_uni_size,kws_extraction,\
    #                     query_number_per_week,observed_time,eta,dataset,\
    #                     {"alg":None},attack_params)
    #                 result.append(r)
    #                 correct_count,acc,correct_id,wrong_id=calculate_acc_weighted(r["data_for_acc_cal"],r["results"])
    #                 print(alpha,acc)
    #         Result.append(result)         
    #     # with open("results/test_comparison_2/Graphm/test_1000_"+dataset+"_"+str(i)+".pkl", "wb") as f:
    #     #     pickle.dump(Result,f)
if __name__ == "__main__":
    # comparison_2(1,"enron")
    # comparison_2(30,"lucene")
    comparison_2(10,"wiki",Query_number_per_week=[1000,5000,25000],Kws_uni_size=[1000,3000,5000],observed_time=30,eta=10)
