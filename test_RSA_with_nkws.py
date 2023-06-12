from tqdm import tqdm
from run_single_attack import *
import pickle
from cal_acc import calculate_acc_weighted
import os

def run_RSA_with_various_known_queries(test_times=1,\
    datasets=["enron"],Kws_uni_size=[500],\
    Known_queries=[5],kws_extraction="random",observe_query_number=500,observe_weeks = 50, tau = 50,refspeed = 10):
    if not os.path.exists("./results"):
        os.makedirs("./results")
    if not os.path.exists("./results/test_RSA_with_nkws"):
        os.makedirs("./results/test_RSA_with_nkws")

    print("Test RSA with various number of kws")
    for dataset in datasets:
        for kws_uni_size in Kws_uni_size:
            for known_query_number in Known_queries:
                Result = []
                for i in tqdm(range(test_times)):
                    attack_params={
                        "alg": "RSA",
                        "refinespeed":refspeed,
                        "known_query_number":known_query_number
                    }
                    countermeasure_params={"alg":None}
                    result = run_single_attack(kws_uni_size,kws_uni_size,kws_extraction,observe_query_number,observe_weeks,tau,dataset,
                    countermeasure_params,attack_params)
                    data_for_acc_cal = result["data_for_acc_cal"]
                    for key in result["results"][1].keys():
                        del result["results"][0][key]
                    correct_count,acc,correct_id,wrong_id = \
                        calculate_acc_weighted(data_for_acc_cal,result["results"][0],is_print=False)
                    print({"dataset":dataset,"kws_uni_size":kws_uni_size,"known_query_number":known_query_number,"acc":acc})
                    Result.append((dataset,kws_uni_size,known_query_number,acc,result))
                if kws_extraction == "sorted":
                    with open("./results/test_RSA_with_nkws/RSA_"+dataset+"_kws_uni_size_"+str(kws_uni_size)+\
                        "_known_query_number_"+str(known_query_number)+\
                        "_test_times_"+str(test_times)+".pkl", "wb") as f:
                        pickle.dump(Result,f)
                else:
                    with open("./results/test_RSA_with_nkws/RSA_random_"+dataset+"_kws_uni_size_"+str(kws_uni_size)+\
                        "_known_query_number_"+str(known_query_number)+\
                        "_test_times_"+str(test_times)+".pkl", "wb") as f:
                        pickle.dump(Result,f)


def read_results(test_times=30,\
    Kws_uni_size=[1000],Known_queries=[5],\
    datasets=["enron"],kws_extraction="random"):
    for dataset in datasets:
        for kws_uni_size in Kws_uni_size:
            for known_query_number in Known_queries:
                if kws_extraction == "sorted":
                    with open("./results/test_RSA_with_nkws/RSA_"+dataset+"_kws_uni_size_"+str(kws_uni_size)+\
                        "_known_query_number_"+str(known_query_number)+\
                        "_test_times_"+str(test_times)+".pkl", "rb") as f:
                        Result = pickle.load(f)
                else:
                    with open("./results/test_RSA_with_nkws/RSA_random_"+dataset+"_kws_uni_size_"+str(kws_uni_size)+\
                        "_known_query_number_"+str(known_query_number)+\
                        "_test_times_"+str(test_times)+".pkl", "rb") as f:
                        Result = pickle.load(f)
                Acc = []
                for result in Result:
                    acc = result[3]
                    Acc.append(acc)
                print(dataset,kws_extraction,kws_uni_size,known_query_number,np.average(np.array(Acc)))

 
run_RSA_with_various_known_queries(test_times=30,\
    Kws_uni_size=[1000],Known_queries=[5,10,25,50],\
    datasets=["enron","lucene"],kws_extraction="random")
run_RSA_with_various_known_queries(test_times=30,\
    Kws_uni_size=[1000],Known_queries=[5,10,25,50],\
    datasets=["enron","lucene"],kws_extraction="sorted")

read_results(test_times=30,\
    Kws_uni_size=[1000],Known_queries=[5,10,25,50],\
    datasets=["enron","lucene"],kws_extraction="random")
read_results(test_times=30,\
    Kws_uni_size=[1000],Known_queries=[5,10,25,50],\
    datasets=["enron","lucene"],kws_extraction="sorted")


run_RSA_with_various_known_queries(test_times=10,\
    Kws_uni_size=[3000],Known_queries=[15,30,75,150],\
    datasets=["wiki"],kws_extraction="random",observe_query_number=5000,observe_weeks = 30, tau = 10)
run_RSA_with_various_known_queries(test_times=10,\
    Kws_uni_size=[3000],Known_queries=[15,30,75,150],\
    datasets=["wiki"],kws_extraction="sorted",observe_query_number=5000,observe_weeks = 30, tau = 10)

read_results(test_times=10,\
    Kws_uni_size=[3000],Known_queries=[15,30,75,150],\
    datasets=["wiki"],kws_extraction="random")
read_results(test_times=10,\
    Kws_uni_size=[3000],Known_queries=[15,30,75,150],\
    datasets=["wiki"],kws_extraction="sorted")
