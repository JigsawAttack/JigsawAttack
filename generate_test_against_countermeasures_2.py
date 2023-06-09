import pickle
from matplotlib import pyplot as plt
import numpy as np
def generate_against_countermeasures(dataset,countermeasure,kws_uni_size,test_times=30,legend=True):
    Our_result = []
    RSA_result = []
    IHOP_result = []
    if countermeasure == "padding_linear":
        params = [0,500,1000,1500]
        string = "_padding_linear_n_"
    elif countermeasure == "padding_cluster":
        params = [1,2,4,8]
        string = "_padding_cluster_knum_in_cluster_"
    elif  countermeasure == "obfuscation":
        params = [0,0.01,0.02,0.05]
        string = "_obfuscation_q_"
    elif  countermeasure == "padding_seal":
        params = [1,2,3,4]
        string = "_padding_seal_"
    for param in params:
        with open("test/results/test_against_countermeasures/Ours_"+dataset+\
            string+str(param)+\
            "_kws_uni_size_"+str(kws_uni_size)+\
            "_test_times_"+str(test_times)+".pkl","rb") as f:
            result = pickle.load(f)
            Our_result.append(result)
        with open("test/results/test_against_countermeasures/RSA_"+dataset+\
                string+str(param)+\
                "_kws_uni_size_"+str(kws_uni_size)+\
                "_test_times_"+str(test_times)+".pkl","rb") as f:
            result = pickle.load(f)
            RSA_result.append(result)
        with open("test/results/test_against_countermeasures/IHOP_"+dataset+\
                string+str(param)+\
                "_kws_uni_size_"+str(kws_uni_size)+\
                "_test_times_"+str(test_times)+".pkl","rb") as f:
            result = pickle.load(f)
            IHOP_result.append(result)
    
    Our_acc = []
    RSA_acc = []
    IHOP_acc = []
    Com_overhead = []
    Sto_overhead = []
    for result in Our_result:
        acc = []
        com_overhead = []
        sto_overhead = []
        for r in result:
            acc.append(r[2])
            com_overhead.append(r[3]["data_for_acc_cal"]["communication overhead"])
            sto_overhead.append(r[3]["data_for_acc_cal"]["storage overhead"])
        Our_acc.append(acc)
        Com_overhead.append(com_overhead)
        Sto_overhead.append(sto_overhead)
    for result in RSA_result:
        acc = []
        for r in result:
            acc.append(r[2])
        RSA_acc.append(acc)
    for result in IHOP_result:
        acc = []
        for r in result:
            acc.append(r[2])
        IHOP_acc.append(acc)

    lineprops = dict(linewidth=1.5,color='black')
    Positions = np.array([i for i in range(len(params))])
    plt.rcParams.update({
    'figure.figsize': '6.5, 4.8',
    "text.usetex": True,
    "font.family": "stix",
    "font.serif": ["Times"],
    "font.size":14,
    "lines.markersize":10})
    plt.style.context(['science', 'no-latex'])
    
    fig, ax = plt.subplots()
    ax2 = ax.twinx()

    for i in range(len(params)):
        our_lineprops = dict(linewidth=1.5,color='darkgreen')
        Bp_Ours=ax.boxplot(Our_acc,positions=Positions-0.15,\
            widths=0.1,patch_artist=True,boxprops=dict(facecolor="mediumseagreen",edgecolor= "darkgreen"),whiskerprops=our_lineprops,
                medianprops=our_lineprops,capprops=our_lineprops)
        ihop_lineprops = dict(linewidth=1.5,color='darkorange')
        Bp_IHOP=ax.boxplot(IHOP_acc,positions=Positions+0.15,\
            widths=0.1,patch_artist=True,boxprops=dict(facecolor="wheat",edgecolor= "darkorange"),whiskerprops=ihop_lineprops,
            medianprops=ihop_lineprops,capprops=ihop_lineprops)
        rsa_lineprops = dict(linewidth=1.5,color='darkblue')
        Bp_RSA=ax.boxplot(RSA_acc,positions=Positions,\
            widths=0.1,patch_artist=True,boxprops=dict(facecolor="cornflowerblue",edgecolor= "darkblue"),
            whiskerprops=rsa_lineprops,medianprops=rsa_lineprops,capprops=rsa_lineprops)
    Com_overhead = np.array(Com_overhead)
    Com_overhead = np.average(Com_overhead,axis=1)
    Sto_overhead = np.array(Sto_overhead)
    Sto_overhead = np.average(Sto_overhead,axis=1)
    sto = ax2.plot(Positions,Sto_overhead,linestyle = "-",c="r")
    com = ax2.plot(Positions,Com_overhead,linestyle = "--",c="orange")
    ax2.set_ylabel("Overhead",fontsize=20)
    ax.set_ylabel("Accuracy",fontsize=20)
    ax.set_ylim(-0.05,1.05)
    if countermeasure=="padding_linear":
        plt.xticks(Positions,["No defence","500","1000","1500"])
        ax.set_xlabel("$k$")
        plt.title("Against padding in CGPR")
    elif countermeasure=="obfuscation":
        plt.xticks(Positions,["No defence","0.01","0.02","0.05"])
        ax.set_xlabel("FPR")
        plt.title("Against obfuscation")
    elif countermeasure=="padding_seal":
        plt.xticks(Positions,["No defence","2","3","4"])
        ax.set_xlabel("$x$")
        #plt.title("Against the padding in SEAL")
    elif countermeasure=="padding_cluster":
        plt.xticks(Positions,["No defence","2","4","8"])
        ax.set_xlabel("$\\alpha$")
        #plt.title("Against the cluster basaed padding")
    if legend == True:
        plt.legend([Bp_Ours["boxes"][0], Bp_RSA["boxes"][0],Bp_IHOP["boxes"][0],sto[0],com[0]],\
            ["Jigsaw+Adp","RSA+Adp","IHOP+Adp","Storage","Com"],\
            loc="center left", bbox_to_anchor=(-0.015, 0.5),\
            fancybox=True, shadow=True)
    
    plt.savefig("test/results/test_against_countermeasures/"+dataset+"_"+countermeasure+"_"+str(kws_uni_size)+".pdf",bbox_inches = 'tight')
    plt.show()

if __name__ == "__main__":
  
    generate_against_countermeasures("enron","padding_seal",1000,test_times=10)
    generate_against_countermeasures("lucene","padding_seal",1000,test_times=10)
    generate_against_countermeasures("wiki","padding_seal",1000,test_times=10)
    generate_against_countermeasures("enron","padding_cluster",1000,test_times=10)
    generate_against_countermeasures("lucene","padding_cluster",1000,test_times=10)

    generate_against_countermeasures("wiki","padding_cluster",1000,test_times=10)