import pickle
from matplotlib import pyplot as plt
import numpy as np
def generate_against_countermeasures(dataset,countermeasure,kws_uni_size,test_times=30,legend=True):
    N = [0,500,1000,1500]
    Our_result = []
    RSA_result = []
    IHOP_result = []
    IHOP_alpha_result = []
    if countermeasure == "padding":
        if dataset== "wiki":
            params = [0,50000,100000,150000]
        else:
            params = [0,500,1000,1500]
        string = "_padding_linear_n_"
    elif  countermeasure == "obfuscation":
        
        if dataset== "wiki":
            params = [0,0.1,0.2,0.3]
        else:
            params = [0,0.01,0.02,0.05]
        string = "_obfuscation_q_"
    elif countermeasure == "padding_cluster":
        params = [1,2,4,8]
        string = "_padding_cluster_knum_in_cluster_"
    elif  countermeasure == "padding_seal":
        params = [1,2,3,4]
        string = "_padding_seal_"
    for param in params:
        with open("results/test_against_countermeasures/Ours_"+dataset+\
                string+str(param)+\
                "_kws_uni_size_"+str(kws_uni_size)+\
                "_test_times_"+str(test_times)+".pkl","rb") as f:
            result = pickle.load(f)
            Our_result.append(result)
        with open("results/test_against_countermeasures/RSA_"+dataset+\
                string+str(param)+\
                "_kws_uni_size_"+str(kws_uni_size)+\
                "_test_times_"+str(test_times)+".pkl","rb") as f:
            result = pickle.load(f)
            RSA_result.append(result)
        with open("results/test_against_countermeasures/IHOP_"+dataset+\
                string+str(param)+\
                "_kws_uni_size_"+str(kws_uni_size)+\
                "_test_times_"+str(test_times)+".pkl","rb") as f:
            result = pickle.load(f)
            IHOP_result.append(result)
        
        
        with open("results/test_IHOP_with_different_alpha/IHOP_"+dataset+\
                string+str(param)+\
                "_kws_uni_size_"+str(kws_uni_size)+\
                "_test_times_"+str(test_times)+\
                "_alpha_"+str(0.1)+\
                ".pkl","rb") as f:
            result = pickle.load(f)
            IHOP_alpha_result.append(result)
    
    Our_acc = []
    RSA_acc = []
    IHOP_acc = []
    IHOP_alpha_acc = []
    Our_time = []
    RSA_time = []
    IHOP_time = []
    

    
    for result in Our_result:
        com_overhead = []
        sto_overhead = []
        acc = []
        attack_time = []
        for r in result:
            acc.append(r[2])
            attack_time.append(r[3]["Attack_time"])
            com_overhead.append(r[3]["data_for_acc_cal"]["communication overhead"])
            sto_overhead.append(r[3]["data_for_acc_cal"]["storage overhead"])
        Our_acc.append(acc)
        Our_time.append(np.average(attack_time))
        print("Communication overhead:",np.average(com_overhead))
        print("Storage overhead:",np.average(sto_overhead))
    for result in RSA_result:
        acc = []
        attack_time = []
        for r in result:
            acc.append(r[2])
            attack_time.append(r[3]["Attack_time"])
        RSA_acc.append(acc)
        RSA_time.append(np.average(attack_time))
    for result in IHOP_result:
        acc = []
        attack_time = []
        for r in result:
            acc.append(r[2])
            attack_time.append(r[3]["Attack_time"])
        IHOP_acc.append(acc)
        IHOP_time.append(np.average(attack_time))
        
    for result in IHOP_alpha_result:
        acc = []
        for r in result:
            acc.append(r[2])
        IHOP_alpha_acc.append(acc)

    lineprops = dict(linewidth=1.5,color='black')
    Positions = np.array([1,2,3,4])
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

    for i in range(len(N)):
        our_lineprops = dict(linewidth=1.5,color='darkgreen')
        Bp_Ours=ax.boxplot(Our_acc,positions=Positions-0.24,\
            widths=0.1,patch_artist=True,boxprops=dict(facecolor="mediumseagreen",edgecolor= "darkgreen"),whiskerprops=our_lineprops,
                medianprops=our_lineprops,capprops=our_lineprops)
        
        
        rsa_lineprops = dict(linewidth=1.5,color='darkblue')
        Bp_RSA=ax.boxplot(RSA_acc,positions=Positions-0.08,\
            widths=0.1,patch_artist=True,boxprops=dict(facecolor="cornflowerblue",edgecolor= "darkblue"),
            whiskerprops=rsa_lineprops,medianprops=rsa_lineprops,capprops=rsa_lineprops)
        
        
        ihop_lineprops = dict(linewidth=1.5,color='darkorange')
        Bp_IHOP=ax.boxplot(IHOP_acc,positions=Positions+0.08,\
            widths=0.1,patch_artist=True,boxprops=dict(facecolor="wheat",edgecolor= "darkorange"),whiskerprops=ihop_lineprops,
            medianprops=ihop_lineprops,capprops=ihop_lineprops)
        
        
        ihop_alpha_lineprops = dict(linewidth=1.5,color='brown')
        Bp_IHOP_alpha=ax.boxplot(IHOP_alpha_acc,positions=Positions+0.24,\
            widths=0.1,patch_artist=True,boxprops=dict(facecolor="pink",edgecolor= "brown"),whiskerprops=ihop_alpha_lineprops,
            medianprops=ihop_alpha_lineprops,capprops=ihop_alpha_lineprops)
        
   
    
    for i in range(len(Our_time)):
        ax2.scatter(y=Our_time[i],x=Positions[i]-0.24,marker="o",color="mediumseagreen",s=60)
    for i in range(len(RSA_time)):
        ax2.scatter(y=RSA_time[i],x=Positions[i]-0.08,marker="s",color="cornflowerblue",s=60)
    for i in range(len(IHOP_time)):
        ax2.scatter(y=IHOP_time[i],x=Positions[i]+0.08,marker="p",color="wheat",s=60)
    for i in range(len(IHOP_time)):
        ax2.scatter(y=IHOP_time[i],x=Positions[i]+0.24,marker="^",color="pink",s=60)
    
    
    
    for pos in [1.5,2.5,3.5]:
        ax.plot([pos, pos], [0, 1], c="black")


    ax2.set_ylabel("Time(seconds)",fontsize=40)
    ax.set_ylabel("Accuracy",fontsize=40)
    ax.set_ylim(-0.05,1.05)
    
    ax2.set_ylim(0,np.max(np.vstack((IHOP_time,RSA_time,Our_time)))*2.3)
    
    ax.set_xlim(0.5,4.5)
    if countermeasure=="padding":
        if dataset != "wiki":
            plt.xticks(Positions,["No defence","500","1000","1500"])
        else:
            plt.xticks(Positions,["No defence","50000","100000","150000"])
        ax.set_xlabel("$k$")
        #plt.title("Against padding in CGPR")
    elif countermeasure=="obfuscation":
        if dataset == "wiki":
            plt.xticks(Positions,["No defence","0.1","0.2","0.3"])
        else:
            plt.xticks(Positions,["No defence","0.01","0.02","0.05"])
        
        ax.set_xlabel("FPR")
        #plt.title("Against obfuscation")
    elif countermeasure=="padding_seal":
        plt.xticks(Positions,["No defence","2","3","4"])
        ax.set_xlabel("$x$",fontsize=40)
    elif countermeasure=="padding_cluster":
        plt.xticks(Positions,["No defence","2","4","8"])
        ax.set_xlabel("$\\alpha$",fontsize=40)

    if legend == True:
        plt.legend([Bp_Ours["boxes"][0], Bp_RSA["boxes"][0],Bp_IHOP["boxes"][0],Bp_IHOP_alpha["boxes"][0]],\
            ["Jigsaw+Adp","RSA+Adp","IHOP+Adp","IHOP+$\\alpha$+Adp"],\
            loc="center left",bbox_to_anchor=(-0.015, 0.7),\
            loc="lower left",\
            fancybox=True, shadow=True,fontsize=17)
    
    plt.savefig("results/test_against_countermeasures/Re"+dataset+"_"+countermeasure+"_"+str(kws_uni_size)+".pdf",pad_inches=0.0, bbox_inches = 'tight')
    plt.show()

if __name__ == "__main__":

    generate_against_countermeasures("enron","padding",1000,test_times=30,legend=True)
    generate_against_countermeasures("lucene","padding",1000,test_times=30,legend=False)
    generate_against_countermeasures("enron","obfuscation",1000,test_times=30,legend=True)
    generate_against_countermeasures("lucene","obfuscation",1000,test_times=30,legend=False)


    generate_against_countermeasures("wiki","obfuscation",1000,test_times=10,legend=True)
    generate_against_countermeasures("wiki","obfuscation",3000,test_times=10,legend=False)
    generate_against_countermeasures("wiki","obfuscation",5000,test_times=10,legend=False)
    
    
    generate_against_countermeasures("wiki","padding",3000,test_times=10,legend=False)
    generate_against_countermeasures("wiki","padding",5000,test_times=10,legend=False)
    generate_against_countermeasures("wiki","padding",1000,test_times=10,legend=True)
   

    generate_against_countermeasures("enron","padding_cluster",1000,test_times=30,legend=True)
    generate_against_countermeasures("lucene","padding_cluster",1000,test_times=30,legend=False)

    generate_against_countermeasures("enron","padding_seal",1000,test_times=30,legend=True)
    generate_against_countermeasures("lucene","padding_seal",1000,test_times=30,legend=False)
    

    generate_against_countermeasures("wiki","padding_seal",1000,test_times=10,legend=False)
    generate_against_countermeasures("wiki","padding_cluster",1000,test_times=10,legend=False)

    generate_against_countermeasures("wiki","padding_seal",3000,test_times=10,legend=False)
    generate_against_countermeasures("wiki","padding_cluster",3000,test_times=10,legend=False)

    generate_against_countermeasures("wiki","padding_seal",5000,test_times=10,legend=False)
    generate_against_countermeasures("wiki","padding_cluster",5000,test_times=10,legend=False)

   