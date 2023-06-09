import pickle
from matplotlib import pyplot as plt
import numpy as np
def generate_test_similar_data_size(Similar_data_p=[0.2,0.4,0.6,0.8,1],dataset="enron",kws_uni_size=1000,test_times=30):
    Our_result = []
    RSA_result = []
    IHOP_result = []
    for similar_data_p in Similar_data_p:
        with open("./results/test_similar_data_p/Ours_"+dataset+\
                "_test_similar_data_p_"+str(similar_data_p)+\
                "_kws_uni_size_"+str(kws_uni_size)+\
                "_test_times_"+str(test_times)+".pkl","rb") as f:
            result = pickle.load(f)
            Our_result.append(result)
        with open("./results/test_similar_data_p/RSA_"+dataset+\
                "_test_similar_data_p_"+str(similar_data_p)+\
                "_kws_uni_size_"+str(kws_uni_size)+\
                "_test_times_"+str(test_times)+".pkl","rb") as f:
            result = pickle.load(f)
            RSA_result.append(result)
        with open("./results/test_similar_data_p/IHOP_"+dataset+\
                "_test_similar_data_p_"+str(similar_data_p)+\
                "_kws_uni_size_"+str(kws_uni_size)+\
                "_test_times_"+str(test_times)+".pkl","rb") as f:
            result = pickle.load(f)
            IHOP_result.append(result)
    
    Our_acc = []
    RSA_acc = []
    IHOP_acc = []
    for result in Our_result:
        acc = []
        for r in result:
            acc.append(r[2])
        Our_acc.append(acc)
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
    Positions = np.array([i for i in range(len(Similar_data_p))])
    plt.rcParams.update({
    "text.usetex": True,
    "font.family": "stix",
    "font.serif": ["Times"],
    "font.size":17,
    "lines.markersize":10})
    plt.style.context(['science', 'no-latex'])
    
    fig, ax = plt.subplots()
    

    for i in range(len(Similar_data_p)):
        Bp_Ours=ax.boxplot(Our_acc,positions=Positions-0.15,\
            widths=0.1,patch_artist=True,boxprops=dict(facecolor="g"),whiskerprops=lineprops,medianprops=lineprops,capprops=lineprops)
        Bp_IHOP=ax.boxplot(IHOP_acc,positions=Positions+0.15,\
            widths=0.1,patch_artist=True,boxprops=dict(facecolor="cyan"),whiskerprops=lineprops,medianprops=lineprops,capprops=lineprops)
        Bp_RSA=ax.boxplot(RSA_acc,positions=Positions,\
            widths=0.1,patch_artist=True,boxprops=dict(facecolor="blue"),whiskerprops=lineprops,medianprops=lineprops,capprops=lineprops)
    
    
    plt.xticks(Positions,Similar_data_p)
    ax.set_xlabel("$|D_s|/|D|$")
    #plt.title("Against padding in CGPR")
    plt.legend([Bp_Ours["boxes"][0], Bp_RSA["boxes"][0],Bp_IHOP["boxes"][0]],\
        ["Ours","RSA","IHOP"],\
        loc=4, \
        fancybox=True, shadow=True)
    ax.set_ylim(0,1)
    ax.set_ylabel("Accuracy")
    plt.savefig("results/test_similar_data_p/"+dataset+"_"+str(test_times)+"_"+str(kws_uni_size)+".pdf",pad_inches=0.0, bbox_inches = 'tight')
    plt.show()

if __name__ == "__main__":
    generate_test_similar_data_size(Similar_data_p=[0.2,0.4,0.6,0.8,1],dataset="enron",kws_uni_size=1000,test_times=30)
    generate_test_similar_data_size(Similar_data_p=[0.2,0.4,0.6,0.8,1],dataset="lucene",kws_uni_size=1000,test_times=30)
    generate_test_similar_data_size(Similar_data_p=[0.01,0.02,0.03,0.04,0.05],dataset="wiki",kws_uni_size=3000,test_times=10)