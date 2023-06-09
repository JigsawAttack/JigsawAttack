import pickle
from matplotlib import pyplot as plt
import numpy as np
datasets = ["enron","lucene","wiki_3000"]#
for dataset in datasets:
    with open("results/test_beta/"+dataset+".pkl", "rb") as tf:
        Beta,Acc_with_freq,Acc_without_freq = pickle.load(tf)
    plt.rcParams.update({
    'figure.figsize': '6, 3',  # set figure size
    "text.usetex": True,
    "font.family": "stix",
    "font.serif": ["Times"],
    "font.size":25,
    "lines.markersize":20})
    plt.style.context(['science', 'no-latex'])


    Beta = [0.0,0.2,0.4,0.6,0.8,1.0]
    plt.boxplot(np.array(Acc_with_freq),labels=Beta,showmeans=False,showfliers=False)
    print(np.average(np.array(Acc_with_freq),axis=0))
    plt.xlabel("$\\beta$")
    plt.ylabel("Accuracy")
    plt.ylim(0,1.05)
    plt.tick_params(labelsize=20) 
    plt.tight_layout()
    plt.savefig("results/test_beta/"+dataset+"_with_freq.pdf",pad_inches=0.0, bbox_inches = 'tight')
    plt.show()
    plt.clf()

    plt.boxplot(np.array(Acc_without_freq),labels=Beta,showmeans=False,showfliers=False)
    print(np.average(np.array(Acc_without_freq),axis=0))
    plt.xlabel("$\\beta$")
    plt.ylabel("Accuracy")
    plt.ylim(0,1.05)
    plt.tick_params(labelsize=20) 
    plt.tight_layout()
    plt.savefig("results/test_beta/"+dataset+"_without_freq.pdf",pad_inches=0.0, bbox_inches = 'tight')
    plt.show()
    plt.clf()
