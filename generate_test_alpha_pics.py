import pickle
import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.cm


from cal_acc import *
datasets = ["wiki_3000","enron","lucene"]

def draw_3D(Acc_M_vf,Ratio_V,Ratio_F,title,path):
    plt.rcParams.update({
    "text.usetex": True,
    "font.family": "stix",
    "font.serif": ["Times"],
    "font.size":20})
    plt.style.context(['science', 'no-latex'])
    fig = plt.figure()
    ax1 = plt.axes(projection='3d')
    X, Y = np.meshgrid(Ratio_F, Ratio_V)
    X, Y = X.ravel(), Y.ravel()
    Acc_M_vf = Acc_M_vf.ravel()
    bottom = np.zeros_like(Acc_M_vf)
    width = 0.02
    depth = 0.02
    cmap = matplotlib.cm.get_cmap('hsv')
    #cmap=mpl.colormaps["hsv"]
    ax1.bar3d(X, Y, bottom, width, depth, Acc_M_vf, shade=True,color = cmap(Acc_M_vf))
    ax1.set_xlabel("$rv$",size=60)
    ax1.set_ylabel("$rf$",size=60)
    ax1.set_zlabel("Accuracy",labelpad=-0.5,size=60)
    ax1.view_init(azim=45)
    ax1.xaxis.set_major_locator(ticker.MultipleLocator(1.0))
    ax1.xaxis.set_minor_locator(ticker.MultipleLocator(0.25))
    ax1.yaxis.set_major_locator(ticker.MultipleLocator(1.0))
    ax1.yaxis.set_minor_locator(ticker.MultipleLocator(0.25))
    ax1.set_zlim(0,1.0)
    ax1.zaxis.set_major_locator(ticker.MultipleLocator(1.0))
    ax1.zaxis.set_minor_locator(ticker.MultipleLocator(0.25))
    
    
    plt.savefig(path+title+".pdf",pad_inches=0.2, bbox_inches = 'tight')
    #plt.show()
    plt.clf()
def draw_alpha_one_quadrant(Alpha,acc,title,path):
    if not os.path.exists(path):
        os.makedirs(path)

    #plt.style.use('seaborn-whitegrid')
    plt.rcParams.update({
    "text.usetex": True,
    "font.family": "stix",
    "font.serif": ["Times"],
    "font.size":30,
    "lines.markersize":7})
    plt.style.context(['science', 'no-latex'])
    plt.plot(Alpha,acc,marker='o', linewidth=2,label="Accuracy")
    plt.ylim(0,1.0)
    plt.ylabel("Accuracy",size=60)
    plt.xlabel("$\\alpha$",size=60)
    #plt.show()
    plt.savefig(path+title+".pdf",pad_inches=0.0, bbox_inches = 'tight')
    plt.clf()

def draw_test_alpha(test_times):
    for dataset in datasets:
        Acc_all = []
        Acc_quadrants = []
        Data_for_qudrants = []
        for i in range(test_times):
            with open("results/test_alpha/"+dataset+"_3000_10_"+str(i)+".pkl", "rb") as tf:
                Alpha,Result = pickle.load(tf)
                acc_all = []
                acc_quadrants = [[],[],[],[]]
                for result in Result:
                    real_F = result["real_F"]
                    real_V = result["real_V"]
                    data_for_acc_cal = result["data_for_acc_cal"]
                    tdid_2_kwid = result["results"][0]
                    correct_count,acc,correct_id,wrong_id=calculate_acc_weighted(data_for_acc_cal,tdid_2_kwid)
                    acc_all.append(acc)
                    results_in_quadrants = show_results(correct_id,wrong_id,real_F,real_V,0.1,0.1,False)
                    for i in range(4):
                        acc_quadrants[i].append(results_in_quadrants[i]["correct_count"]/results_in_quadrants[i]["qcount"])
                    if result["attack_params"]["alpha"]==0.5:
                        Data_for_qudrants.append([correct_id,wrong_id,real_F,real_V])
            Acc_all.append(acc_all)
            Acc_quadrants.append(acc_quadrants) 
        draw_alpha_one_quadrant(Alpha,np.average(Acc_all,axis=0),"All","results/test_alpha_pic_"+dataset+"/")      
        title = ["LVLF","LVHF","HVLF","HVHF"]
        for i in range(4):
            draw_alpha_one_quadrant(Alpha,np.average(Acc_quadrants,axis=0)[i],title[i],"results/test_alpha_pic_"+dataset+"/")
        Acc_M = []
        Ratio_V = [(i+1)*0.02 for i in range(49)]
        Ratio_F = [(i+1)*0.02 for i in range(49)]
        for ratio_V in Ratio_V:
            Acc_M_F = []
            for ration_F in Ratio_F:
                Acc_quadrants = []
                for data in Data_for_qudrants:
                    results_in_quadrants = show_results(data[0],data[1],data[2],data[3],ratio_V,ration_F,False)
                    acc_quadrants = [[],[],[],[]]
                    for i in range(4):
                        if results_in_quadrants[i]["qcount"] == 0:
                            acc_quadrants[i].append(0)
                        else:
                            acc_quadrants[i].append(results_in_quadrants[i]["correct_count"]/results_in_quadrants[i]["qcount"])
                    Acc_quadrants.append(acc_quadrants)
                Acc_quadrants = np.average(np.array(Acc_quadrants),axis=0)  
                Acc_M_F.append(Acc_quadrants)
            Acc_M.append(Acc_M_F)
        Acc_M = np.array(Acc_M)
        for i in range(4):
            draw_3D(Acc_M[:,:,i,:],Ratio_V,Ratio_F,title[i]+"_3D","results/test_alpha_pic_"+dataset+"/")
draw_test_alpha(30)
