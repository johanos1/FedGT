import numpy as np
import matplotlib.pyplot as plt

# my_data = np.genfromtxt("results/20_it/mesid20it.csv", delimiter=",")

"""
merge = True
if merge is False:
    no_it = 50
    if no_it == 40:
        my_data1 = np.genfromtxt("results/20_it/mesid20it.csv", delimiter=",")
        my_data2 = np.genfromtxt("results/20_it/mesid.csv", delimiter=",")
        my_data = np.hstack((my_data1, my_data2))
        print(my_data.shape)
    elif no_it == 50:
        my_data = np.genfromtxt("results/50_it/mesid50.csv", delimiter=",")
        print(my_data.shape)
    elif no_it == 20:
        my_data = np.genfromtxt("results/20_it/mesid20it.csv", delimiter=",")
        print(my_data.shape)
else:
    my_data1 = np.genfromtxt("results/20_it/mesid20it.csv", delimiter=",")
    my_data2 = np.genfromtxt("results/20_it/mesid.csv", delimiter=",")
    my_data = np.hstack((my_data1, my_data2))
    my_data3 = np.genfromtxt("results/50_it/mesid50.csv", delimiter=",")
    my_data = np.hstack((my_data, my_data3))
    print(my_data.shape)
    """
filename = "20221212_233658"
my_data = np.genfromtxt("results/" + filename + "/accs.csv", delimiter=",")
threshold_vec = np.genfromtxt("results/" + filename + "/lambdas.csv", delimiter=",")
P_FA = np.genfromtxt("results/" + filename + "/P_FA.csv", delimiter=",")
P_MD = np.genfromtxt("results/" + filename + "/P_MD.csv", delimiter=",")
MCits = np.genfromtxt("results/" + filename + "/MCits.csv", delimiter=",")
Chm = np.genfromtxt("results/" + filename + "/ChannelMatrix.csv", delimiter=",")
assert Chm[0, 1] == Chm[1, 0], "They should have same flip prob"
p = Chm[0, 1]

no_defense_acc = np.genfromtxt(
    "results/No_defense20221212_185658_mali_5/accs.csv", delimiter=","
)
only_ben_acc = np.genfromtxt(
    "results/Only_benign_updates_20221212_140744/accs.csv", delimiter=","
)
no_mali_acc = np.genfromtxt(
    "results/Only_benign_updates_20221212_141046_mali_0/accs.csv", delimiter=","
)
no_defense_acc_mn = np.mean(no_defense_acc) * np.ones(threshold_vec.shape)
only_ben_acc_mn = np.mean(only_ben_acc) * np.ones(threshold_vec.shape)
no_mali_acc_mn = np.mean(no_mali_acc) * np.ones(threshold_vec.shape)

no_defense_acc_md = np.median(no_defense_acc) * np.ones(threshold_vec.shape)
only_ben_acc_md = np.median(only_ben_acc) * np.ones(threshold_vec.shape)
no_mali_acc_md = np.median(no_mali_acc) * np.ones(threshold_vec.shape)

f, axarr = plt.subplots(1, 3)
f.suptitle(
    "Accuracy after Group testing after MCits = "
    + str(MCits)
    + "Test flip prob: "
    + str(p)
)
y_axismn = np.mean(my_data, axis=1)
ylabelmn = "Mean accuracy"
y_axismd = np.median(my_data, axis=1)
ylabelmd = "Median accuracy"

axarr[0].plot(threshold_vec, y_axismn, label="GT FW")
axarr[0].plot(threshold_vec, no_mali_acc_mn, label="no_mali")
axarr[0].plot(threshold_vec, only_ben_acc_mn, label="oracle")
axarr[0].plot(threshold_vec, no_defense_acc_mn, label="No_defense")
axarr[0].set(xlabel="Gianluigi $Lambda$")
axarr[0].set(ylabel=ylabelmn)
axarr[0].legend()
axarr[1].plot(threshold_vec, y_axismd, label="GT FW")
axarr[1].plot(threshold_vec, no_mali_acc_md, label="no_mali")
axarr[1].plot(threshold_vec, only_ben_acc_md, label="oracle")
axarr[1].plot(threshold_vec, no_defense_acc_md, label="No_defense")
axarr[1].set(xlabel="Gianluigi $Lambda$")
axarr[1].set(ylabel=ylabelmd)
axarr[1].legend()
axarr[2].plot(P_FA, 1 - P_MD, label="ROC curve")
axarr[2].set(xlabel="P_FA$")
axarr[2].set(ylabel="1 - P_MD")
axarr[2].legend()
# axarr[1].ylabel(ylabelmd)
plt.show()
