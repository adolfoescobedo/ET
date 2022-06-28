import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import json
import matplotlib.cm as cm
from urllib.request import urlopen
import seaborn as sns; sns.set_theme()
import pandas as pd

def cmap_to_rgb(img):
    return cm.viridis(img/np.amax(img))[:,:,:-1]

def cmap_to_rgb_gray(img):
    return cm.gray(img/np.amax(img))[:,:,:-1]

#with open('11-08-2021Test4_4_1.json') as f: #data file location
#    data = json.load(f)

def cmap_to_rgb(img):
    return cm.viridis(img/np.amax(img))[:,:,:-1]

def cmap_to_rgb_gray(img):
    return cm.gray(img/np.amax(img))[:,:,:-1]

# Doesn't show anything
# print("hello")
#
# im = plt.imread("Playground.png")
#
# plt.show()



# uniform_data = np.random.rand(12, 10)
#
# ax = sns.heatmap(uniform_data)
#
# plt.show()


#gbvs_data = np.empty([10,10], dtype=float)

#print(gbvs_data)

df = pd.read_csv('gbvs_out.txt', sep=' ',header=None,names=None)

ax = sns.heatmap(df, cmap="PiYG")
#ax = sns.heatmap(df, linewidths=60)

plt.show()

#print(df)


#
#
# ax = sns.heatmap(uniform_data)
#
# plt.show()


# print("hello")
#
# plt.figure().clear()
#
# im=plt.figure(1)
#
# im = plt.imread("Playground.png")
#
# plt.show()


#for i in range(0,499):
#    for j in range(0,499):
#        plt.plot(i, j, "og", markersize=10) #plot points

# Tester plots
# x = np.arange(0, 500, 1)
# y = np.sin(x)
# fig, ax = plt.subplots()
# ax.plot(x, y)
#
# plt.show()


