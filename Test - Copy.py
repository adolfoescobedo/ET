import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import json
import matplotlib.cm as cm
from urllib.request import urlopen
import seaborn as sns; sns.set_theme()
import pandas as pd

class Fixations:
    def __init__(self):
        self.num_fixations = 0
        self.fixation_start = []
        self.fixation_end = []
        self.durations = []


    def __init__(self, total_fixations):
        self.num_fixations = total_fixations
        self.durations = [0 for i in range(self.num_fixations)]
        self.fixation_start = [0 for i in range(self.num_fixations)]
        self.fixation_end = [0 for i in range(self.num_fixations)]

    def get_durations(self, data_frame):
        print(data_frame.at[3, 'Timestamp'])

        fixation_counter = 0
        #fixation_start = 0
        #fixation_end = 0

        new_fixation = False

        temp_time_stamps = [0]
        for i in range(200):
            if data_frame.at[i, 'Fixation Index'] > 0:
                print("Row " + str(i + 1))
                if (int(data_frame.at[i, 'Fixation Index']) == fixation_counter and (new_fixation == False)):
                    #print("Row " + str(i+1))
                    new_fixation = True
                    self.fixation_start[fixation_counter] = int(data_frame.at[i, 'Timestamp'])
                    self.fixation_end[fixation_counter] = int(data_frame.at[i, 'Timestamp'])
                    j = 0

                # while(new_fixation==True):
                #    j = j+1
                #    if data_frame.at[i+j, 'Fixation Index'] > 0:
                #        self.fixation_end[fixation_counter] = int(data_frame.at[i+j, 'Timestamp'])
                #    else:
                #         new_fixation = False
                #         self.durations[fixation_counter] = self.fixation_end[fixation_counter]  - self.fixation_start[fixation_counter]
                #         fixation_counter = fixation_counter + 1

        print(self.durations)

                #temp_time_stamps.clear()

                #while()



            #print(len(data_frame))


        #for i in range(self.num_fixations):
        #    j = 0
        #    while(int(data_frame.at[j,"Fixation Index"]) == )




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




## ARE - Save as example of working heatmap

# df = pd.read_csv('gbvs_out.txt', sep=' ',header=None,names=None)
#
# ax = sns.heatmap(df, cmap="PiYG")
# #ax = sns.heatmap(df, linewidths=60)
#
# plt.show()


#df = pd.read_csv('NE_Rec4_IVT_iMotions.csv', dtype=float)

df = pd.read_csv('NE_Rec4_IVT_iMotions_shorter.csv', dtype={"Row": int, "Timestamp": int})
print(df.at[0, 'Timestamp']) #ARE - This commands prints '5'

df_ext = pd.read_csv('NE_Rec4_IVT_iMotions.csv', skiprows= 24, usecols=['Timestamp', 'Gaze X', 'Gaze Y', 'Interpolated Gaze X',
                                                                    'Interpolated Gaze Y', 'Gaze Velocity', 'Gaze Acceleration',
                                                                    'Fixation Index', 'Fixation X', 'Fixation Y', 'Fixation Start',
                                                                    'Fixation End',	'Fixation Duration', 'Fixation Dispersion',
                                                                    'Saccade Index', 'Saccade Index by Stimulus', 'Saccade Start',
                                                                    'Saccade End',	'Saccade Duration',	'Saccade Amplitude',
                                                                    'Saccade Peak Velocity', 'Saccade Peak Acceleration',
                                                                    'Saccade Peak Deceleration', 'Saccade Direction'],
                 dtype={'Timestamp': int})

print(df_ext.at[0, 'Timestamp']) #ARE - This commands prints '5'

print(df_ext.head())

fixation_data = Fixations(6209)

print(fixation_data.durations)
fixation_data.get_durations(df_ext)

#print(type(df_ext.at[0, 'Timestamp']))
#print(type(df_ext.at[0, 'Gaze X']))
#print(type(df_ext.at[0, 'Fixation X']))

#print(pd.options.display.max_rows)# ARE - Max number of df rows printed to screen when print is executed
#print(df.head())# ARE - prints header and first fine lines of data frame

#for i in range(len(df_ext)):
#    if(df_ext.at[i,'Fixation Index'] > 0):
#        print(i)



#fixation_times = [0 for i in range(6209)]

#print(fixation_times)