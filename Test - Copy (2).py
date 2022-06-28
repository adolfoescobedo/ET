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
        self.start_row = [] ## Stores the data frame row index when each fixation starts
        self.end_row = [] ## Stores the data frame row index when each fixation ends
        self.duration = [] ## Stores fixation durations recorded by the eye tracking software
        self.dispersion = []  ## Stores fixation dispersions recorded by the eye tracking software
        self.xy = [] ## Stores x and y coordinates of each fixation

        self.durations_calculated = []  ## Calculates fixation duration from the rows (it appears to be an underestimate of the actual duration)
        self.start_frame = [] ## Stores the number of the video frame when the fixation starts
        self.end_frame = [] ## Stores the number of the video frame when the fixation ends




    def __init__(self, num_fixations_):
        self.num_fixations = num_fixations_
        self.start_row = [0 for i in range(self.num_fixations)]
        self.end_row = [0 for i in range(self.num_fixations)]
        self.duration = [0.0 for i in range(self.num_fixations)]
        self.dispersion = [0.0 for i in range(self.num_fixations)]
        self.xy = [[0.0, 0.0] for i in range(self.num_fixations)]

        self.durations_calculated = [0.0 for i in range(self.num_fixations)]
        self.start_frame = [0 for i in range(self.num_fixations)]
        self.end_frame = [0 for i in range(self.num_fixations)]


    def get_durations(self, data_frame):
        print(data_frame.at[3, 'Timestamp'])

        fixation_counter = 1 ## Fixation indices recorded by software start at "1"
        fixation_start = 0
        fixation_end = 0

        ## Helps identify when a fixation starts and ends
        new_fixation = False

        for i in range(len(data_frame)):
            ## Check only rows that have positive number under fixation index column
            if data_frame.at[i, 'Fixation Index'] > 0:
                ## Identifies start of new fixation
                if (int(data_frame.at[i, 'Fixation Index']) == fixation_counter and (new_fixation == False)):
                    new_fixation = True
                    fixation_start = int(data_frame.at[i, 'Timestamp'])
                    fixation_end = int(data_frame.at[i, 'Timestamp'])

                    self.start_row[fixation_counter - 1] = i
                    self.end_row[fixation_counter - 1] = i
                    j = 0

                ## Keep scanning rows until a nonzero appears under the fixation index column
                ## When rows stopped being scanned, signal end of fixation and move up fixation counter
                while(new_fixation==True):
                   j = j+1
                   if data_frame.at[i+j, 'Fixation Index'] > 0:
                       fixation_end = int(data_frame.at[i+j, 'Timestamp'])
                       self.end_row[fixation_counter - 1] = i + j
                   else:
                        new_fixation = False
                        self.durations_calculated[fixation_counter - 1] = fixation_end - fixation_start
                        fixation_counter = fixation_counter + 1

        for i in range(self.num_fixations):
            self.duration[i] = data_frame.at[self.start_row[i], 'Fixation Duration']

    ## Calcualtes the starting/ending frames of each fixation; first extract what second it is and multiply by 30; then calculate how many 30ths of a second there are in the remainder
    def get_frame_indices(self, data_frame):
        for i in range(self.num_fixations):
            self.start_frame[i] = 30*int(data_frame.at[self.start_row[i], 'Timestamp']/1000)
            self.start_frame[i] = self.start_frame[i] + int(int(data_frame.at[self.start_row[i], 'Timestamp'])%1000/33.33)

            self.end_frame[i] = 30*int(data_frame.at[self.end_row[i], 'Timestamp'] / 1000)
            self.end_frame[i] = self.end_frame[i] + int(int(data_frame.at[self.end_row[i], 'Timestamp'])%1000/33.33)

    ## Obtain x,y coordinates for each fixation
    def get_xy(self, data_frame):
        for i in range(self.num_fixations):
            self.xy[i] = [float(data_frame.at[self.end_row[i], 'Fixation X']), float(data_frame.at[self.end_row[i], 'Fixation Y'])]

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

#print(df_ext.at[0, 'Timestamp']) #ARE - This commands prints '5'

#print(df_ext.head())

fixation_data = Fixations(6209)

fixation_data.get_durations(df_ext)
fixation_data.get_frame_indices(df_ext)
fixation_data.get_xy(df_ext)


#print(fixation_data.start_row)
#print(fixation_data.end_row)
#print(fixation_data.durations_calculated)

#print(min(fixation_data.durations_calculated))
#print(fixation_data.durations)

for i in range(fixation_data.num_fixations):
    #print("Fixation #" + str(i+1) + ". Calculated Duration:" + str(fixation_data.durations_calculated[i]) + ". Actual Duration:" + str(fixation_data.durations[i]))
    #print("Fixation #" + str(i+1) + ". Frame start:" + str(fixation_data.start_frame[i]) + ". Frame end:" + str(fixation_data.end_frame[i]))
    print("Fixation #" + str(i+1) + ": " + str(fixation_data.xy[i]))
    print("Fixation #" + str(i+1) + ". (" + str(fixation_data.xy[i][0]) + ", " + str(fixation_data.xy[i][1]) + ")")

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