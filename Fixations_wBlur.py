import matplotlib.image as mpimg
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.spatial import distance
import json
import matplotlib.cm as cm
from urllib.request import urlopen
from statistics import mean
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

        self.duration_calculated = []  ## Calculates fixation duration from the rows (it appears to be an underestimate of the actual duration)
        self.start_frame = [] ## Stores the number of the video frame when the fixation starts
        self.end_frame = [] ## Stores the number of the video frame when the fixation ends

    def __init__(self, num_fixations_):
        self.num_fixations = num_fixations_
        self.start_row = [0 for i in range(self.num_fixations)]
        self.end_row = [0 for i in range(self.num_fixations)]
        self.duration = [0.0 for i in range(self.num_fixations)]
        self.dispersion = [0.0 for i in range(self.num_fixations)]
        self.xy = [[0.0, 0.0] for i in range(self.num_fixations)]

        self.duration_calculated = [0.0 for i in range(self.num_fixations)]
        self.start_frame = [0 for i in range(self.num_fixations)]
        self.end_frame = [0 for i in range(self.num_fixations)]

    def get_eye_tracking_data(self, data_frame):

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
                        self.duration_calculated[fixation_counter - 1] = fixation_end - fixation_start
                        fixation_counter = fixation_counter + 1

        ## Get recorded durations
        for i in range(self.num_fixations):
            self.duration[i] = data_frame.at[self.start_row[i], 'Fixation Duration']

        ## Get starting/ending frames of each fixation; first extract what second it is and multiply by 30; then calculate how many 30ths of a second there are in the remainder
        for i in range(self.num_fixations):
            self.start_frame[i] = 30 * int(data_frame.at[self.start_row[i], 'Timestamp'] / 1000)
            self.start_frame[i] = self.start_frame[i] + int(
                int(data_frame.at[self.start_row[i], 'Timestamp']) % 1000 / 33.33)

            self.end_frame[i] = 30 * int(data_frame.at[self.end_row[i], 'Timestamp'] / 1000)
            self.end_frame[i] = self.end_frame[i] + int(
                int(data_frame.at[self.end_row[i], 'Timestamp']) % 1000 / 33.33)

        ## Get x,y coordinates for each fixation
        for i in range(self.num_fixations):
            self.xy[i] = [float(data_frame.at[self.start_row[i], 'Fixation X']), float(data_frame.at[self.start_row[i], 'Fixation Y'])]

        ## Get dispersions for each fixation
        for i in range(self.num_fixations):
            self.dispersion[i] = float(data_frame.at[self.start_row[i], 'Fixation Dispersion'])

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

df_ext = pd.read_csv('NE_Rec4_IVT_iMotions.csv', skiprows= 24, usecols=['Timestamp', 'Gaze X', 'Gaze Y', 'Interpolated Gaze X',
                                                                    'Interpolated Gaze Y', 'Gaze Velocity', 'Gaze Acceleration',
                                                                    'Fixation Index', 'Fixation X', 'Fixation Y', 'Fixation Start',
                                                                    'Fixation End',	'Fixation Duration', 'Fixation Dispersion',
                                                                    'Saccade Index', 'Saccade Index by Stimulus', 'Saccade Start',
                                                                    'Saccade End',	'Saccade Duration',	'Saccade Amplitude',
                                                                    'Saccade Peak Velocity', 'Saccade Peak Acceleration',
                                                                    'Saccade Peak Deceleration', 'Saccade Direction'],
                 dtype={'Timestamp': int})

## Initialize fixation_data object by passing number of fixations in the input file (inspect Excel ahead of time to get this number)
fixation_data = Fixations(6209)

## Retrieve instance data from Excel file
fixation_data.get_eye_tracking_data(df_ext)

#for i in range(fixation_data.num_fixations):
#for i in range(665):
    #print("Fixation #" + str(i+1) + ". Calculated Duration:" + str(fixation_data.duration_calculated[i]) + ". Actual Duration:" + str(fixation_data.duration[i]))
    #print("Fixation #" + str(i+1) + ". Frame start:" + str(fixation_data.start_frame[i]) + ". Frame end:" + str(fixation_data.end_frame[i]) + ".")
    #print("Fixation #" + str(i+1) + ". (" + str(fixation_data.xy[i][0]) + ", " + str(fixation_data.xy[i][1]) + ")")
    #print("Fixation #" + str(i+1) + ". Dispersion:" + str(fixation_data.dispersion[i]))

## Create and display heat map of the discrete fixations
heat_map_data = np.zeros((1080,1920))

## According to PDF, each visual degree corresponds to 30-60 pixels
radius = 45 #30*2
blur_thresholds = [1,2,3]

###Adjustments to coordinates needed to draw a circle from each fixation coordinate
coord_offset = ([])

for x_offset in range(radius+1):
    for y_offset in range(radius+1):
        a = np.array([x_offset,y_offset])
        if(np.linalg.norm(a) <= radius):
            coord_offset.append([-x_offset, -y_offset])
            coord_offset.append([x_offset, y_offset])

        a = np.array([-x_offset, y_offset])
        if (np.linalg.norm(a) <= radius):
            coord_offset.append([-x_offset, y_offset])
            coord_offset.append([x_offset, -y_offset])


coord = np.array([0,0])

### Need to associated a blur level wtih each coordinate within a certain distance of each fixation point
##blur_data = set([])

for i in range(60): #range(fixation_data.num_fixations):# range(1):
    #print(int(fixation_data.xy[i][1]),int(fixation_data.xy[i][0]))
    fixation_coords = np.array([int(fixation_data.xy[i][1]),int(fixation_data.xy[i][0])])

   #print(int(fixation_data.xy[0][1]), int(fixation_data.xy[0][0]))
    for offset in coord_offset:
        coord = np.array([int(fixation_data.xy[i][1]) + offset[0], int(fixation_data.xy[i][0]) + offset[1]])
        if (coord[0] >= 0 and coord[0] < 1080 and coord[1] >= 0 and coord[1] < 1920):
            heat_map_data[coord[0],coord[1]] = 1.0




HM = sns.heatmap(heat_map_data, alpha=0.6, zorder=2) ## alpha controls how visible the fixation data is atop the image

map_img = mpimg.imread('Background_Image.PNG')

HM.imshow(map_img,
          aspect = HM.get_aspect(),
          extent = HM.get_xlim() + HM.get_ylim(),
          zorder = 1) ##puts the map under the heatmap

plt.show()

#print(max(fixation_data.duration))
#print(mean(fixation_data.duration))

## Write frame numbers and coordinates to output file (this corresponds to the first five minutes of Anthony's video)
# outF = open("Fixations.txt", "w")
#
# for i in range(665):
#     for j in range(fixation_data.end_frame[i]-fixation_data.start_frame[i]):
#         outF.write(str(fixation_data.start_frame[i]-1+j) + "\t")
#         outF.write(str(fixation_data.xy[i][0]) + "\t")
#         outF.write(str(fixation_data.xy[i][1]) + "\n")
#     #outF.write("\n")
#
# outF.close()