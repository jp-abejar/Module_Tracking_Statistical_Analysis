#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 18 09:07:52  2521

@author: Juan Abejar
"""
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pyvista as pv
import pyvistaqt as pvqt
from threading import Thread
import time
from scipy.stats import linregress

""" This software is designed as an initial test for tracking data provided by in development tracking modules.
    """


def ParseGcodeData(file):
    ''' Return the time and position parsed from the file provided.
        This is data interpreted from a GRBL controller as the device sustaining the tag was translated in the x-direction.
        This only works for a specific text file format'''
    
    Xgc = [] # Initialize list for position data
    Tgc = [] # Initialize list for time data
    for j, line in enumerate(fiPos):
        
        if j == 0:
            Tsub = float(line.split(":")[-1])
        # We only obtain data captured while in a constant velocity (between 5 and 20 seconds)
        if float(line.split(":")[-1]) - Tsub >5.0 and float(line.split(":")[-1]) - Tsub < 20.0:
            Tgc.append(float(line.split(":")[-1]) - Tsub)
            Xgc.append(float(line.split("|")[1].split(",")[2]))
            
    return Tgc,Xgc


def ParseTrackingData(file):
    ''' Return the time and position parsed from the file provided.
        This is data interpreted from in-development tracking modules, which provide x,y, and z positioning for 3 individual sensors
        placed in a triagular arrangement. This data has been pre-processed using a preliminary version of a kalman filter
        so most of the outliers have been omitted.
        This only works for a specific text file format'''
    # These lists will be temporary holders of iterative data extrracted from lines of the file
    XYZ1 = []
    XYZ2 = []
    XYZ3 = []
    
    
    # These lists will have the cumulative data stored
    XYZ1f = []
    XYZ2f = []
    XYZ3f = []
    
    T = []
    for  k, line in enumerate(file):
        
        xyzL = line.split("[")[1].split("]")[0].split(" ")
        
        if k == 0:
            Tsub = float(line.split("]")[1])
            
        if k%3 == 0:
            for val in xyzL:
                if val != '':
                    XYZ1.append(float(val))
            if XYZ1 != []:
                XYZ1f.append(XYZ1)
                XYZ1 = []
                
        if k%3 == 1:
            for val in xyzL:
                if val != '':
                    XYZ2.append(float(val))
            if XYZ2 != []  :
                XYZ2f.append(XYZ2)
                XYZ2 = []
            
        if k%3 == 2:
            T.append(float(line.split("]")[1])- Tsub-10.0)
            for val in xyzL:
                if val != '':
                    XYZ3.append(float(val))
            if XYZ3 != []:
                XYZ3f.append(XYZ3)
                XYZ3 = []
        
    minV = min(len(XYZ1f),len(XYZ2f),len(XYZ3f)) # For truncating the data to the smallest sized list to facilitate array conversion
    
    pos = (np.array(XYZ1f)[25:minV-25].T + np.array(XYZ2f)[ 25:minV- 25].T  + np.array(XYZ3f)[ 25:minV- 25].T)/3 # Get the midpoint of the 3 sensors
    Time = (np.array(T) - T[0])[ 25:minV- 25].T #Remove the time offset and truncate to minimize idle state
    
    return Time,pos 


def LinearFit(Tf,Tags,minTime, MaxTime,numTags):
    ''' This function utilizes linregress from the scipy.stats library to apply linear regression to our 3D positional data.
    
        fitParams consists of 5 parameters: slope, intercept, rvalue, pvalue, standard error, and intercept standard error.
        
        fits is the fitted data for all sensors uning the fitParams values.
        We also return the indexes for the desired time interval.
        '''
    fitParams = np.zeros((numTags,5,3))
    fits = [[] for i in range(numTags)]
    TfStartIndx = [np.where(Tf[i]//1 == minTime)[0][0] for i in range(numTags)]
    TfStopIndx = [np.where(Tf[i]//1 == maxTime)[0][0] for i in range(numTags)]


    for i in range(numTags):
        for n in range(3):
            fitParams[i,:,n]= linregress(Tf[i][TfStartIndx[i]:TfStopIndx[i]], Tags[i][n][TfStartIndx[i]:TfStopIndx[i]])
            
            fits[i].append(fitParams[i,0,n] * Tf[i] + fitParams[i,1,n])
        fits[i] = np.array(fits[i])
            
    return fitParams,fits,TfStartIndx,TfStopIndx


def lines_from_points(points):
    """Given an array of points, make a line set"""
    poly = pv.PolyData()
    poly.points = points
    cells = np.full((len(points)-1, 3), 2, dtype=np.int_)
    cells[:, 1] = np.arange(0, len(points)-1, dtype=np.int_)
    cells[:, 2] = np.arange(1, len(points), dtype=np.int_)
    poly.lines = cells
    return poly


numTags = 4

minTime = 35 # Start Time
maxTime = 50# End Time


COL = sns.color_palette('colorblind',n_colors = numTags) # Create a list of colors 


gcd = [] # For gcode data 
tagDat = [] # To store function data as it comes
Tags = [] # For appending tagDat[1]
Tf = [] # For appending tagDat[0]

for i in range(numTags):
    
    fi = open("TXYZ_%003i.dat"%i,"r")
    fiPos = open("Position_Time_LOG%0003i.dat"%i,"r")
    
    
    gcd.append( ParseGcodeData(fiPos) )

    tagDat = ParseTrackingData(fi) 
    Tags.append(tagDat[1])
    Tf.append(tagDat[0])
    del tagDat
    



fitParams, fits, TfStartIndx, TfStopIndx =LinearFit(Tf,Tags,minTime, maxTime,numTags) # Applying linear regression to all tracking module datasets



plt.figure()
labels = ['X', 'Y', 'Z']
for i in range(numTags):
    for j in range(3):
        plt.subplot(4,1,j+1)
        plt.title('Time vs {} position'.format(labels[j]))
        plt.plot(Tf[i][TfStartIndx[i]:TfStopIndx[i]], Tags[i][j][TfStartIndx[i]:TfStopIndx[i]],color = COL[i],linestyle = 'dotted',label = 'Module {}'.format(i))

        plt.plot(Tf[i][TfStartIndx[i]:TfStopIndx[i]], fits[i][j][TfStartIndx[i]:TfStopIndx[i]],'-k')
        plt.ylabel(labels[j]+'[mm]')
        plt.xlabel("Time[s]")
        plt.legend(fontsize = 8)
        
plt.subplot(4,1,4)
plt.title('Time vs X position (Gcode)')
plt.plot(gcd[0][0],gcd[0][1],".k")
plt.ylabel("X[mm]")
plt.xlabel("Time[s]")
plt.subplots_adjust(top=0.926,
                    bottom=0.121,
                    left=0.066,
                    right=0.977,
                    hspace=0.835,
                    wspace=0.2)


plt.show()


''' The code below is aimed to plot the raw motion of the tracking modules
    over the linear regression outputs for each respective tracking module using pyvista'''
    
plotter=pvqt.BackgroundPlotter()

minV=[-592.805,-121.364,635.751]
maxV=[775.765,-4.72084,1468.98]

boundMesh=pv.Box([minV[0],maxV[0],minV[1],maxV[1],minV[2],maxV[2]])
plotter.add_mesh(boundMesh,style="wireframe", color="w")

plotter.add_axes()
plotter.show_bounds()   
plotter.view_xy()
plotter.render()
plotter.show()


TagPoints = []

# COL = 'rgby'
PrevPoints = []

# This block of code adds the mesh spheres for each tracking module onto the pyvista plotter.
# This also initialized the list of points that will store the previous data position for use
# in the iterative translation process
for num in range(numTags):
    TagPoints.append(pv.Sphere(12.5,center=(0, 0, 0)))
    plotter.add_mesh(TagPoints[num],color=COL[num])
    PrevPoints.append([0,0,0])



newarr = []
line1 = []
# This block of code adds the line fit data to the pyvista plotter
for j in range(numTags):
    
    newarr = np.array(fits[j].T[TfStartIndx[j]:TfStopIndx[j]])
    line1.append(lines_from_points(newarr))
    plotter.add_mesh(line1[j],color='k')
    newarr = []
    
newarr = []
line1 = []
for j in range(numTags):
    
    newarr = np.array(Tags[j].T[TfStartIndx[j]:TfStopIndx[j]*2])
    line1.append(lines_from_points(newarr))
    plotter.add_mesh(line1[j],color=COL[j])
    newarr = []

def RUN():
    ''' This function is meant to run as a thread. 
        It will iterate over the position values for each tracking module and update the mesh sphere
        location using a translate function provided in the pyvista library.
        '''
    time.sleep(1)
    J = 0 # initial value
    step = 5 # Number of values to skip in iteration
    completed = [False for i in range(numTags)] # boolean values that will be updated as each dataset reaches its end (each tag array has a different size)
        
        
    while sum(completed) != numTags:
        
        
        for i in range(numTags):
            
            if not completed[i]:
          
                try:
                    TagPoints[i].translate( [Tags[i][0][int(J*step)] - PrevPoints[i][0], 
                                             Tags[i][1][int(J*step)] - PrevPoints[i][1], 
                                             Tags[i][2][int(J*step)] - PrevPoints[i][2]] )
                    PrevPoints[i]= [Tags[i][0][int(J*step)],
                                    Tags[i][1][int(J*step)],
                                    Tags[i][2][int(J*step)]] 
                except:
                    completed[i] = True
                
           
            
            time.sleep(0.002)
            plotter.update()
           
                
                
        J+=1
    
thread = Thread(target=RUN) # Assign the thread
thread.start()  # Run the thread




