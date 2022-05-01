#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 18 09:07:52 2021

@author: Juan Abejar
"""
import numpy as np
import matplotlib.pyplot as plt
import pyvista as pv
import pyvistaqt as pvqt
from threading import Thread
import time
Xf = []
Yf = []
Zf = []



Xgcf = []
Tgcf = []

TF = []

XYZs1 = []
XYZs2 = []
XYZs3 = []

Xff = []
Yff = []
Zff = []
COL = [".r",".g",".b",".y"]
for K in range(4):
    fi = open("TXYZ_%003i.dat"%K,"r")
    fiPos = open("Position_Time_LOG%0003i.dat"%K,"r")
    
    XYZ1f = []
    XYZ2f = []
    XYZ3f = []
    
    XYZ1 = []
    XYZ2 = []
    XYZ3 = []
    T = []
    Tf = []
    Xgc = []
    Tgc = []
    
    
    Tsub = 0
    
    for i, line in enumerate(fiPos):
        if i == 0:
            Tsub = float(line.split(":")[-1])
        if float(line.split(":")[-1]) - Tsub >5.0 and float(line.split(":")[-1]) - Tsub <20.0:
            Tgc.append(float(line.split(":")[-1]) - Tsub)
            Xgc.append(float(line.split("|")[1].split(",")[2]))
        Tgcf.append(Tgc)
        Xgcf.append(Xgc)
        
        
        
    for i, line in enumerate(fi):
        XYZ = line.split("[")[1].split("]")[0].split(" ")
        if i == 0:
            Tsub = float(line.split("]")[1])
        if i%3 == 0:
            
            for val in XYZ:
                if val != '':
                    XYZ1.append(float(val))
        if i%3 == 1:
            
            for val in XYZ:
                if val != '':
                    XYZ2.append(float(val))
                    
            
        if i%3 == 2:
            T.append(float(line.split("]")[1])- Tsub-10.0)
            for val in XYZ:
                if val != '':
                    XYZ3.append(float(val))
        
        if XYZ1 != []:
            XYZ1f.append(XYZ1)
            XYZ1 = []
        if XYZ2 != []  :
            XYZ2f.append(XYZ2)
            XYZ2 = []
        if XYZ3 != []:
            XYZ3f.append(XYZ3)
            XYZ3 = []
    #print(XYZ1f)
                    
    for i in range(len(XYZ1f)-2):
        
        if i == 0:
            Tsub = T[i]
        if T[i] -Tsub >15.0 and T[i]-Tsub <30.0:
           
            Tf.append(T[i])
            XYZ1.append(XYZ1f[i])
            XYZ2.append(XYZ2f[i])
            XYZ3.append(XYZ3f[i])
    XYZ1f = []
    XYZ2f = []
    XYZ3f = []
    
    XYZ1f = np.array(XYZ1)
    XYZ2f = np.array(XYZ2)
    XYZ3f = np.array(XYZ3)
    Tf = np.array(Tf)
    
    XYZs1.append(XYZ1f)
    XYZs2.append(XYZ2f)
    XYZs3.append(XYZ3f)
    TF.append(Tf)
Tag1 = []
Tag2 = []
Tag3 = []
Tag4 = []

Tag1 =[ ( XYZs1[0][:,0] + XYZs2[0][:,0] + XYZs3[0][:,0] )/3.0,
           ( XYZs1[0][:,1] + XYZs2[0][:,1] + XYZs3[0][:,1] )/3.0,
               ( XYZs1[0][:,2] + XYZs2[0][:,2] + XYZs3[0][:,2] )/3.0 ]

Tag2=[ ( XYZs1[1][:,0] + XYZs2[1][:,0] + XYZs3[1][:,0] )/3.0,
           ( XYZs1[1][:,1] + XYZs2[1][:,1] + XYZs3[1][:,1] )/3.0,
               ( XYZs1[1][:,2] + XYZs2[1][:,2] + XYZs3[1][:,2] )/3.0 ]

Tag3 =[ ( XYZs1[2][:,0] + XYZs2[2][:,0] + XYZs3[2][:,0] )/3.0,
           ( XYZs1[2][:,1] + XYZs2[2][:,1] + XYZs3[2][:,1] )/3.0,
               ( XYZs1[2][:,2] + XYZs2[2][:,2] + XYZs3[2][:,2] )/3.0 ]

Tag4 =[ ( XYZs1[3][:,0] + XYZs2[3][:,0] + XYZs3[3][:,0] )/3.0,
           ( XYZs1[3][:,1] + XYZs2[3][:,1] + XYZs3[3][:,1] )/3.0,
               ( XYZs1[3][:,2] + XYZs2[3][:,2] + XYZs3[3][:,2] )/3.0 ]

#linear Fits
Xff = []
Yff = []
Zff = []
for i in range(3):
    fit1 = []
    fit2 = []
    fit3 = []
    fit4 = []
    PF1 = np.polyfit(TF[0],Tag1[i],1)
    PF2 = np.polyfit(TF[1],Tag2[i],1)
    PF3 = np.polyfit(TF[2],Tag3[i],1)
    PF4 = np.polyfit(TF[3],Tag4[i],1)
    
    for num in TF[0]:
        fit1.append(float(PF1[0])*num + float(PF1[1]))
    for num in TF[1]:
        fit2.append(float(PF2[0])*num + float(PF2[1]))
    for num in TF[2]:
        fit3.append(float(PF3[0])*num + float(PF3[1]))
    for num in TF[3]:
        fit4.append(float(PF4[0])*num + float(PF4[1]))
    if i == 0:
        Xff.append(fit1)
        Xff.append(fit2)
        Xff.append(fit3)
        Xff.append(fit4)
    if i == 1:
        Yff.append(fit1)
        Yff.append(fit2)
        Yff.append(fit3)
        Yff.append(fit4)
    if i == 2:
        Zff.append(fit1)
        Zff.append(fit2)
        Zff.append(fit3)
        Zff.append(fit4) 
    
    
Xff = np.array(Xff)
Yff = np.array(Yff)
Zff = np.array(Zff)
    #print(XYZ2)
plt.figure(0)
plt.subplot(4,1,1)
plt.plot(TF[0],Tag1[0],COL[0])
plt.plot(TF[1],Tag2[0],COL[1])
plt.plot(TF[2],Tag3[0],COL[2])
plt.plot(TF[3],Tag4[0],COL[3])

plt.plot(TF[0],Xff[0],"-k")
plt.plot(TF[1],Xff[1],"-k")
plt.plot(TF[2],Xff[2],"-k")
plt.plot(TF[3],Xff[3],"-k")
plt.ylabel("X")

plt.subplot(4,1,2)
plt.plot(TF[0],Tag1[1],COL[0])
plt.plot(TF[1],Tag2[1],COL[1])
plt.plot(TF[2],Tag3[1],COL[2])
plt.plot(TF[3],Tag4[1],COL[3])

plt.plot(TF[0],Yff[0],"-k")
plt.plot(TF[1],Yff[1],"-k")
plt.plot(TF[2],Yff[2],"-k")
plt.plot(TF[3],Yff[3],"-k")
plt.ylabel("Y")

plt.subplot(4,1,3)
plt.plot(TF[0],Tag1[2],COL[0])
plt.plot(TF[1],Tag2[2],COL[1])
plt.plot(TF[2],Tag3[2],COL[2])
plt.plot(TF[3],Tag4[2],COL[3])

plt.plot(TF[0],Zff[0],"-k")
plt.plot(TF[1],Zff[1],"-k")
plt.plot(TF[2],Zff[2],"-k")
plt.plot(TF[3],Zff[3],"-k")
plt.ylabel("Z")

plt.subplot(4,1,4)
plt.plot(Tgcf[0],Xgcf[0],".k")
plt.ylabel("X")
plt.xlabel("Time[s]")





minV=[-592.805,-121.364,635.751]
maxV=[775.765,-4.72084,1468.98]




plotter=pvqt.BackgroundPlotter()
#plotter.open_movie("movie.mp4",framerate = 30)   
boundMesh=pv.Box([minV[0],maxV[0],minV[1],maxV[1],minV[2],maxV[2]])
plotter.add_mesh(boundMesh,style="wireframe", color="w")

plotter.add_axes()
plotter.show_bounds()   
plotter.view_xy()
plotter.render()
plotter.show()
time.sleep(3)

p1 = pv.Sphere(12.5,center=(0, 0, 0))
p2 = pv.Sphere(12.5,center=(0, 0, 0))
p3 = pv.Sphere(12.5,center=(0, 0, 0))
p4 = pv.Sphere(12.5,center=(0, 0, 0))

prev = -340.0
COL = 'rgby'
plotter.add_mesh(p1,color=COL[0])
plotter.add_mesh(p2,color=COL[1])
plotter.add_mesh(p3,color=COL[2])
plotter.add_mesh(p4,color=COL[3])

def lines_from_points(points):
    """Given an array of points, make a line set"""
    poly = pv.PolyData()
    poly.points = points
    cells = np.full((len(points)-1, 3), 2, dtype=np.int_)
    cells[:, 1] = np.arange(0, len(points)-1, dtype=np.int_)
    cells[:, 2] = np.arange(1, len(points), dtype=np.int_)
    poly.lines = cells
    return poly
newarr = []
line1 = []
for j in range(4):
    for i in range(len(TF[j])):
        
        newarr.append([Xff[j][i],Yff[j][i],Zff[j][i]])
    newarr = np.array(newarr)
    line1.append(lines_from_points(newarr))
    plotter.add_mesh(line1[j],color=COL[j])
    newarr = []

def RUN():
    time.sleep(10)
    J = 0
    p1p = []
    p2p = []
    p3p = []
    p4p = []
    

    for i in range(3):
        p1p.append(0)
        p2p.append(0)
        p3p.append(0)
        p4p.append(0)
        
    while J < 2000:
        
        if J%1 == 0:
            try:
                p1.translate( [Tag1[0][J] - p1p[0],Tag1[1][J] - p1p[1],Tag1[2][J] - p1p[2]] )
                p1p = [ Tag1[0][J] , Tag1[1][J] , Tag1[2][J] ] 
            except:
                print("end1")
            try:
                p2.translate( [Tag2[0][J] - p2p[0],Tag2[1][J] - p2p[1],Tag2[2][J] - p2p[2]] )
                p2p = [ Tag2[0][J] , Tag2[1][J] , Tag2[2][J] ] 
            except:
                print("end1")
            try:
                p3.translate( [Tag3[0][J] - p3p[0],Tag3[1][J] - p3p[1],Tag3[2][J] - p3p[2]] )
                p3p = [ Tag3[0][J] , Tag3[1][J] , Tag3[2][J] ] 
            except:
                print("end1")
            try:
                p4.translate( [Tag4[0][J] - p4p[0],Tag4[1][J] - p4p[1],Tag4[2][J] - p4p[2]] )
                p4p = [ Tag4[0][J] , Tag4[1][J] , Tag4[2][J] ]
            except:
                print("end1")
            
            
            time.sleep(0.002)
            plotter.update()
           
                
                
        J+=1
   
thread = Thread(target=RUN)
thread.start()  

        
        
        
        
        


    