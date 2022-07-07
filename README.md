#  Module_Tracking_Statistical_Analysis
The purpose of this code is to quickly return feedback for data obtained from tracking modules, effectively measuring the error correlated with each of the 3 cartesian dimensions [X,Y,Z].

# Sample Results

The data for the tracking system used to triangulate position for 4 individual modules is shown below

![alt text](https://github.com/jp-abejar/Module_Tracking_Statistical_Analysis/blob/main/img/Figure_1.png?raw=true)

In this example, We can se that we have high reliability in the measurements of the X and Y positioning of each module, all but 1 showing an R^2 value of 0.98 or greater. The Z component measurements are not as reliable but based on known positioning.

The PyVista liobrary allows for visual comparison of the expected trajectories for each module. We know that the modules were translated in linear trajectories through diamond shaped positioning in the Y-Z plane. 


![alt text](https://github.com/jp-abejar/Module_Tracking_Statistical_Analysis/blob/main/img/fig2.png?raw=true)
