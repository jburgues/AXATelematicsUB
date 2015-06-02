# -*- coding: utf-8 -*-
"""
Created on Wed Apr 01 09:44:59 2015

@author: F59JBC0
"""

#%% Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
from sklearn import cross_validation
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
from os import listdir
from os.path import isdir, isfile, join
from scipy.stats import kde
from scipy.stats import chisquare
import random
import sys
sys.setrecursionlimit(10000)

#%%
def plotTrip(driver, trip):
    """Calculate the speed of a trip given its coordinates (x, y) meters.
    The sampling freq is assumed to be 1 sec.

    Parameters
    ----------
    x,y: numpy.array(dtype=float) shape (n_points, )

    Returns
    -------
    speed: numpy.array shape (n_points-1, )
    """

    coords = pd.read_csv('./drivers/' + str(driver) + '/' + str(trip) + '.csv')
    x = np.asarray(coords['x'])
    y = np.asarray(coords['y'])
    
    speed = getSpeed(x, y) # km/h
    accel = np.diff(speed) # km/h
    # Case either speed or acceleration is out of range (abnormal trip)        
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title('Driver: ' + str(driver) + ' / Trip: ' + str(trip) + '.csv', fontsize = 14)
    plt.plot(x,y,'bo')
    plt.plot(x[abs(speed)>300],y[abs(speed)>300],'ro')
    plt.plot(x[abs(accel)>50],y[abs(accel)>50],'yo')       
    plt.xlabel('x', fontsize=14)
    plt.ylabel('y', fontsize=14)
    plt.show()
    
    
#%%
def getSpeed(x, y):
    """Calculate the speed of a trip given its coordinates (x, y) meters.
    The sampling freq is assumed to be 1 sec.

    Parameters
    ----------
    x,y: numpy.array(dtype=float) shape (n_points, )

    Returns
    -------
    speed: numpy.array shape (n_points-1, )
    """
    
    dx = np.diff(x) # Differences in 'x'.
    dy = np.diff(y) # Differences in 'y'.  
    
    spd = ((dx**2) + (dy**2))**0.5 # Speed is the euclidean distance of differences.
    
    return 3.6*spd #km/h
    
#%%
def getDistance(spd):
    """Calculate the traveled distance of a trip given its speed (m/s).
    The sampling freq is assumed to be 1 sec.

    Parameters
    ----------
    x,y: numpy.array(dtype=float) shape (n_points, )

    Returns
    -------
    speed: numpy.array shape (n_points-1, )
    """
    
    dist = np.cumsum(spd/float(3600))
    
    return dist # km

#%%
def getClusters1D(feature):
    """Find clusters in a one dimensional signal. 

    Parameters
    ----------
    x: numpy.array(dtype=float) shape (n_points, )

    Returns
    -------
    locs: numpy.array shape (n_divisions, ). division points between the different modes of the
    """
    x = feature['data']
    # Kernel density estimation
    density = kde.gaussian_kde(x)
    
    # Find local minima (x' == 0 && x'' > 0)
    npoints = 1000
    xgrid = np.linspace(0, max(x), npoints)
    delta = max(x)/float(npoints)
    densityx = np.asarray(density(xgrid))
    locs_xgrid = np.r_[True, densityx[1:] < densityx[:-1]] & np.r_[densityx[:-1] < densityx[1:], True]
    locs = (delta * np.asarray(np.where(locs_xgrid == True)))[0]
    
    # Plot histogram + density estimation + selected intervals
    fig = plt.figure()
    # Plot histogram
    plt.hist(x, bins=30, normed=True, histtype="stepfilled")
    # Plot density estimation
    plt.plot(xgrid, densityx, 'r-')
    # Plot vertical lines at the selected intervals
    for i in range(len(locs)):
        plt.axvline(locs[i], color='g', linestyle='dashed', linewidth=2)
    plt.show()
    fig.suptitle('Histogram of trip ' + feature['name'], fontsize=18)
    plt.xlabel(feature['units'], fontsize=18)
    
    return locs

#%%
def plotTripColoredSpd(coords):
    """Find clusters in a one dimensional signal. 

    Parameters
    ----------
    feature: name, units, data

    Returns
    -------
    locs: numpy.array shape (n_divisions, ). division points between the different modes of the
    """
    x = np.asarray(coords['x'])
    y = np.asarray(coords['y'])    
    spd = getSpeed(x, y) #km/h
    norm = colors.Normalize(vmin=0, vmax=max(spd))
    m = cm.ScalarMappable(norm=norm, cmap=cm.hot)
    for i in range(0, len(x)-1):
        plt.plot(x[i:i+2],y[i:i+2], color = m.to_rgba(spd[i]))
        
#%%
def plotSimilarTrips(feature, intervals, trips, max_trips_per_plot, color):
    """Find clusters in a one dimensional signal. 

    Parameters
    ----------
    feature: name, units, data

    Returns
    -------
    locs: numpy.array shape (n_divisions, ). division points between the different modes of the
    """
    
    x = feature['data']
    for i in range(0, len(intervals)-1):
        # Find which trips are contained in the interval [i, i+1]
        indices_interval = np.where((x > intervals[i]) & (x < intervals[i+1]))[0] # [0] transforms a column vector to a row vector
        num_trips_interval = len(indices_interval)
        # The interval should contain at least one trip
        if (num_trips_interval > 0):
            # Limit the number of trips to be plotted to 'max_trips_per_plot'
            num_trips_plot = min(num_trips_interval, max_trips_per_plot)
            # Randomly select 'num_trips_plot' from the indices of this interval
            random_selector = random.sample(indices_interval, num_trips_plot)
            # Plot
            fig = plt.figure()
            for idx in random_selector:
                if (color == False):
                    plt.plot(trips[idx]['x'], trips[idx]['y'])
                else:
                    plotTripColoredSpd(trips[idx])
            plt.show()
            # Add legend to plot
            fig.suptitle('Trips with ' + feature['name'] + ' in the interval [' + str(intervals[i]) + ', ' + str(intervals[i+1]) + '] ' + feature['units'], fontsize=18)
            
#%%
def sampleTripPoints(trip_dist, delta_km):
    """Calculate the traveled distance of a trip given its speed (m/s).
    The sampling freq is assumed to be 1 sec.

    Parameters
    ----------
    x,y: numpy.array(dtype=float) shape (n_points, )

    Returns
    -------
    speed: numpy.array shape (n_points-1, )
    """

    # Find points separated window_dist from each other
    list_points_curv = [0] # List starts with the first point
    end = False
    i = 0
    while(end == False):
        list_points_after = np.where(trip_dist >= trip_dist[i] + delta_km)[0]
        # Case when exist at least one point after the current one
        if (len(list_points_after) > 0):
            point_after = list_points_after[0] #[0] returns the first element that meets the condition
            if (point_after != (len(trip_dist)-1)):
                list_points_curv.append(point_after)
            # The list of sampled points should not contain the end point of the trip
            else:
                list_points_curv.append(point_after-1)
            i = point_after
        else:
            end = True
    
    return list_points_curv
   
def findMissingData(x, y):
    """
    Returns a boolean array with True if points are outliers and False 
    otherwise.
    """

    speed = getSpeed(x, y) # km/h
    accel = np.diff(speed) # km/h
    
    # Case either speed or acceleration is out of range (abnormal trip)                   
    return np.where((abs(speed[:-1]) > 300) & (abs(accel) > 50))
    
#%%
def filterCurves(listOfCurves, missingDataLocs):
    """Calculate the traveled distance of a trip given its speed (m/s).
    The sampling freq is assumed to be 1 sec.
    
    Curvature = |x'y\" -y'x\"| /((x'**2 + y'**2)**(3/2))
    """
    
    valid_indices = range(0, len(listOfCurves))
    invalid_indices = []
    #print "valid_indices: ", valid_indices
    #print "missingDataLocs: ", missingDataLocs[0]
    for missing_point in missingDataLocs[0]:
        for c in range(0, len(listOfCurves)):
            curve = listOfCurves[c]
            # Case missing point is contained inside the curve
            #print "miss: ", missing_point, " - curve0: ", curve[0], " - curve2: ", curve[2]
            if (missing_point >= curve[0]) & (missing_point <= curve[2]):
                invalid_indices.append(c)# invalid curve
                
    valid_indices = list(set(valid_indices) - set(invalid_indices))
    return [listOfCurves[i] for i in valid_indices]
    
#%%
def getCurves(coords, trip_dist, curve_window_km, cur_thr):
    """Calculate the traveled distance of a trip given its speed (m/s).
    The sampling freq is assumed to be 1 sec.
    
    Curvature = |x'y\" -y'x\"| /((x'**2 + y'**2)**(3/2))

    Parameters
    ----------
    x,y: numpy.array(dtype=float) shape (n_points, )

    Returns
    -------
    speed: numpy.array shape (n_points-1, )
    """
    x = np.asarray(coords['x'])
    y = np.asarray(coords['y']) 
    
    # Sample points of the trip at a fixed distance
    sampled_points = sampleTripPoints(trip_dist, curve_window_km)
    
    missingDataLocs = findMissingData(x, y)    
    
    listOfCurves = []
    for k in range(1, len(sampled_points)-1):   
        idx_bef = sampled_points[k-1]
        idx_mid = sampled_points[k]
        idx_aft = sampled_points[k+1]
        # first derivative
        dx = (x[idx_aft] - x[idx_bef])/2
        dy = (y[idx_aft] - y[idx_bef])/2
        # second derivative
        dxdx = (x[idx_aft]-2*x[idx_mid]+x[idx_bef])/2
        dydy = (y[idx_aft]-2*y[idx_mid]+y[idx_bef])/2
        # curvature (1/m)
        cur = np.abs(dx*dydy-dy*dxdx) / ((dx**2+dy**2)**(3/2))
        # Curvature threshold depends on speed
        cur_thr = 0.1#-0.031*spd[k] + 1.0
        if (cur>=cur_thr):
            listOfCurves.append(np.array([idx_bef,  idx_mid,  idx_aft,  cur]))
            
    listOfCurves_filt = filterCurves(listOfCurves, missingDataLocs)
            
    return listOfCurves_filt

#%%  
def is_outlier(points, thresh=3.5):
    """
    Returns a boolean array with True if points are outliers and False 
    otherwise.

    Parameters:
    -----------
        points : An numobservations by numdimensions array of observations
        thresh : The modified z-score to use as a threshold. Observations with
            a modified z-score (based on the median absolute deviation) greater
            than this value will be classified as outliers.

    Returns:
    --------
        mask : A numobservations-length boolean array.

    References:
    ----------
        Boris Iglewicz and David Hoaglin (1993), "Volume 16: How to Detect and
        Handle Outliers", The ASQC Basic References in Quality Control:
        Statistical Techniques, Edward F. Mykytka, Ph.D., Editor. 
    """
    if len(points.shape) == 1:
        points = points[:,None]
    median = np.median(points, axis=0)
    diff = np.sum((points - median)**2, axis=-1)
    diff = np.sqrt(diff)
    med_abs_deviation = np.median(diff)

    modified_z_score = 0.6745 * diff / med_abs_deviation

    return modified_z_score > thresh
#%% 
def basicFeatures(coords):
    """
    Returns a boolean array with True if points are outliers and False 
    otherwise.
    """
    x = np.asarray(coords['x'])
    y = np.asarray(coords['y'])
    
    duration = len(x) / float(60) # min
    speed = getSpeed(x, y) # km/h
    dist = getDistance(speed) # km
    dist_total= dist[-1]
    start_end_dist = np.linalg.norm(np.array([x[-1], y[-1]]))
    accel = np.diff(speed) # km/h/s
            
    return (duration, dist, dist_total, start_end_dist, speed, accel)

def curveFeatures(listOfCurves, speed,  accel):
    """
    Returns a boolean array with True if points are outliers and False 
    otherwise.
    """        
    curvature_bins = np.array([0, 0.35, 0.7, 1.0])  # 3 equally spaced bins for the degree of curvature. (lower values mean softer curves)
    features_len =18
    # Only calculate features when there are curves detected in the trip
    if (len(listOfCurves) > 0):
        curves_curvature = np.asarray(listOfCurves)[:, 3]   # curvature of each curve
        curves_bins = np.digitize(curves_curvature, curvature_bins)  # find in which bin falls each curve's curvature  
        
        # Speed
        curves_speed_bef = speed[np.asarray(listOfCurves)[:, 0].astype(int)] # speed in the beginning of the curve
        curves_speed = speed[np.asarray(listOfCurves)[:, 1].astype(int)] # speed in the cener of the curve 
        curves_speed_aft = speed[np.asarray(listOfCurves)[:, 2].astype(int)] # speed in the end of the curve 
        
        # Accel
        curves_accel_bef = accel[np.asarray(listOfCurves)[:, 0].astype(int)] # accel in the beginning of the curve
        curves_accel = accel[np.asarray(listOfCurves)[:, 1].astype(int)] # accel in the cener of the curve 
        curves_accel_aft = accel[np.asarray(listOfCurves)[:, 2].astype(int)] # accel in the end of the curve  
        
        features=[]
        # For each bin, calculate the features        
        for b in range(1, len(curvature_bins)):
            # Case there is at least one curve whose curvature falls into this bin
            if (any(curves_bins==b) == True):
                features.append(np.median(curves_speed_bef[curves_bins==b]))
                features.append(np.median(curves_speed[curves_bins==b]))
                features.append(np.median(curves_speed_aft[curves_bins==b]))
                features.append(np.median(curves_accel_bef[curves_bins==b]))
                features.append(np.median(curves_accel[curves_bins==b]))
                features.append(np.median(curves_accel_aft[curves_bins==b]))
            # Case there is no curve whose curvature falls into this bin
            else:
                features.append(0)
                features.append(0)
                features.append(0)
                features.append(0)
                features.append(0)
                features.append(0)
    # Case no curves in the trip
    else:
        features=features_len*[0]

    return (np.array(features))
   
#%%
def areSimilarTrips(duration1,  duration2,  dist1, dist2,  tol):
    """
    Find trips with similar duration (min) and total distance (km)
    """    
    # Difference in distance and duration should be less than the tolerance defined.
    if ((abs(dist1-dist2)<tol) & (abs(duration1-duration2)<tol)):
        return True
    else:
        return False
    
#%%
def chisquareModified(observed_freqs,  expected_freqs):
    """
    Sum of squared differences between observed and expected frequencies divided by the sum of expected frequencies
    The modification to the standard test consists in considering only non-empty categories.
    """    
    test = 0
    
    non_empty_categories = (observed_freqs>0) & (expected_freqs>0)
    numerator =  np.sum((observed_freqs[non_empty_categories] - expected_freqs[non_empty_categories])**2)
    denominator = np.sum(expected_freqs[non_empty_categories])
    if (denominator > 0):
        test = numerator/float(denominator)    
    return test
 
# MAIN LOOP
#plotTrip(1046,109)
#%%  Load trips into dataframes
# Set path of the directory containing the trips
path = './drivers'

curve_window_km = 25 / float(1000) # window frame to calculate curvature (km)
cur_thr = 0.1
curve_bins = np.linspace(0, 1, 10) # 1/m
#speed_bins = np.hstack((np.arange(0, 120, 5), np.arange(120, 220, 20))) # km/h
#accel_bins = np.hstack((np.arange(-40, -20, 10), np.arange(-20, 20, 2), np.arange(20, 40, 10))) # km/h/s
speed_bins = np.arange(0, 120, 2)
accel_bins = np.arange(-15, 15, 1) # km/h/s


# Initialize arrays
#curvesDF = pd.DataFrame(columns=['start', 'center', 'end',  'curvature'],  dtype=int)
#curvesDF[['start',  'center',  'end']] = curvesDF[['start',  'center',  'end']] .astype(int)
#curvesDF[['curvature']] = curvesDF[['curvature']].astype(float)

# Get the list of drivers
#drivers = ['1800', '1', '122', '1923', '298']
drivers = [f for f in listdir(path) if isdir(join(path,f)) and not f.startswith('.')]

# Loop through the drivers
max_drivers = 30
cnt_driver = 0

all_speeds = []
all_accels = []
features_driver = []
features_trip = []
labels_driver = []
labels_trip = []
spd_bin_acc = np.zeros([len(speed_bins)-1])
acc_bin_acc = np.zeros([len(accel_bins)-1])
for driver in drivers:
    print driver
    cnt_driver = cnt_driver + 1
    path_driver = join(path,driver)
    # Get the list of trips for this driver     
    trip_list = [f for f in listdir(join(path_driver)) if isfile(join(path_driver,f)) and not f.startswith('.')]
    driver_curves = []
    driver_curves_curvature =[]
    driver_curves_speed = []   
    # Loop through the trips 
    for trip in trip_list:
        path_trip = join(path_driver, trip)

        # Extract coordinates (x,y) of this trip
        coords = pd.read_csv(path_trip)
        
        # Basic features of the trip
        duration, dist, dist_total, start_end_dist, speed, accel = basicFeatures(coords)
        spd_bin_count = np.histogram(speed, speed_bins)[0] # km/h
        acc_bin_count = np.histogram(accel, accel_bins)[0] # km/h/s
        
#        spd_bin_acc = spd_bin_acc + spd_bin_count
#        acc_bin_acc = acc_bin_acc + acc_bin_count
        
        # Curve features
#        list_of_curves = getCurves(coords, dist, curve_window_km, cur_thr)
#        curve_features = curveFeatures(list_of_curves, speed[:-1],  accel)
#        if (len(list_of_curves)>0):
#            crv_bin_count = np.histogram(np.array(list_of_curves)[:,3], curve_bins)[0]
#        else:
#            crv_bin_count = np.zeros([len(curve_bins)])
#        
#        # Driver matching
#        features_driver.append(np.hstack((spd_bin_count, crv_bin_count)))
        labels_driver.append(str(driver))
#        
        # Trip matching
        features_trip.append(np.hstack((duration, dist_total, start_end_dist, spd_bin_count, acc_bin_count)))
        labels_trip.append(path_trip)
    
    if (cnt_driver == max_drivers):
        break


#%% Random Forest classifier
#split into a training and testing set
X_train, X_test, y_train, y_test = train_test_split(features_trip, labels_driver, test_size=0.33)
 
rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train, y_train)
#rf.predict(test)
scores = cross_validation.cross_val_score(rf, X_test, y_test, cv=5)
print "Scores (mean): ", np.mean(scores)
print "Scores (median): ", np.median(scores)

# Results
# 10 drivers -> 20% (speed_curve_Center), 26% (all curveFeatures, 100 estimators), 20% (all curveFeatures, 5 estimators), 28% (all curveFeatures, 200 estimators)
# 50 drivers -> 4.7% (speed_curve_Center), 9% (all curveFeatures, 100 estimators), 5% (all curveFeatures, 5 estimators), 9% (all curveFeatures, 200 estimators)
# 100 drivers -> 2.1% (speed_curve_Center), 5% (all curveFeatures, 100 estimators), 3% (all curveFeatures, 5 estimators), 5% (all curveFeatures, 200 estimators)

#%% Scatter plot (duration and dist and num curves)

#fig = plt.figure()
#ax = Axes3D(fig)
#ax.scatter(features_trip[:,0], features_trip[:,1],  features_trip[:,2])
#plt.show()
#plt.xlabel('duration (min)', fontsize=14)
#plt.ylabel('dist (km)', fontsize=14)
#fig.suptitle('Scatter plot (duration and distance)', fontsize=16)

#%%
#features_trip = np.array(features_trip)
#speed_start = 3
#speed_indices = np.arange(speed_start, speed_start+len(speed_bins)-1)
#accel_start = speed_indices[-1] + 1
#accel_indices = np.arange(accel_start, accel_start+len(accel_bins)-1)

#speed_freqs = features_trip[:,speed_indices].sum(axis=0)
#widths = np.diff(speed_bins)
#fig = plt.figure()
#plt.bar(speed_bins[:-1], spd_bin_acc/sum(spd_bin_acc), widths)
#plt.show()
#plt.xlabel('Speed (km/h)', fontsize=16)
#plt.ylabel('Count', fontsize=16)
#
#
##accel_freqs = features_trip[:,accel_indices].sum(axis=0)
#widths = np.diff(accel_bins)
#fig = plt.figure()
#plt.bar(accel_bins[:-1], acc_bin_acc/sum(acc_bin_acc), widths)
#plt.show()
#plt.xlabel('Acceleration (km/h/s)', fontsize=16)
#plt.ylabel('Count', fontsize=16)

#%%

#for i in range(0, len(labels_trip)):
#    d = ((features_trip[i]-features_trip[:])**2).sum(axis=1)  # compute distances
#    ndx = d.argsort() # indirect sort
#    closest_trip = ndx[1]
#    if ((d[closest_trip] < 1)): #& (sum(point[3:]) > 0)):
#        print "features: ", features_trip[i], "\n", features_trip[closest_trip]
#        print "labels: ", labels_trip[i], " , " ,labels_trip[closest_trip]
#        
#        #        chi_test, chi_p = chisquare(features_trip[i,3:], features_trip[closest_trip,3:])
#        #        print "chi_test: ", chi_test
#        #        if (chi_test < 20):
#        coords1 = pd.read_csv(labels_trip[i])
#        coords2 = pd.read_csv(labels_trip[closest_trip])
#  
#%%
coords1 = pd.read_csv('./drivers/1045/10.csv')
coords2 = pd.read_csv('./drivers/1051/74.csv')

duration, dist, dist_total, start_end_dist, speed1, accel1 = basicFeatures(coords1)
duration, dist, dist_total, start_end_dist, speed2, accel2 = basicFeatures(coords2)

fig = plt.figure()
plt.plot(coords1['x'], coords1['y'], 'r')
plt.plot(coords1['x'][0], coords1['y'][0], 'go')
plt.plot(coords1['x'][len(coords1['x'])-1], coords1['y'][len(coords1['x'])-1], 'go')

plt.plot(coords2['x'], coords2['y'], 'b')
plt.plot(coords2['x'][0], coords2['y'][0], 'go')
plt.plot(coords2['x'][len(coords2['x'])-1], coords2['y'][len(coords2['x'])-1], 'go')
plt.show()


fig = plt.figure()
plt.hist(speed1, speed_bins, color="#6495ED", alpha=.5)
plt.hist(speed2, speed_bins, color="#F08080", alpha=.5)
fig.suptitle('Histogram of speeds', fontsize=14)
plt.show()
plt.xlabel('km/h', fontsize=14)
plt.ylabel('Count', fontsize=14)

fig = plt.figure()
plt.hist(accel1, accel_bins, color="#6495ED", alpha=.5)
plt.hist(accel2, accel_bins, color="#F08080", alpha=.5)
fig.suptitle('Histogram of accelerations', fontsize=14)
plt.show()
plt.xlabel('km/h/s', fontsize=14)
plt.ylabel('Count', fontsize=14)
#%%
num_drivers = np.array([10, 50, 100, 1000])
main_loop_time = np.array([4, 67, 110, 1100]) # secs
trip_matching_time = np.array([0.73, 11, 47, 6600]) # secs
total_time = main_loop_time + trip_matching_time
fig = plt.figure()
plt.plot(num_drivers, main_loop_time/60.0, color="#6495ED", alpha=.5)
plt.plot(num_drivers, trip_matching_time/60.0, color="#F08080", alpha=.5)
plt.plot(num_drivers, total_time/60.0, color="#008F80", alpha=.5)
fig.suptitle('Execution time', fontsize=14)
plt.xlabel('number of drivers', fontsize=14)
plt.ylabel('minutes', fontsize=14)
plt.legend(['Load data + Compute features', 'Trip matching', 'Total'])
plt.show()


  #%% Scatter plot of curvature vs. speed
    # Keep only the "good" points
    # "~" operates as a logical not operator on boolean numpy arrays
#    curvatureC = np.asarray(driver_curves_curvature)
#    speedC = np.asarray(driver_curves_speed)
#    curvatureC_clean = curvatureC[~is_outlier(curvatureC)]
#    speedC_clean = speedC[~is_outlier(speedC)]    
    
#    #%%
#    fig = plt.figure()
#    plt.hist(all_curvatures)
#    plt.show()
#    
#    fig = plt.figure()
#    plt.hist(curvatureC_clean)
#    plt.show()
#    #%%
#    fig = plt.figure()
#    plt.hist(speedC)
#    plt.show()
#    
#    fig = plt.figure()
#    plt.hist(speedC_clean)
#    plt.show()
    #%%
#    curvature_bins = np.arange(0.1,1.1, 0.1)
#    curv_median = np.zeros([len(curvature_bins)-1])
#    for c in range(0, len(curvature_bins)-1):
#        curv_median[c] = np.median(speedC[(curvatureC >= curvature_bins[c]) & (curvatureC <= curvature_bins[c+1])])
#    #X_train, X_test, y_train, y_test = train_test_split(curv_median, (len(curvature_bins)-1)*drivers, test_size=0.33, random_state=42)
#    features.append(curv_median)

    #%%
    #plt.plot(np.arange(0.15,1.05, 0.1), curv_median)
    
#features = np.array(features)
#plt.show()
#fig.suptitle('Median speed for different curvatures', fontsize=16)
#plt.xlabel('Curvature (1/m)', fontsize=16)
#plt.ylabel('Speed (km/h)', fontsize=16)
#plt.legend(drivers)
#    plt.xlabel('curvature (1/m)', fontsize=14)
#    plt.ylabel('speed (km/h)', fontsize=14)
#    fig.suptitle('Curvature vs. Speed (outliers removed)', fontsize=16)

#%% Random Forest classifier
#rf = RandomForestClassifier(n_estimators=100)
#rf.fit(train, target)
#rf.predict(test)
#scores = cross_validation.cross_val_score(rf, iris.data, iris.target, cv=5)



    #%%
#    plt.scatter(curvatureC, speedC)
#    H, xedges, yedges = np.histogram2d(curvatureC, speedC, bins=20)
#    fig = plt.figure(figsize=(7, 7))
#    ax = fig.add_subplot(111)
#    ax.set_title('Driver ' + str(driver))
#    im = mpl.image.NonUniformImage(ax, interpolation='bilinear')
#    xcenters = xedges[:-1] + 0.5 * (xedges[1:] - xedges[:-1])
#    ycenters = yedges[:-1] + 0.5 * (yedges[1:] - yedges[:-1])
#    im.set_data(xcenters, ycenters, H)
#    ax.images.append(im)
#    ax.set_xlim(xedges[0], xedges[-1])
#    ax.set_ylim(yedges[0], yedges[-1])
#    #ax.set_aspect('equal')
#    plt.show()    

##%% Plot clusters of trips (based on duration)
#max_trips_per_plot = 5
#trip_durations = np.asarray(trip_durations)
#feature = dict()
#feature['data'] = trip_durations
#feature['name'] = 'duration'
#feature['units'] = 'minutes'
## Get the intervals that best separate the histogram of trip durations
#intervals = getClusters1D(feature)
## Plot trips contained in the same interval, one figure per interval
#plotSimilarTrips(feature, intervals, normal_trips, max_trips_per_plot, color = True)
#
##%% Plot clusters of trips (based on max speed)
#max_trips_per_plot = 5
#trip_spd_max = np.asarray(trip_spd_max)
#feature = dict()
#feature['data'] = trip_spd_max
#feature['name'] = 'maxSpd'
#feature['units'] = 'km/h'
## Get the intervals that best separate the histogram of trip durations
#intervals = getClusters1D(feature)
## Plot trips contained in the same interval, one figure per interval
#plotSimilarTrips(feature, intervals, normal_trips, max_trips_per_plot, color = True)
#
##%% Plot clusters of trips (based on traveled distance)
#max_trips_per_plot = 5
#trip_spd_max = np.asarray(trip_distances)
#feature = dict()
#feature['data'] = trip_distances
#feature['name'] = 'distance'
#feature['units'] = 'km'
## Get the intervals that best separate the histogram of trip durations
#intervals = getClusters1D(feature)
## Plot trips contained in the same interval, one figure per interval
#plotSimilarTrips(feature, intervals, normal_trips, max_trips_per_plot, color = False)

##%% Save list of erroneous trips to a file
#f = open('trips_err.txt', 'w')
#for item in trip_err:
#  f.write("%s\n" % item)
#f.close()
#
##%% Percentage of erroneous trips
#err_pcg = float(len(trip_err))/(len(trip_err)+len(durations)) # approx 10 %

#plt.close("all")

#%% Histogram of trip durations 
#fig = plt.figure()
#nbins = 200
#plt.hist(trip_durations, nbins, histtype="stepfilled")
#fig.suptitle('Histogram of trip durations (' + str(nbins) + ' bins)', fontsize=18)
#plt.xlabel('minutes', fontsize=18)
#fig.savefig('tripDurations_'+ str(nbins) +'bins.jpg')
#plt.show()
#
##%% Histogram of trip distances 
#fig = plt.figure()
#nbins = 5000
#plt.hist(trip_distances, nbins, histtype="stepfilled")
#fig.suptitle('Histogram of trip distances (' + str(nbins) + ' bins)', fontsize=18)
#plt.xlabel('km', fontsize=18)
#fig.savefig('tripDistances_'+ str(nbins) +'bins.jpg')
#plt.show()
#       
##%% Histogram of max speed
#fig = plt.figure()
#nbins = 200
#plt.hist(trip_spd_max, nbins, histtype="stepfilled")
#fig.suptitle('Histogram of max speed (' + str(nbins) + ' bins)', fontsize=18)
#plt.xlabel('km/h', fontsize=18)
#fig.savefig('tripMaxSpd_'+ str(nbins) +'bins.jpg')
#plt.show()
#       
##%% Histogram of avg speed
#fig = plt.figure()
#nbins = 2000
#plt.hist(trip_spd_avg, nbins, histtype="stepfilled")
#fig.suptitle('Histogram of avg speed (' + str(nbins) + ' bins)', fontsize=18)
#plt.xlabel('km/h', fontsize=18)
#fig.savefig('tripAvgSpd_'+ str(nbins) +'bins.jpg')
#plt.show() 
#       
##%% Histogram of std speed
#fig = plt.figure()
#nbins = 200
#plt.hist(trip_spd_std, nbins, histtype="stepfilled")
#fig.suptitle('Histogram of std speed (' + str(nbins) + ' bins)', fontsize=18)
#plt.xlabel('km/h', fontsize=18)
#fig.savefig('tripStdSpd_'+ str(nbins) +'bins.jpg')
#plt.show() 
#
##%% Speed
#spd = getSpeed(x, y)
#dist = getDistance(spd)
#plt.figure()
#plt.plot(dist)
#plt.show()



#%% Smoothing (optional)
#t = np.arange(x.shape[0])
#std = 0.1 * np.ones_like(x)
#
#x = np.asarray(x)
#y = np.asarray(y)
#fx = UnivariateSpline(t, x, k=4, w=1 / np.sqrt(std))
#fy = UnivariateSpline(t, y, k=4, w=1 / np.sqrt(std))
#plt.figure()
#plt.plot(fx(t),fy(t))
#plt.show()

#%% Plot Speed
#plt.figure()
#plt.plot(spd)
        


#fig = plt.figure()
#plt.plot(x,y)
#plt.plot(x[curves],y[curves],'ro')
#for i in range(0, len(curves)):
#    plt.text(x[curves[i]], y[curves[i]], str('%.2f' % curvature[i]))
#plt.show()
#plt.xlabel('x', fontsize=16)
#plt.ylabel('y', fontsize=16)
#fig.suptitle('Trip coordinates', fontsize=16)

#%% Features related to curvature

#fig = plt.figure()
#plt.plot(cur_acc, 'b', label='window = ' + str(delta_m*2) + 'meters')
#plt.show()
#plt.xlabel('point index', fontsize=14)
#plt.ylabel('1/m', fontsize=14)
#plt.legend(loc='upper left', numpoints = 1, fontsize=10)
#fig.suptitle('Curvature', fontsize=16)

#%% Find loction of curves
#curves_idx = [item for sublist in curves_idx for item in sublist]
#curves_set = set(curves_idx)
#curvv_x = x.ix[curves_set]
#curvv_y = y.ix[curves_set]

#%% Plot trip
#fig = plt.figure()
#plt.plot(x,y)
##plt.plot(x,y, 'bo')
#plt.plot(x[list_points_curv],y[list_points_curv],'go')
#for i in range(1,13):
#    plt.text(x[list_points_curv[delta_m*i]], y[list_points_curv[delta_m*i]], str(delta_m*i))
#plt.plot(x[curves_idx],y[curves_idx],'ro')
#plt.show()
#plt.xlabel('x', fontsize=16)
#plt.ylabel('y', fontsize=16)
#fig.suptitle('Trip coordinates', fontsize=16)

#%% Spline interpolation
#from scipy import interpolate
#xnew = x[list_points_curv[50:60]]
#ynew = y[list_points_curv[50:60]]
#tck = interpolate.splrep(xnew, ynew)
#yinterp = interpolate.splev(xnew, tck, der=0)
#
#plt.figure()
#plt.plot(xnew, ynew, 'go', xnew, ynew, xnew, 'r', np.sin(xnew), x, y, 'b')
#plt.legend(['5m points', 'Cubic Spline', 'Original'])
#plt.title('Cubic-spline interpolation')
#plt.show()
#    
#%% Plot traject with colours associated to speed
#plt.figure()
#for i in range(0, len(x)-1):
#    plt.plot(x[i:i+2],y[i:i+2], color = m.to_rgba(spd[i]))
#
## Plot trip with curves
#plt.plot(x, y, 'go')
#plt.plot(x[curves_idx], y[curves_idx], 'ro')
##idx = np.arange(0,len(x)+1, 50)
##plt.plot(x[idx], y[idx], 'go') 
#plt.show()
      
#%%  Correlation of curvature with speed
# For curvatures above certain threshold (e.g. 0.1 m^-1), calculate:
#   - mean speed inside the curve
#   - deceleration before the curve
#   - acceleration after the curve

#plt.figure()
#plt.scatter(cur[1::], spd[1::])
#plt.show()

#plt.close("all")
