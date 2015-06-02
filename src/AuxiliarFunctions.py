# -*- coding: utf-8 -*-
"""
Created on Wed Apr 01 09:44:59 2015

@author: F59JBC0
"""

#%% Import libraries
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm
from scipy.stats import kde
import random
import math
    
#%%
def getSpeed(coords):
    """Calculate the speed of a trip given its coordinates (x, y) meters.
    The sampling freq is assumed to be 1 sec.

    Parameters
    ----------
    x,y: numpy.array(dtype=float) shape (n_points, )

    Returns
    -------
    speed: numpy.array shape (n_points-1, )
    """
    x = np.array(coords)[:, 0]
    y = np.array(coords)[:, 1]
    
    dx = np.diff(x) # Differences in 'x'.
    dy = np.diff(y) # Differences in 'y'.  
    
    spd = ((dx**2) + (dy**2))**0.5 # Speed is the euclidean distance of differences.
    
    # Add speed with value '0' in the first position of the return array
    return np.hstack((0, 3.6*spd)) #km/h
    
#%%
def getDistance(coords):
    """Calculate the traveled distance of a trip given its speed (m/s).
    The sampling freq is assumed to be 1 sec.

    Parameters
    ----------
    x,y: numpy.array(dtype=float) shape (n_points, )

    Returns
    -------
    speed: numpy.array shape (n_points-1, )
    """
    
    dist = np.cumsum(getSpeed(coords)/float(3600))
    
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
    x = coords[:, 0]
    y = coords[:, 1]  
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
def centralCurvature(XY):
    """ X,Y : np.array (dim: 3 rows by 2 column)
    """
    x = np.array(XY)[:, 0]
    y = np.array(XY)[:, 1]
    
    # first derivative
    dx = (x[2] - x[0])/2.0
    dy = (y[2] - y[0])/2.0
    # second derivative
    dxdx = (x[2] - 2*x[1] + x[0])/2.0
    dydy = (y[2] - 2*y[1] + y[0])/2.0
    # curvature (1/m)
    cur = (dx*dydy-dy*dxdx) / ((dx**2+dy**2)**(3/2))
    
    return cur

#%%
def getTripCurvature(coords, indices, dist_thr = 0.025):
    """ coords : np.array (dim: N rows by 2 column)
        rdp_indices: list (dim: N rows by 1 column)
        dist_thr: distance threshold (in meters) that define the size of the curve
    """
    dist = getDistance(coords)
    XY = np.zeros([3, 2])
    curvature = []
    #print "indices:", indices
    for idx in indices:
        dist_bef = dist[idx] - dist_thr
        dist_aft = dist[idx] + dist_thr
        #print "idx, dist_bef, aft: ", idx, dist_bef, dist_aft
        if ((dist_bef >= dist[0]) & (dist_aft <= dist[-1])):  # control overflow & underflow
            XY[0] = coords[dist <= dist_bef][-1]            
            XY[1] = coords[idx]    
            XY[2] = coords[dist >= dist_aft][0]
            curvature.append(centralCurvature(XY))
        else:
            curvature.append(0.0)
    
    return np.array(curvature)
    
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
    x = coords[:, 0]
    y = coords[:, 1]
    
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
    x = coords[:, 0]
    y = coords[:, 1]
    
    duration = len(coords) / float(60) # min
    speed = getSpeed(coords) # km/h
    accel = np.diff(speed) # km/h/s
    dist = getDistance(coords) # km
    dist_total= dist[-1]
    start_end_dist = np.sqrt((x[-1]-x[0])**2 + (y[-1]-y[0])**2)   
            
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

#%%
def diffDist(X):
    dist = []
    for i in range(1, len(X)):
        dist.append(np.linalg.norm(X[i] - X[i-1]))
        
    return np.array(dist)
    
#%%
def diffAngle(X):
    angles = [0]
    for i in range(1, len(X)-1):
        b = np.linalg.norm(X[i] - X[i-1])
        a = np.linalg.norm(X[i+1] - X[i])
        c = np.linalg.norm(X[i+1] - X[i-1])
        theta = math.acos((a**2 + b**2 - c**2) / (2 * a * b))
        angles.append(theta)
    
    angles.append(0)
    return angles
