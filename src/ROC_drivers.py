# -*- coding: utf-8 -*-
"""
Created on Mon May 18 17:15:47 2015

@author: jordi
"""
import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from scipy.spatial import distance
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import label_binarize
from sklearn.cross_validation import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn import svm
from sklearn.metrics import roc_curve, auc

#global vars
PATH = './drivers'

URBAN_TRIP = 0
HIGHWAY_TRIP = 1
MAX_SPEED = 220 # km/h
MIN_SPEED = -10.0

MAX_REAL_SPEED = 220.0
MAX_URBAN_SPEED = 55.0 # km/h
WIDTH_WINDOW = 60 # = 1 minut

RANDOM_FOREST = 0
LOG_REG = 1
CLASSIFICATION_TYPE = RANDOM_FOREST

CURVE_WINDOW_KM = 25 / float(1000) # window frame to calculate curvature (km)
CUR_THR = 0.1

def main():

    list_trip_matching = read_number_repeat_trips('tripMatching_dr0_dr99.txt')
    drivers = getAllDriversFolders()
    #train
    trips_data = []
    y = []
    drivers = drivers[:10]
    n_drivers = len(drivers)
    count_driver = 0
    count_trip = 0
    print 'read files and create data'
    for d in drivers:
        current_driver = int(d.split('/')[2])
        trips_file = getAllTripsFromDriver(d)
        for t in trips_file:
            #read file
            trip=pd.read_csv(d+'/'+t)
            #put trip features in data list
            trips_data.append(calculate_features_for_trip(trip,list_trip_matching[count_driver][count_trip]))
            y.append(int(current_driver))
            count_trip += 1
        count_driver += 1
        count_trip = 0
    #list to numpy array
    trips_data = np.array(trips_data)
    y = np.array(y)
    #normalize
    X = (trips_data - np.mean(trips_data)) / (np.max(trips_data) - np.min(trips_data))
    
    y = label_binarize(y,classes=np.unique(y))
    n_classes = y.shape[1]
    # shuffle and split training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=.1,random_state=0)
    print 'size train',X_train.shape
    print 'size test',X_test.shape
    
    random_state = np.random.RandomState(0)
    # Learn to predict each class against the other
    #classifier = OneVsRestClassifier(svm.SVC(kernel='linear', probability=True,random_state=random_state))
    #y_score = classifier.fit(X_train, y_train).decision_function(X_test)
    
    classifier = OneVsRestClassifier(RandomForestClassifier(random_state=random_state,n_estimators=150,n_jobs=-1, max_depth=10), multilabel_=False)

    print 'start train'
    print y_test
    y_score = classifier.fit(X_train, y_train).predict_proba(X_test)
    
    #classifier = OneVsRestClassifier(KNeighborsClassifier(n_neighbors=11) )
    #print 'start train'
    #y_score = classifier.fit(X_train, y_train).predict_proba(X_test)   
    #classifier = OneVsRestClassifier(KNeighborsClassifier(n_neighbors=11))    
    #y_score = classifier.fit(X_train, y_train).predict_proba(X_test)
    #classifier = OneVsRestClassifier(LogisticRegression(penalty='l2'))    
    #y_score = classifier.fit(X_train, y_train).predict_proba(X_test)
    #Second test
    print 'end train and create roc curve'  
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    """
    # Plot of a ROC curve for a specific class
    plt.figure()
    plt.plot(fpr[2], tpr[2], label='ROC curve (area = %0.2f)' % roc_auc[2])
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic for')
    plt.legend(loc="lower right")
    plt.show()
    """
    # Plot ROC curve
    fig = plt.figure()
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]))
    #for i in range(n_classes):
    #   plt.plot(fpr[i], tpr[i], label='ROC curve of class {0} (area = {1:0.2f})'''.format(i, roc_auc[i]))
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Some extension of Receiver operating characteristic to multi-class ('+str(n_drivers)+" drivers) Random Forest")
    #plt.legend(loc="lower right")
    plt.legend(bbox_to_anchor=(0.1,0.1), loc=2, borderaxespad=0.)
    plt.show()
    
    fig.savefig('figure_ROC_'+str(n_drivers)+'drivers_rf2.png', dpi=100)
    
    print 'end execution'
    
####################### 

def read_number_repeat_trips(filename):
    trip_matching_values = []
    with open(filename, 'r') as f:
        file_trip_matching = [line.strip() for line in f]
    for dr in file_trip_matching:
        trip_matching_values.append(dr.split(' '))
    return trip_matching_values  
def calculate_features_for_trip(trip,repet_number):
     
    x = np.asarray(trip['x'])
    y = np.asarray(trip['y'])
    
    total_time = (len(x)/60.0)
    speed = (getSpeed(x,y))
    dist = getDistance(speed)
    total_distance =dist[-1]
    #Convert speed from m/s to km/h 
    speed *= 3.6
    #Common features
    #speed
    total_max_speed = max(speed)
    total_mean_speed = np.mean(speed)
    total_median_speed = np.median(speed)
    total_std_speed = np.std(speed)
    #acc
    acc = np.diff(speed)
    total_acc_max = max(acc) #màxim
    total_acc_min = min(acc) #mínim
    total_acc_mean= np.mean(acc) #mitjana
    total_acc_negavg = np.mean(acc[acc < 0.0]) #mitjana dels valors negatius
    total_acc_posavg = np.mean(acc[acc > 0.0]) #mitjana dels valors positius
    total_acc_median = np.median(acc) #mediana
    total_acc_negmed = np.median(acc[acc < 0.0]) #mediana dels valors negatius
    total_acc_posmed = np.median(acc[acc > 0.0]) #mediana dels valors positius
    total_acc_std = np.std(acc) #desviació estàndard
    total_acc_negstd = np.std(acc[acc < 0.0]) #desviació estàndard dels valors negatius
    total_acc_posstd = np.std(acc[acc > 0.0]) #desviació estàndard dels valors positius    

    #euclidan dist of first point and last point of trip
    first_point = [x[0],y[0]]
    last_point = [x[-1],y[-1]]
    
    dist_start_end = distance.euclidean(first_point,last_point)

    MIN_SUBTRIP_SIZE = len(x) * 0.30
    
    #Init vars for urban trip
    trip_type = URBAN_TRIP
    #time and distance 
    time_highway = 0
    distance_highway = 0
    number_splits = 0    
    #speed
    max_speed_highway = 0
    min_speed_highway = 0
    mean_speed_highway = 0
    speed_std_highway = 0
    speed_median_highway = 0
    #acc
    max_acc_highway = 0
    min_acc_highway = 0
    mean_acc_highway = 0
    acc_std_highway = 0
    acc_median_highway = 0
    acc_negavg_highway = 0 
    acc_posavg_highway = 0
    acc_negmed_highway = 0
    acc_posmed_highway = 0
    acc_negstd_highway = 0
    acc_posstd_highway = 0 
    #curve
    highway_curve_feature_1 = 0
    highway_curve_feature_2 = 0
    highway_curve_feature_3 = 0
    #############################

    MIN_POINTS_NUMBER_OVER_URBAN_SPEED = len(x)*0.10
    
    #create sliding window
    speed_window = rolling_window(speed,WIDTH_WINDOW)
    #calculate means for all window
    speed_means_window = np.mean(speed_window,axis=1)
    #Check if it is a hightway trip
    if(len(speed[speed > MAX_URBAN_SPEED]) > MIN_POINTS_NUMBER_OVER_URBAN_SPEED):
        trip_type = HIGHWAY_TRIP
        #Highway
        #Get index of elements that the mean > MAX_URBAN_SPEED
        indx = np.where(speed_means_window > MAX_URBAN_SPEED)
        if(len(indx[0]) > 0):
            MIN_SUBTRIP_SIZE = len(x) * 0.55          
            #Get las index
            last_point = indx[0][-1]
            #Afageixo els index per tenir els punts del tram (+ width window)
            indx = np.append(indx,[e+last_point for e in range(1,WIDTH_WINDOW)])
            split = np.where(np.diff(indx) > 1)[0]
            number_splits = len(split)            
            highways_points = []
            left_slice = 0
            #first point of trip
            last_point = 0           
            current_subtrip = []
            if(len(split) > 0):
                for sp in split:
                    right_slice = sp + 1
                    current_subtrip = indx[left_slice:right_slice]
                    diff = (current_subtrip[0] - last_point)
                    #first point (empty list)
                    if(diff < MIN_SUBTRIP_SIZE and len(highways_points) == 0):
                        if(diff == 0):
                            highways_points.append(current_subtrip)
                        else:
                            #generate points to join two subtrips
                            tmp = np.asarray([p for p in range(last_point,current_subtrip[0])])
                            highways_points.append(np.concatenate([tmp,current_subtrip]))
                    #points during the trip       
                    elif(diff < MIN_SUBTRIP_SIZE and len(highways_points) > 0):
                        #generate points to join two subtrips
                        gen_points = np.asarray([p for p in range(last_point + 1,current_subtrip[0])])
                        highways_points[-1] = np.concatenate([highways_points[-1],gen_points,current_subtrip])
                    #split subtrip   
                    else:
                        highways_points.append(current_subtrip)    
                    last_point = current_subtrip[-1]
                    left_slice = (right_slice)   
                #The last subtrip
                final_subtrip = indx[(split[-1] + 1):]
                diff = (final_subtrip[0] - current_subtrip[-1])
                if(diff <= MIN_SUBTRIP_SIZE):
                    #generate points to join two subtrips
                    gen_points = np.asarray([p for p in range(current_subtrip[-1] + 1,final_subtrip[0])])
                    highways_points[-1] = np.concatenate([highways_points[-1],gen_points,final_subtrip])   
                else:
                    highways_points.append(final_subtrip)
            else:
                highways_points.append(indx)
            #Filtrate list                
            #TODO best way to filter subtrips < MIN_SUBTRIP_SIZE delete
            all_highways_points = []
            for h in highways_points:
                if(len(h) >= MIN_POINTS_NUMBER_OVER_URBAN_SPEED):
                    all_highways_points.append(h)
            if(len(all_highways_points) > 0):
                #init vars:
                list_time_highway = []
                list_distance_highway = []
                list_max_speed_highway = []
                list_min_speed_highway = []
                list_mean_speed_highway = []
                list_speed_std_highway = []
                list_speed_median_highway = []
                list_max_acc_highway = []
                list_min_acc_highway = []
                list_mean_acc_highway = []
                list_acc_std_highway = []
                list_acc_median_highway = []
                list_acc_negavg_highway = []
                list_acc_posavg_highway = []
                list_acc_negmed_highway =[]
                list_acc_posmed_highway =[]
                list_acc_negstd_highway =[]
                list_acc_posstd_highway = []
                list_curbature_highway_feature_1 = []
                list_curbature_highway_feature_2 = []
                list_curbature_highway_feature_3 = []
                
                for subtrip_highway in all_highways_points:
                    #Calculate features
                    highway_speed = []
                    highway_speed = getSpeed(x[subtrip_highway],y[subtrip_highway])
                
                    #time and distance
                    list_time_highway.append(len(subtrip_highway)/60.0)
                    #Last element of return array
                    dist_highway = getDistance(highway_speed)
                    list_distance_highway.append(dist_highway[-1])
                    
                    #Convert to km/h
                    highway_speed *= 3.6
                    list_max_speed_highway.append(np.max(highway_speed))
                    list_min_speed_highway.append(np.min(highway_speed))
                    list_mean_speed_highway.append(np.mean(highway_speed))
                    list_speed_std_highway.append(np.std(highway_speed))
                    list_speed_median_highway.append(np.median(highway_speed))
                    
                    #acc
                    acc_highway = np.diff(highway_speed)
                    list_max_acc_highway.append(np.max(acc_highway))
                    list_min_acc_highway.append(np.min(acc_highway))
                    list_mean_acc_highway.append(np.mean(acc_highway))
                    list_acc_std_highway.append(np.std(acc_highway))
                    list_acc_median_highway.append(np.median(acc_highway))
                    list_acc_negavg_highway.append(np.mean(acc_highway[acc_highway < 0.0])) #mitjana dels valors negatius
                    list_acc_posavg_highway.append(np.mean(acc_highway[acc_highway > 0.0])) #mitjana dels valors positius
                    list_acc_negmed_highway.append(np.median(acc_highway[acc_highway < 0.0])) #mediana dels valors negatius
                    list_acc_posmed_highway.append(np.median(acc_highway[acc_highway > 0.0])) #mediana dels valors positius
                    list_acc_negstd_highway.append(np.std(acc_highway[acc_highway < 0.0])) #desviació estàndard dels valors negatius
                    list_acc_posstd_highway.append(np.std(acc_highway[acc_highway > 0.0])) #desviació estàndard dels valors positius 
                
                    #curvature
                    list_of_highway_curves = getCurves(x[subtrip_highway],y[subtrip_highway],dist_highway, CURVE_WINDOW_KM, CUR_THR)    
                    total_highway_curve_features = curveFeatures(list_of_highway_curves,highway_speed)
                    list_curbature_highway_feature_1.append(total_highway_curve_features[0])
                    list_curbature_highway_feature_2.append(total_highway_curve_features[1])
                    list_curbature_highway_feature_3.append(total_highway_curve_features[2])
                #Calculate means                 
                #time and distance
                time_highway = np.sum(list_time_highway)
                distance_highway = np.sum(list_distance_highway)
                
                #Convert to km/h
                highway_speed *= 3.6
                max_speed_highway = np.max(list_max_speed_highway)
                min_speed_highway = np.min(list_min_speed_highway)
                mean_speed_highway = np.mean(list_mean_speed_highway)
                speed_std_highway = np.mean(list_speed_std_highway)
                speed_median_highway = np.mean(list_speed_median_highway)
                
                #acc
                max_acc_highway = np.max(list_max_acc_highway)
                min_acc_highway = np.min(list_min_acc_highway)
                mean_acc_highway = np.mean(list_mean_acc_highway)
                acc_std_highway = np.mean(list_acc_std_highway)
                acc_median_highway = np.mean(list_acc_median_highway)
                acc_negavg_highway = np.mean(list_acc_negavg_highway) 
                acc_posavg_highway = np.mean(list_acc_posavg_highway) 
                acc_negmed_highway = np.mean(list_acc_negmed_highway) 
                acc_posmed_highway = np.mean(list_acc_posmed_highway) 
                acc_negstd_highway = np.mean(list_acc_negstd_highway) 
                acc_posstd_highway = np.mean(list_acc_posstd_highway)
                
                #curve
                highway_curve_feature_1 = np.mean(list_curbature_highway_feature_1)
                highway_curve_feature_2 = np.mean(list_curbature_highway_feature_2)
                highway_curve_feature_3 = np.mean(list_curbature_highway_feature_3)
    features = []
    #Curve features
    listOfCurves = getCurves(x,y,dist, CURVE_WINDOW_KM, CUR_THR)    
    total_curve_features = curveFeatures(listOfCurves,speed)
    
    #common features
    #Curve features
    features.append(total_curve_features[0])
    features.append(total_curve_features[1])
    features.append(total_curve_features[2])
    features.append(trip_type)
    #speed
    features.append(total_max_speed)
    features.append(total_mean_speed)
    features.append(total_median_speed)
    features.append(total_std_speed)
    #acc
    features.append(total_acc_max)
    features.append(total_acc_min)
    features.append(total_acc_mean)
    features.append(total_acc_negavg)
    features.append(total_acc_posavg)
    features.append(total_acc_median)
    features.append(total_acc_negmed)
    features.append(total_acc_posmed)
    features.append(total_acc_std)
    features.append(total_acc_negstd)
    features.append(total_acc_posstd)
    #times and distances
    features.append(total_distance)
    features.append(total_time)
    features.append(dist_start_end)
    #highway features
    #time and distance
    features.append(time_highway)
    features.append(distance_highway)
    features.append(number_splits)
    #speed
    features.append(max_speed_highway)
    features.append(min_speed_highway)
    features.append(mean_speed_highway)
    features.append(speed_std_highway)
    features.append(speed_median_highway)
    #acc
    features.append(max_acc_highway)
    features.append(min_acc_highway)
    features.append(mean_acc_highway)
    features.append(acc_std_highway)
    features.append(acc_median_highway)
    features.append(acc_negavg_highway)
    features.append(acc_posavg_highway)
    features.append(acc_negmed_highway)
    features.append(acc_posmed_highway)
    features.append(acc_negstd_highway)
    features.append(acc_posstd_highway)
    #curve
    features.append(highway_curve_feature_1)
    features.append(highway_curve_feature_2)
    features.append(highway_curve_feature_3)

    #trip matching
    #features.append(int(repet_number))
      
    return np.append([],features)
    
def getAllDriversFolders():
    return [x[0] for x in os.walk(PATH)][1:]
    

def getAllTripsFromDriver(path):
    return os.listdir(path)   

def getSpeed(x,y):
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
    
    return spd

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
    
    dist = np.cumsum(spd)
    
    return dist

def rolling_window(a, window):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a,shape=shape,strides=strides)
    
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
            list_points_curv.append(point_after)
            i = point_after
        else:
            end = True
    
    return list_points_curv
   
def findMissingData(x, y):
    """
    Returns a boolean array with True if points are outliers and False 
    otherwise.
    """

    speed = 3.6 * getSpeed(x, y) # km/h
    accel = np.diff(speed) # km/h
    
    # Case either speed or acceleration is out of range (abnormal trip)                   
    return np.where((abs(speed[:-1]) > 300) & (abs(accel) > 50))
    
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
    
def getCurves(x,y, trip_dist, curve_window_km, cur_thr):
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
    #x = np.asarray(coords['x'])
    #y = np.asarray(coords['y']) 
    
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
    
def curveFeatures(listOfCurves, speed):
    """
    Returns a boolean array with True if points are outliers and False 
    otherwise.
    """
    curvature_bins = np.array([0, 0.35, 0.7, 1.0])
    curves_speed_bins = np.zeros([len(curvature_bins)-1])
    
    if (len(listOfCurves) > 0):    
        curves_curvature = np.asarray(listOfCurves)[:, 3]
        curves_speed = speed[np.asarray(listOfCurves)[:, 1].astype(int)] # spped in the cener of the curve  
        curves_bins = np.digitize(curves_curvature, curvature_bins)    
        
        for b in range(1, len(curvature_bins)):
            if (len(curves_speed[curves_bins==b]) > 0):
                curves_speed_bins[b-1] = np.median(curves_speed[curves_bins==b])
       
    return (curves_speed_bins)

#######################    
if __name__ == "__main__":
    main()