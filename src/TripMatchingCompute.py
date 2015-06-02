# -*- coding: utf-8 -*-
"""
Created on Thu May 14 21:15:53 2015

@author: jburgues
"""

import numpy as np
import pandas as pd
import AuxiliarFunctions as auxf
from dtw import dtwAll
from rdp import rdp_numpy
from os import listdir
from os.path import isfile, join
import sys
sys.setrecursionlimit(10000)

#%% 
def tripMatching(path, driver,  dtw_thr):
    path_driver = join(path,driver) 
    trip_list = [f for f in listdir(join(path_driver)) if isfile(join(path_driver,f)) and not f.startswith('.')]
    features_fus = []
    trip_lengths = []
    for trip in trip_list:  
        # Load trip coordinates    
        path_trip = join(path_driver, trip)
        coords = np.array(pd.read_csv(path_trip))
        
        # Trip simplification using RDP
        rdp_mask = rdp_numpy(coords, epsilon=10) # Boolean mask
    
        # Create the feature vectors
        dists_rdp = auxf.diffDist(coords[rdp_mask])
        angles_rdp = auxf.diffAngle(coords[rdp_mask])
        dists_and_angles = zip(angles_rdp, dists_rdp)
        features_fus.append(dists_and_angles)
        trip_lengths.append(len(coords))
    
    
    ##%% TRIP-MATCHING
    features_fus = np.array(features_fus) # Convert back to np.array
    S = np.zeros([len(trip_list), len(trip_list)])
    for i in range(0, len(trip_list)-1):
        #print "trip i: ",  i
        len_feat_i = len(features_fus[i])
        len_trip_i = trip_lengths[i]
        max_angle_i = max(zip(*features_fus[i])[0])
        max_len_i = max(zip(*features_fus[i])[1])
        for j in range(i+1, len(trip_list)):            
            len_feat_j = len(features_fus[j])            
            len_trip_j = trip_lengths[j]
            len_diff = abs(len_trip_i - len_trip_j)
            len_max = float(max(len_trip_i, len_trip_j))     
            if ((len_feat_i != 0) & (len_feat_j != 0)):
                if ((len_diff/len_max <= 0.2)): # trips with very different lengths are not dtw'ed
                    max_angle = max(max_angle_i, max(zip(*features_fus[j])[0]))  # Maxima along the first axis
                    max_len= max(max_len_i, max(zip(*features_fus[j])[1]))  # Maxima along the second axis     
                    feat_i_norm = features_fus[i]/np.array([max_len, max_angle])
                    feat_j_norm = features_fus[j]/np.array([max_len, max_angle])
                    dtw_dist,  dtw_trnsf = dtwAll(feat_i_norm, feat_j_norm)
                    if (dtw_dist<dtw_thr):
                        S[i,j] = S[j,i] = 1

    return S
