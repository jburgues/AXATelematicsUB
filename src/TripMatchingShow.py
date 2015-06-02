# -*- coding: utf-8 -*-
"""
Created on Thu May 14 21:15:53 2015

@author: jburgues
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join
import sys
import json
sys.setrecursionlimit(10000)

# TRIP-MATCHING
print "Path configuration.."
plt.close("all")
path = './drivers'
driver = '2'
path_driver = join(path,driver) 
trip_list = [f for f in listdir(join(path_driver)) if isfile(join(path_driver,f)) and not f.startswith('.')]

#%% Load DTW distances & ordered trips
print "Loading input files.."
with open('rdp_points_driver2.txt',  'r') as json_data:
    trips_rdp = np.array(json.load(json_data))
    json_data.close()
with open('features_driver2.txt',  'r') as json_data:
    features_fus = np.array(json.load(json_data))
    json_data.close()
dtw_dists_driver = np.loadtxt('dtw_dists_driver2.txt')
dtw_trnsfs_driver = np.loadtxt('dtw_trnsfs_driver2.txt')
dtw_ndx_driver = np.loadtxt('dtw_ndx_driver2.txt')

#%% SHOW RESULTS
print "Showing results.."
for trip in ['172.csv']:
    # Plot ordered distances to other trips
    fig1 = plt.figure()
    fig1.suptitle('Driver: %s - Trip %s'%(driver,trip[:-4]))
    plt.plot(dtw_dists_driver[trip_list.index(trip)])
    
    # Show subset of matched trips
    ref_coords = np.array(pd.read_csv(join(path_driver, trip)))
    fig2, axarr = plt.subplots(1, 2,  sharex=False)
    fig2.suptitle('Driver: %s - Trip %s'%(driver,trip[:-4]))
    axarr[0].set_title('10 most similar trips')
    axarr[1].set_title('10 most different trips')
    closest_trips = np.array(trip_list)[dtw_ndx_driver[trip_list.index(trip)].astype(int)] 
    for ctrip in closest_trips[0:10]:
        coords = np.array(pd.read_csv(join(path_driver, ctrip)))
        dist = np.array(dtw_dists_driver)[trip_list.index(trip),closest_trips.tolist().index(ctrip)]
        axarr[0].plot(ref_coords[:,0], ref_coords[:,1], 'b', lw=3)
        axarr[0].plot(coords[:,0], coords[:,1], label='trip %s: %.1f'%(ctrip[:-4],dist))
        axarr[0].text(coords[-1, 0],  coords[-1, 1],  ctrip[:-4], fontsize=12)    
    for ctrip in closest_trips[-10:-1]:
        coords = np.array(pd.read_csv(join(path_driver, ctrip)))
        dist = np.array(dtw_dists_driver)[trip_list.index(trip),closest_trips.tolist().index(ctrip)]
        axarr[1].plot(ref_coords[:,0], ref_coords[:,1], 'b', lw=3)
        axarr[1].text(ref_coords[-1, 0],  ref_coords[-1, 1],  trip[:-4], fontsize=12)
        axarr[1].plot(coords[:,0], coords[:,1], label='trip %s: %.1f'%(ctrip[:-4],dist))
        axarr[1].text(coords[-1, 0],  coords[-1, 1],  ctrip[:-4], fontsize=12)
        
    axarr[0].legend(loc='upper right')
    axarr[1].legend(loc='upper right')
    
    plt.show() 

#%% Compare two individual trips
trip1 = '172.csv'
trip2 = '5.csv'
coords1 = np.array(pd.read_csv(join(path_driver, trip1)))
coords2 = np.array(pd.read_csv(join(path_driver, trip2)))
rdp1 = coords1[trips_rdp[trip_list.index(trip1)]]
rdp2 = coords2[trips_rdp[trip_list.index(trip2)]]
print "len t1: %s, t2: %s, feat1: %s, feat2: %s" %(len(coords1), len(coords2),  len(rdp1),  len(rdp2))

# Show features over trips silhouette
plt.figure()
plt.plot(coords1[:,0], coords1[:,1], 'b')
plt.plot(rdp1[:,0], rdp1[:,1], 'go')
for c in range(0, len(features_fus[trip_list.index(trip1)])):
    plt.text(rdp1[c, 0], rdp1[c, 1], "(%.1f, %.2f)" % (round(features_fus[trip_list.index(trip1)][c][1],2),  round(features_fus[trip_list.index(trip1)][c][0],2)), fontsize=10)
 
plt.plot(coords2[:,0], coords2[:,1], 'r')
plt.plot(rdp2[:,0], rdp2[:,1], 'go')
for c in range(0, len(features_fus[trip_list.index(trip2)])):
    plt.text(rdp2[c, 0], rdp2[c, 1], "(%.1f, %.2f)" % (round(features_fus[trip_list.index(trip2)][c][1],2),  round(features_fus[trip_list.index(trip2)][c][0],2)), fontsize=10)

# Show features plot in 1D
plt.figure()
plt.plot(zip(*features_fus[trip_list.index(trip1)])[1],  'b')
plt.plot(zip(*features_fus[trip_list.index(trip2)])[1], 'r')

plt.figure()
plt.plot(zip(*features_fus[trip_list.index(trip1)])[0],  'b')
plt.plot(zip(*features_fus[trip_list.index(trip2)])[0], 'r')

plt.show()
