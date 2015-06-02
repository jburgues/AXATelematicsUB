# -*- coding: utf-8 -*-
"""
Created on Thu May 14 21:15:53 2015

@author: jburgues
"""

from os import listdir
from os.path import isdir, join
import sys
import numpy as np
sys.setrecursionlimit(10000)
import TripMatchingCompute as tmc

#%% MAIN LOOP
path = './drivers'

driver_list = [int(f) for f in listdir(path) if isdir(join(path,f)) and not f.startswith('.')]
driver_list.sort()
start = 101
end = 200
dtw_thr = 7

allMatches = []
driverNames = []
for driver in driver_list[start:end]:
#for driver in driver_list[start:end]:
    print 'driver: ',  driver
    S = tmc.tripMatching(path, str(driver),  dtw_thr)
    S_sum = np.sum(S, axis=1)
    allMatches.append(S_sum)
    driverNames.append(driver)

np.savetxt('tripMatching_dr%s_dr%s_thr%s.txt'%(start,end, dtw_thr), allMatches, fmt='%d')
np.savetxt('driverNames_dr%s_dr%s_thr%s.txt'%(start,end, dtw_thr), np.array(driverNames).astype(int), fmt='%d')
