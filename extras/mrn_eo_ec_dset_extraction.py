#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 20 09:36:36 2025

@author: jstout
"""

import mne
import os, os.path as op
import glob
import numpy as np
import sys


# CODES
# 6 -> eyes open begin
# 7 -> eyes open end
# 4 -> eyes closed begin
# 5 -> eyes closed end
# 8 -> eyes closed begin
# 9 -> eyes closed end
eo_on_id = 6
eo_off_id = 7
ec1_on_id = 4
ec1_off_id = 5
ec2_on_id = 8
ec2_off_id = 9



#%%
def write_dataset(fname):
    TaskName = op.basename(fname).split('_task-')[-1].split('_')[0]
    
    # Load the data
    raw = mne.io.read_raw_fif(fname, preload=True)
    evts = mne.find_events(raw, stim_channel='STI101')
    
    # EO events
    assert evts[evts[:,2]==eo_on_id].shape[0]==1, 'No eyes open onset'
    eo_onset = evts[evts[:,2]==eo_on_id][0][0]-raw.first_samp
    eo_offset = evts[evts[:,2]==eo_off_id][0][0]-raw.first_samp
    
    # EC events should be 30s each 
    assert evts[evts[:,2]==ec1_on_id].shape[0]==1, 'No eyes closed (part1) onset'
    assert evts[evts[:,2]==ec2_on_id].shape[0]==1, 'No eyese closed (part2) onset'
    
    ec1_onset = evts[evts[:,2]==ec1_on_id][0][0]-raw.first_samp
    ec1_offset = evts[evts[:,2]==ec1_off_id][0][0]-raw.first_samp
    
    ec2_onset = evts[evts[:,2]==ec2_on_id][0][0]-raw.first_samp
    ec2_offset = evts[evts[:,2]==ec2_off_id][0][0]-raw.first_samp
    
    
    # Extract segments
    _eo_data = raw._data[:, eo_onset : eo_offset]
    _ec1_data = raw._data[:, ec1_onset : ec1_offset]
    _ec2_data = raw._data[:, ec2_onset : ec2_offset]
    
    
    # Save out data 
    dirname, basename = op.dirname(fname), op.basename(fname)
    raw_eo_fname = op.join(dirname, basename.replace(TaskName, TaskName+'EO'))
    raw_ec_fname = op.join(dirname, basename.replace(TaskName, TaskName+'EC'))
    
    #EO data
    raw_eo = mne.io.RawArray(_eo_data, info=raw.info)
    print(f'Writing {raw_eo_fname}')
    raw_eo.save(raw_eo_fname, overwrite=True)
    
    #EC data (stitch 2 segments - remove offset)
    ec_stack = [_ec1_data]
    _ec2_data -= _ec1_data[:,-1][:,np.newaxis]
    ec_stack.append(_ec2_data)
    ec_array = np.concatenate(ec_stack, axis=-1)
    
    raw_ec = mne.io.RawArray(ec_array, info=raw.info)
    print(f'Writing {raw_ec_fname}')
    raw_ec.save(raw_ec_fname, overwrite=True)

fname = sys.argv[1]
print(f'Processing {fname}:')
try:
    write_dataset(fname)
except BaseException as e:
    print(f'Failed: {fname}')
    with open('FAILED_convert.txt', 'a') as f:
        f.write(fname + '\n')
        f.write(str(e))




# # Add BAD designation to all breaks in data
# new_evts = evts[::2,:]
# new_evts[0,0]=0
# current_start=0
# for row_idx in range(new_evts.shape[0]-1):
#     print(row_idx)
#     current_start+=(new_evts[row_idx, 1] / sfreq)
#     new_evts[row_idx, 0]=current_start -1 


# annots = mne.Annotations(new_evts[:,0], 2, description='BAD_seg')
# raw_hack.set_annotations(annots)

# raw_hack.save(fname.replace('.ds','.fif'))
