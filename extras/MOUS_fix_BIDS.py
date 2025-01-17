#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 11 11:14:02 2022

@author: jstout
"""

import logging
logging.basicConfig(
                level=logging.INFO,
                force=True,
                format='%(asctime)s %(message)s',
                handlers=[logging.FileHandler("mous_fid_conv.log")])

import os
import os.path as op

import json
import copy
import nibabel as nb
import numpy as np
import glob
from mne.transforms import apply_trans, invert_transform
import mne

def list_nparr2list(nparray):
    return [float(i) for i in nparray]
    
def convert_headcoils2mm(coordsys):
    '''If the fiducials are in cm convert to mm'''    
    coordsys = copy.copy(coordsys)
    #Convert to mm
    if coordsys["HeadCoilCoordinateUnits"].lower()=='cm':
        fids = {key:np.array(value)*10 for key,value in coordsys["HeadCoilCoordinates"].items()}
        fids = {key:list_nparr2list(value) for key,value in fids.items()}
        coordsys["HeadCoilCoordinates"]=fids
        coordsys["HeadCoilCoordinateUnits"]='mm'
    return coordsys

def convert_ctf2t1(fidval, ctfmat):
    '''Provide the voxel index
    Assumes that both input and output mats are the same size'''
    i0,i1,i2=fidval
    l0,l1,l2 = ctfmat.shape
    i0=l0-1-i0               # note because of zero indexing we need to subtract 1 from the dimension
    o0 = i2
    o1 = i0
    o2 = i1
    return o0,o1,o2 

def write_anat_json(anat_json=None, 
                    fids=None,
                    overwrite=False):
    '''Write the AnatomicalLandmark keys from the meg json to the mri json.
    Takes the dictionary pulled from the MEG json - (use get_anat_json)'''
    with open(anat_json) as w:
        json_in = json.load(w)
        
    #Check to see if the AnatomicalLandmarks are already present 
    anat_keys = {x:y for (x,y) in json_in.items() if 'AnatomicalLandmark' in x}
    if overwrite==False:
        assert anat_keys == {}

    json_in['AnatomicalLandmarkCoordinates']=fids  
    
    #Write json
    with open(anat_json, 'w') as w:
        json.dump(json_in, w, indent='    ') 


def convert_single_subject(bids_root=None,
                           subject=None,
                           overwrite=True
                           ):
    
    coordsys_fname = f'{bids_root}/{subject}/meg/{subject}_coordsystem.json'
    with open(coordsys_fname) as w:
        coordsys = json.load(w)
        
    intend_for = coordsys['IntendedFor']
    intended_for = f'{bids_root}/{subject}/{intend_for}'
    
    if 'space-CTF' not in op.basename(intended_for).split('_'):
        raise('space-CTF is not in the intended for label')
    
    anat_t1w = intended_for.replace('space-CTF_','')
    anat_t1w_json = anat_t1w.replace('.nii','.json')
    
    fids = convert_headcoils2mm(coordsys)['HeadCoilCoordinates']
    
    #Get the FIDS
    fid_arr=np.stack([np.array(fids['nasion']), 
             np.array(fids['left_ear']),
             np.array(fids['right_ear'])])
    
    #CTFmri
    ctf_mri_fname=intended_for
    ctf_mri=nb.load(ctf_mri_fname)
    aff = ctf_mri.affine
    mat = ctf_mri.get_fdata()
    inv_trans = invert_transform(mne.Transform('ctf_meg','mri_voxel', aff))
    fid_vox = apply_trans(inv_trans, fid_arr)
    
    nasT1V=convert_ctf2t1(fid_vox[0], mat)
    lpaT1V=convert_ctf2t1(fid_vox[1], mat)
    rpaT1V=convert_ctf2t1(fid_vox[2], mat)
    
    t1_vox_fids=dict(NAS=nasT1V,
                     LPA=lpaT1V,
                     RPA=rpaT1V)
    write_anat_json(anat_json=anat_t1w_json, 
                    fids=t1_vox_fids,
                    overwrite=overwrite)
    logging.info(f'Success: {subject}')
    
    
def convert_mous_project(bids_dir=None):
    '''Loop over all subjects and add the Voxel index Fiducials to the 
    T1w.json file in the anatomy folder
    '''
    bids_subjs = glob.glob(op.join(bids_dir, 'sub-*'))
    bids_subjs = [op.basename(i) for i in bids_subjs] 
    
    for subjid in bids_subjs:
        print(f'Running: {subjid}')
        logging.info(f'Running: {subjid}')
        try:
            convert_single_subject(bids_dir, subject=subjid, overwrite=True)
        except BaseException as e:
            print(f'Error with {subjid}:\n {e}')
            logging.Exception(f'Error with {subjid}:\n {e}')
        

if __name__=='__main__':
    import sys
    bids_dir = sys.argv[1]
    convert_mous_project(bids_dir=bids_dir)

# =============================================================================
# TESTS
# =============================================================================
def test_unitconv():
    json=	{"HeadCoilCoordinateUnits": "cm",
    	"HeadCoilCoordinates": {
    		"nasion": [10.5092,0,0],
    		"left_ear": [-0.014438,7.03839,0],
    		"right_ear": [0.014438,-7.03839,0]
    	}}
    gtruth =  {'HeadCoilCoordinateUnits': 'mm',
     'HeadCoilCoordinates': {'nasion': [105.092, 0.0, 0.0],
      'left_ear': [-0.14438, 70.3839, 0.0],
      'right_ear': [0.14438, -70.3839, 0.0]}}
    assert convert_headcoils2mm(json)==gtruth
    
def test_ctf2t1():
    nasV = [32,122,87]
    lpaV = [137,93,22]
    rpaV = [130,96,163]
    nasT1V=convert_ctf2t1(nasV, t1ctfmat)
    lpaT1V=convert_ctf2t1(lpaV, t1ctfmat)
    rpaT1V=convert_ctf2t1(rpaV, t1ctfmat)
    
    assert nasT1V==(87,224, 122)
    assert lpaT1V==(22,119,93)
    assert rpaT1V==(163, 126, 96)
