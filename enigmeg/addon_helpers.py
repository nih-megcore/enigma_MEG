#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  6 16:37:09 2026

@author: nugenta and jstout
"""

import json
import numpy as np
import nibabel as nb
import mne
from mne.io.constants import FIFF
import os, os.path as op
import mne_bids


class ctf_hs_mne_bids_coreg():
    '''
    Generate the coregstration matrix when using CTF data, where the fiducial
    locations are listed in the headshape and not the data raw.info["dig"] block
    as NAS/LPA/RPA
    '''
    def __init__(
            self,
            subjects_dir=None,
            meg_raw_fname=None, 
            anat_fname=None):
        self.subjects_dir=subjects_dir
        assert op.exists(meg_raw_fname), f'{meg_raw_fname} does not exist'
        self.meg_raw_fname = meg_raw_fname
        assert op.exists(anat_fname), f'{anat_fname} does not exist'
        self.anat_fname = anat_fname
        self.anat_landmarks_array = self.extract_anat_landmarks_array()
        
        if self.coordsys==False:
            raise ValueError(f'''The coordsys file associated with {self.meg_raw_fname}
                             does not exist''')
        
    
    @property        
    def coordsys(self):
        _bids_tmp = mne_bids.get_bids_path_from_fname(self.meg_raw_fname)
        _bids_tmp.update(task=None, run=None, suffix='coordsystem',extension='.json')
        if op.exists(_bids_tmp.fpath):
            return _bids_tmp.fpath
        else:
            return False
    
    def load_coordsys(self):
        with open(self.coordsys,'r') as f:
            coordsys = json.load(f)
        return coordsys
    
    @property
    def digspace_landmarks(self):
        'Extract the digspace landmarks and return the matrix in NAS/LPA/RPA order'
        coordsys = self.load_coordsys()
        assert 'AnatomicalLandmarkCoordinates' in coordsys.keys(), f'Missing AnatomicalLandmarkCoordinates tag in coordsys json'
        digspace_landmarks = coordsys['AnatomicalLandmarkCoordinates']
        landmark_order = ['NAS', 'LPA', 'RPA']
        assert 'NAS' in digspace_landmarks
        assert 'LPA' in digspace_landmarks
        assert 'RPA' in digspace_landmarks
        digspace_landmarks_array_m = np.array([digspace_landmarks[key] for key in landmark_order])/100
        return digspace_landmarks_array_m
    
    @property
    def transset(self):
        meg_res4 = mne.io.ctf.res4._read_res4(self.meg_raw_fname)
        meg_coils = mne.io.ctf.hc._read_hc(self.meg_raw_fname)
        transset = mne.io.ctf.trans._make_ctf_coord_trans_set(meg_res4, meg_coils)
        return transset
        
    @property
    def headspace_landmarks_m(self):
        headspace_landmarks_m = mne.transforms.apply_trans(
            self.transset['t_ctf_head_head'], self.digspace_landmarks)
        return headspace_landmarks_m
    
    def extract_anat_landmarks_array(self):
        '''open the anatomical .json to extract the landmark coordinates in 
        mri space - these are in voxel coords'''
        anat_img_path = self.anat_fname
        if anat_img_path.endswith('.nii.gz'):
            anat_json_path = anat_img_path[:-len('.nii.gz')] + '.json'
        elif anat_img_path.endswith('.nii'):
            anat_json_path = anat_img_path[:-len('.nii')] + '.json'
        
        with open(anat_json_path,'r') as f:
            anat_json = json.load(f)
        
        assert 'AnatomicalLandmarkCoordinates' in anat_json, f'AnatomicalLandmarkCoordinates not in {anat_json_path}'
        anat_landmarks = anat_json['AnatomicalLandmarkCoordinates']
        landmark_order = ['NAS', 'LPA', 'RPA']
        anat_landmarks_array = np.array([anat_landmarks[key] for key in landmark_order])
        return anat_landmarks_array
    
    def extract_head_mri_t(self):
        '''convert the voxel space anatomical landmarks to ras space using the 
        affine transform, and calculate and subtract cras'''
        im = nb.load(self.anat_fname)
        anat_landmarks_xyz_ras = mne.transforms.apply_trans(im.affine, 
                                                            self.anat_landmarks_array)
        anat_landmarks_xyz_ras_m = anat_landmarks_xyz_ras/1000
        cras = im.affine.dot(np.hstack((np.array(im.shape[:3]) / 2.0, [1])))[:3]
        cras_m = np.array(cras/1000)
        print(cras_m)

        anat_landmarks_xyz_ras_m_cras = anat_landmarks_xyz_ras_m - cras_m

        # derive the transform from the headspace digitized anatomical landmarks to the mri space anatomical landmarks
        xform = mne.coreg.fit_matched_points(self.headspace_landmarks_m, 
                                             anat_landmarks_xyz_ras_m_cras)
        head_mri_t = mne.transforms.Transform(FIFF.FIFFV_COORD_HEAD, 
                                              FIFF.FIFFV_COORD_MRI, 
                                              xform)
        self.head_mri_t = head_mri_t
        return head_mri_t
    
    def plot_fs_coreg(self):
        '''Plot the freesurfer coreg'''
        #HACK - not sure if this holds the plot open
        
        fs_subject = op.basename(self.meg_raw_fname).split('_')[0]
        
        # fs_subject = 'sub-' + self.subject
        _raw = mne.io.read_raw_ctf(self.meg_raw_fname)
        info = _raw.info #mne.io.read_info(self.meg_raw_fname)
        if not hasattr(self, 'head_mri_t'): self.extract_head_mri_t()
        mne.viz.plot_alignment(info,trans=self.head_mri_t, 
                               subject=fs_subject,
                               subjects_dir=self.subjects_dir,
                               surfaces='head',meg=['helmet','sensors'], 
                               coord_frame='mri')
        
        



    
    
    
    
    
    
        


#%%
bids_root='/data/EnigmaMeg/BIDS/OMEGA2'
meg_fname = 'sub-0001/ses-01/meg/sub-0001_ses-01_task-rest_run-05_meg.ds'

import glob
subj='sub-CONP0002'
meg_dsets = glob.glob(op.join(bids_root, subj, 'ses-01', 'meg', '*rest*.ds'))
meg_fname='/data/EnigmaMeg/BIDS/OMEGA2/sub-CONP0002/ses-01/meg/sub-CONP0002_ses-01_task-rest_run-01_meg.ds'

subject='PD1751',
_tmp = ctf_hs_mne_bids_coreg(
                      anat_fname = '/data/EnigmaMeg/BIDS/OMEGA2/sub-CONP0002/ses-02/anat/sub-CONP0002_ses-02_run-1_T1w.nii.gz',
                      subjects_dir='/data/EnigmaMeg/BIDS/OMEGA2/derivatives/freesurfer/subjects', 
                      meg_raw_fname=meg_fname,
                      )
print(_tmp.digspace_landmarks)
print(_tmp.headspace_landmarks_m)
# _tmp.extract_head_mri_t()

# if _tmp.coordsys:
#     print(f'{_tmp.coordsys} exists')

# _tmp.digspace_landmarks
    
    
#%%




# open the anatomical .json to extract the landmark coordinates in mri space - these are in voxel coords

# anat_img_path = process_subj.fnames['anat']

# if anat_img_path.endswith('.nii.gz'):
#     anat_json_path = anat_img_path[:-len('.nii.gz')] + '.json'
# elif anat_img_path.endswith('.nii'):
#     anat_json_path = anat_img_path[:-len('.nii')] + '.json'

# with open(anat_json_path,'r') as f:
#     anat_json = json.load(f)

# anat_landmarks = anat_json['AnatomicalLandmarkCoordinates']
# landmark_order = ['NAS', 'LPA', 'RPA']
# anat_landmarks_array = np.array([anat_landmarks[key] for key in landmark_order])

# print(anat_landmarks_array)

# convert the voxel space anatomical landmarks to ras space using the affine transform, and
# calculate and subtract cras

# im = nb.load(anat_img_path)
# anat_landmarks_xyz_ras = mne.transforms.apply_trans(im.affine, anat_landmarks_array)
# anat_landmarks_xyz_ras_m = anat_landmarks_xyz_ras/1000
# cras = im.affine.dot(np.hstack((np.array(im.shape[:3]) / 2.0, [1])))[:3]
# cras_m = np.array(cras/1000)
# print(cras_m)

# anat_landmarks_xyz_ras_m_cras = anat_landmarks_xyz_ras_m - cras_m

# # derive the transform from the headspace digitized anatomical landmarks to the mri space anatomical landmarks

# xform = mne.coreg.fit_matched_points(headspace_landmarks_m, anat_landmarks_xyz_ras_m_cras)
# head_mri_t = mne.transforms.Transform(FIFF.FIFFV_COORD_HEAD, FIFF.FIFFV_COORD_MRI, xform)

# # I added in this to visualize so I could check

# fs_subject = 'sub-' + subject
# mne.viz.plot_alignment(meg_omega.info,trans=head_mri_t, subject=fs_subject,subjects_dir=subjects_dir,
#  surfaces='head',meg=['helmet','sensors'], coord_frame='mri')