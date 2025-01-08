#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 12:18:23 2023

@author: jstout
"""


import glob
import os, os.path as op
import mne 
import mne_bids
from mne_bids import BIDSPath, write_raw_bids, write_anat
import pandas as pd
import logging

logger=logging.getLogger()
logdir  = op.join(os.getcwd(), 'logdir')
if not op.exists(logdir):
    os.mkdir(logdir)
topdir='/data/EnigmaMeg/BIDS/REMPE'
os.chdir(topdir)

def get_subj_logger(subjid, session, log_dir=None):
     '''Return the subject specific logger.
     This is particularly useful in the multiprocessing where logging is not
     necessarily in order'''
     fmt = '%(asctime)s :: %(levelname)s :: %(message)s'
     sub_ses = f'{subjid}_ses_{session}'
     subj_logger = logging.getLogger(sub_ses)
     if subj_logger.handlers != []: # if not first time requested, use the file handler already defined
         tmp_ = [type(i) for i in subj_logger.handlers ]
         if logging.FileHandler in tmp_:
             return subj_logger
     else: # first time requested, add the file handler
         fileHandle = logging.FileHandler(f'{log_dir}/{subjid}_ses-{session}_log.txt')
         fileHandle.setLevel(logging.INFO)
         fileHandle.setFormatter(logging.Formatter(fmt)) 
         subj_logger.addHandler(fileHandle)
         subj_logger.setLevel(logging.INFO)
         subj_logger.info('Initializing subject level enigma_anonymization log')
     return subj_logger   


def get_dset_info(fname, return_type=None):
    if return_type=='run':
        return '01'
    elif return_type=='task':
        return 'rest'


def do_auto_coreg(raw_fname, subject, subjects_dir):
    '''Localize coarse fiducials based on fsaverage coregistration
    and fine tuned with iterative headshape fit.'''
    raw_rest = mne.io.read_raw_fif(raw_fname)
    if not op.exists(op.join(subjects_dir, subject, 'bem')):
        mne.bem.make_watershed_bem(subject, subjects_dir=subjects_dir)
    
    coreg = mne.coreg.Coregistration(raw_rest.info, 
                                     subject=subject,
                                     subjects_dir=subjects_dir, 
                                     fiducials='estimated')
    coreg.fit_fiducials(verbose=True)
    coreg.omit_head_shape_points(distance=5. / 1000)  # distance is in meters
    coreg.fit_icp(n_iterations=6, nasion_weight=.5, hsp_weight= 5, verbose=True)
    return coreg.trans

def get_transfile(meg_fname):
    meg_dir = op.dirname(op.dirname(meg_fname))
    trans_all = glob.glob(op.join(meg_dir, '*trans.fif'))
    if len(trans_all) == 0:
        tmp_ = meg_fname.split('sub-')
        return tmp_[0]+tmp_[1][0:4]+'trans.fif'
    elif len(trans_all) > 1:
        curr = mne.read_trans(trans_all[0])
        for i in trans_all[1:]:
            tmp_ = mne.read_trans(i)
            if tmp_ != curr:
                return 'Multiple'
        return trans_all[0]
    else:
        return trans_all[0]
    
    

dsets = glob.glob(op.join(topdir, 'source_data','ecr_fifs','*.fif'))
subjids = [i.split('ecr_fifs')[-1].split('/')[1][0:7] for i in dsets]
dframe = pd.DataFrame(zip(dsets,subjids), columns=['fname','subjid'])

dframe['task']= dframe.fname.apply(get_dset_info, **{'return_type':'task'})
dframe['run']= dframe.fname.apply(get_dset_info, **{'return_type':'run'})
dframe['subjects_dir']=op.join(topdir, 'derivatives','freesurfer','subjects') 
dframe['session'] = '1'
dframe['transfile'] = dframe.fname.apply(get_transfile)

csv_outfname = op.join(logdir, 'dframe.csv')
if op.exists(csv_outfname):
    os.remove(csv_outfname)
dframe.to_csv(op.join(logdir, 'dframe.csv'))


    
# =============================================================================
#  Make Transform from automated fit
# =============================================================================
failed=[]
for idx,row in dframe.iterrows():
    logger = get_subj_logger(row.subjid, row.session, log_dir=logdir)
    try:
        if (row.run=='01') & (row.task not in ['empty','closed']):
            logger.info(f'Estimating Transform')
            trans = do_auto_coreg(row.fname, row.subjid, row.subjects_dir)
            dframe.loc[idx,'trans_fname']=op.join(op.dirname(op.dirname(row.fname)), f'{row.subjid}_{row.session}_trans.fif')
            if op.exists(dframe.loc[idx,'trans_fname']):
                os.remove(dframe.loc[idx,'trans_fname'])
            trans.save(dframe.loc[idx,'trans_fname'])
            logger.info(f'Successfully saved trans file')
    except BaseException as e:
        print(row.fname)
        logger.exception(str(e))
        failed.append(row.fname)


# =============================================================================
# MEG BIDS
# =============================================================================
dframe['newsubjid']=dframe.subjid.str[4:]
bids_dir = op.join(topdir, 'BIDS')
for idx,row in dframe.iterrows():
    logger = get_subj_logger(row.subjid, row.session, log_dir=logdir)
    try:
        logger.info(f'Starting MEG BIDS')
        raw = mne.io.read_raw_fif(row.fname)
        raw.info['line_freq'] = 60 
        ses = row.session
        run = str(row.run)
        if len(run)==1: run='0'+run
        bids_path = BIDSPath(subject=row.newsubjid, session=ses, task=row.task,
                              run=run, root=bids_dir, suffix='meg')
        write_raw_bids(raw, bids_path, overwrite=True)
        logger.info(f'Successful MEG BIDS: {bids_path.fpath}')
    except BaseException as e:
        print(row.fname)
        logger.exception(f'failed MEG BIDS:  {str(e)}') 

# =============================================================================
# MRI BIDS
# =============================================================================

for idx,row in dframe.iterrows():
    logger = get_subj_logger(row.subjid, row.session, log_dir=logdir)
    try:
        raw = mne.io.read_raw_fif(row.fname)
        trans_fname = row.trans_fname 
        trans = mne.read_trans(trans_fname)
        
        t1w_bids_path = \
            BIDSPath(subject=row.newsubjid, session=row.session, root=bids_dir, suffix='T1w')
    
        landmarks = mne_bids.get_anat_landmarks(
            image=op.join(row.subjects_dir, row.subjid, 'mri','T1.mgz'),
            info=raw.info,
            trans=trans,
            fs_subject=row.subjid,
            fs_subjects_dir=row.subjects_dir
            )
        logger.info('Calc-ed landmarks')
        
        # Write regular
        t1w_bids_path = write_anat(
            image=op.join(row.subjects_dir, row.subjid, 'mri','T1.mgz'),
            bids_path=t1w_bids_path,
            landmarks=landmarks,
            deface=False, 
            overwrite=True
            )
        logger.info(f'Successful MRI BIDS: {t1w_bids_path.fpath}')
    except BaseException as e:
        logger.exception(str(e))
            
# =============================================================================
# mne_bids won't write out the MEG dataset
# run the following in bash
#
# =============================================================================
# cd /data/EnigmaMeg/BIDS/REMPE/
# for megf in  source_data/ecr_fifs/*.fif; do tmp=$(basename $megf); subjid=${tmp:0:7}; echo cp $megf BIDS/${subjid}/ses-1/meg/${subjid}_ses-1_task-rest_run-01_meg.fif ; done

#      ----- Run the above to finish up the BIDS ------ 

trans_dir = '/data/EnigmaMeg/BIDS/REMPE/source_data'
fif_dir = '/data/EnigmaMeg/BIDS/REMPE/source_data/ecr_fifs'
nii_dir = '/data/EnigmaMeg/BIDS/REMPE/source_data/raw_nii'
fs_dir = '/data/EnigmaMeg/BIDS/REMPE/BIDS/derivatives/freesurfer/subjects'

# Only 388 final datasets

#Freesurfer
tmp = glob.glob(fs_dir + '/sub-*')
fs_subjs = [op.basename(i) for i in tmp]
print(f'Freesurfers:{len(fs_subjs)}')

#Raw niftis
tmp = glob.glob(nii_dir + '/sub-*')
nii_subjs =  [op.basename(i) for i in tmp]
print(f'Niftis:{len(nii_subjs)}')
