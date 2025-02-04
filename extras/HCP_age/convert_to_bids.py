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
topdir='/data/EnigmaMeg/BIDS/UPITT_Age_HCP/sourcedata'
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
    tasktype=op.basename(op.dirname(fname))
    if 'empty' in tasktype.lower():
        task='empty'
    elif 'close' in tasktype.lower():
        task='eyesclosed'
    elif 'open' in tasktype.lower():
        task='eyesopen'
    elif 'rest' in tasktype.lower():
        task='eyesNA'
    
    if tasktype[-1]=='1':
        run=1
    elif tasktype[-1]=='2':
        run=2
    else:
        run=1
    if return_type == 'run':
        return run
    else:
        return task


def do_auto_coreg(raw_fname, subject, subjects_dir):
    '''Localize coarse fiducials based on fsaverage coregistration
    and fine tuned with iterative headshape fit.'''
    raw_rest = mne.io.read_raw_fif(raw_fname)
    coreg = mne.coreg.Coregistration(raw_rest.info, 
                                     subject=subject,
                                     subjects_dir=subjects_dir, 
                                     fiducials='estimated')
    coreg.fit_fiducials(verbose=True)
    coreg.omit_head_shape_points(distance=15. / 1000)  # distance is in meters
    coreg.fit_icp(verbose=True) #n_iterations=20, nasion_weight=10, hsp_weight= 1, verbose=True)
    return coreg.trans

def get_transfile(meg_fname):
    meg_dir = op.dirname(op.dirname(meg_fname))
    trans_all = glob.glob(op.join(meg_dir, '*trans.fif'))
    if len(trans_all) == 0:
        return None
    elif len(trans_all) > 1:
        curr = mne.read_trans(trans_all[0])
        for i in trans_all[1:]:
            tmp_ = mne.read_trans(i)
            if tmp_ != curr:
                return 'Multiple'
        return trans_all[0]
    else:
        return trans_all[0]
    
    
    

dsets = glob.glob(op.join(topdir, 'HCP*','unprocessed','MEG','*','*raw.fif'))
dsets += glob.glob(op.join(topdir, 'HCP*', 'unprocessed','MEG', 'EMPTY', '*.fif'))
subjids = [i.split('sourcedata')[-1].split('/')[1] for i in dsets]
dframe = pd.DataFrame(zip(dsets,subjids), columns=['fname','subjid'])

dframe['task']= dframe.fname.apply(get_dset_info, **{'return_type':'task'})
dframe['run']= dframe.fname.apply(get_dset_info, **{'return_type':'run'})
dframe['subjects_dir']=topdir + '/' +  dframe.subjid  + '/T1w'
dframe['session'] = '1'
dframe['transfile'] = dframe.fname.apply(get_transfile)

for idx, row in dframe.iterrows():
    if row.subjid[-2:] == 'V2':
        dframe.loc[idx,'subjid'] = row.subjid[:-2]
        dframe.loc[idx, 'session'] = '2'

csv_outfname = op.join(logdir, 'dframe.csv')
if op.exists(csv_outfname):
    os.remove(csv_outfname)
dframe.to_csv(op.join(logdir, 'dframe.csv'))

manual_check = dframe[dframe['transfile']=='Multiple']
dframe = dframe[dframe['transfile']!='Multiple']
    
# =============================================================================
#  Make Transform from automated fit
# =============================================================================
failed=[]
for idx,row in dframe.iterrows():
    logger = get_subj_logger(row.subjid, row.session, log_dir=logdir)
    try:
        if (row.run==1) & (row.task not in ['empty','closed']):
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

bids_dir = op.join(op.dirname(topdir), 'BIDS')
for idx,row in dframe.iterrows():
    logger = get_subj_logger(row.subjid, row.session, log_dir=logdir)
    try:
        logger.info(f'Starting MEG BIDS')
        raw = mne.io.read_raw_fif(row.fname)
        raw.info['line_freq'] = 60 
        ses = row.session
        run = str(row.run)
        if len(run)==1: run='0'+run
        bids_path = BIDSPath(subject=row.subjid, session=ses, task=row.task,
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
    if (row.task != 'eyesopen') & (row.run != 1):
        logger.info(f'Ignoring {row.fname}')
        continue
    else:
        try:
            raw = mne.io.read_raw_fif(row.fname)
            trans_fname = row.trans_fname 
            trans = mne.read_trans(trans_fname)
            
            t1w_bids_path = \
                BIDSPath(subject=row.subjid, session=row.session, root=bids_dir, suffix='T1w')
        
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
# Link freesurfer data
# =============================================================================
bids_subjects_dir = op.join(bids_dir, 'derivatives', 'freesurfer','subjects')
if not(op.join(bids_subjects_dir)):
    os.makedirs(bids_subjects_dir)

for idx, row in dframe.iterrows():
    logger = get_subj_logger(row.subjid, session=row.session, log_dir=logdir)
    print(row.subjid, row.session)
    if row.session == '1':
        try:
            out_slink = op.join(bids_subjects_dir,'sub-'+row.subjid)
            if not(op.exists(out_slink)):
                os.symlink(op.join(row.subjects_dir, row.subjid), out_slink)
        except BaseException as e:
            logger.exception('Could not link the subject freesurfer folder: {str(e)}')
    else:
        logger.warning('Freesurfer linking was skipped - mostly due to session = {row.session}')
        continue
            

