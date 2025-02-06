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

topdir='/data/EnigmaMeg/BIDS/UPMC_BL_MEG/sourcedata'
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
    tasktype=op.basename(fname)
    if 'empty' in tasktype.lower():
        if 'wm_empty' in tasktype.lower():
            task='empty'
            run=1
        elif 'clock_empty' in tasktype.lower():
            task='empty'
            run=2
        elif 'switch_empty' in tasktype.lower():
            task='empty'
            run=3
        else:
            task='empty'
            run=4
    elif 'rest' in tasktype.lower():
        if 'wm_rest' in tasktype.lower():
            task='eyesclosed'
            run=1
        elif 'clock_rest' in tasktype.lower():
            task='eyesclosed'
            run=2
        elif 'switch_rest' in tasktype.lower():
            task='eyesclosed'
            run=3
        else: 
            task='eyesclosed'
            run=4
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
    coreg.fit_icp(n_iterations=20, nasion_weight=2, hsp_weight= 1, verbose=True)
    return coreg.trans
    

dsets = glob.glob(op.join(topdir, '1*','*rest_raw.fif'))
dsets += glob.glob(op.join(topdir, '1*','*rest.fif'))
dsets += glob.glob(op.join(topdir, '1*','*restb_raw.fif'))
subjids = [i.split('sourcedata')[-1].split('/')[1].split('_')[0] for i in dsets]

# before adding in the empty rooms, make this into a dataframe and sort by first MEG

dframe_megrest = pd.DataFrame(zip(dsets,subjids), columns=['fname','subjid'])
df_megrest_sorted = dframe_megrest.sort_values(by='subjid')
df_megrest_first = df_megrest_sorted.drop_duplicates(subset='subjid',keep='first')

# now add in the emptyrooms

dsets += glob.glob(op.join(topdir, '1*', '*empty*.fif'))
mris = glob.glob(op.join(topdir, '1*','*t1*nii.gz'))
subjids = [i.split('sourcedata')[-1].split('/')[1].split('_')[0] for i in dsets]
subjids_mris = [i.split('sourcedata')[-1].split('/')[1].split('_')[0] for i in mris]
dframe = pd.DataFrame(zip(dsets,subjids), columns=['fname','subjid'])
dframe_mri = pd.DataFrame(zip(mris,subjids_mris), columns=['fname','subjid'])

dframe['task']= dframe.fname.apply(get_dset_info, **{'return_type':'task'})
dframe['run']= dframe.fname.apply(get_dset_info, **{'return_type':'run'})
dframe['session'] = '1'
dframe_mri['session'] = '1'


#Clean duplicate entries
dframe = dframe.drop_duplicates(subset=['subjid'], inplace=False).reset_index(drop=True, inplace=False)
dframe_mri = dframe_mri.drop_duplicates(subset=['subjid'], keep='first').reset_index(drop=True)


#for idx, row in dframe.iterrows():
#    if row.subjid[-2:] == 'V2':
#        dframe.loc[idx,'subjid'] = row.subjid[:-2]
#        dframe.loc[idx, 'session'] = '2'

csv_outfname = op.join(topdir, 'dframe.csv')
if op.exists(csv_outfname):
    os.remove(csv_outfname)
dframe.to_csv(csv_outfname)

# dframe_mri = dframe_mri.subjid.drop_duplicates(keep='first').reset_index(drop=True, inplace=True)

mri_csv_outfname = op.join(topdir, 'dframe_mri.csv')
if op.exists(mri_csv_outfname):
    os.remove(mri_csv_outfname)
dframe_mri.to_csv(mri_csv_outfname)


#####
# IMPORTANT
# At this point, you can edit the csv file to make sure you have no duplicate run numbers
# then re-open and continue processing
#####

dframe = pd.read_csv(csv_outfname)
dframe_mri = pd.read_csv(mri_csv_outfname)


# =============================================================================
#  Make Transform from automated fit
# =============================================================================

failed = []
for idx,row in dframe.iterrows():
    try:
        subjid_bids = 'sub-' + row.subjid
        fs_subjects_dir = op.join('derivatives','freesurfer','subjects')
        subjects_dir = op.join('derivatives','freesurfer','subjects',subjid_bids)
        #mne.bem.make_watershed_bem(subjid_bids,subjects_dir=fs_subjects_dir,gcaatlas=True, preflood=None, volume='T1',overwrite=True)
        #mne.bem.make_scalp_surfaces(subjid_bids, subjects_dir=fs_subjects_dir,force=True,overwrite=True)
        trans = do_auto_coreg(row.fname, subjid_bids, fs_subjects_dir)
        dframe.loc[idx,'trans_fname']=op.join(op.dirname(op.dirname(row.fname)), f'{row.subjid}_1_trans.fif')
        #if op.exists(df_megrest_first.loc[idx,'trans_fname']):
        #    os.remove(df_megrest_first.loc[idx,'trans_fname'])
        trans.save(dframe.loc[idx,'trans_fname'])
    except:
        failed.append(row.subjid)
# =============================================================================
# MEG BIDS
# =============================================================================

    bids_dir = op.join(op.dirname(topdir), 'BIDS')
    
    for idx,row in dframe.iterrows():
        logger = get_subj_logger(row.subjid, row.session, log_dir=logdir)

        logger.info(f'Starting MEG BIDS')
        raw = mne.io.read_raw_fif(row.fname)
        raw.info['line_freq'] = 60 
        ses = str(row.session)
        run = str(row.run)
        if len(run)==1: run='0'+run
        bids_path = BIDSPath(subject=str(row.subjid), session=ses, task=row.task,
                                  run=run, root=bids_dir, suffix='meg')
        write_raw_bids(raw, bids_path, overwrite=True)
        logger.info(f'Successful MEG BIDS: {bids_path.fpath}')


# =============================================================================
# MRI BIDS
# =============================================================================
failed_mri_bids=[]
for idx,row in dframe_mri.iterrows():
    try:
        subjid=str(row['subjid'])
        megfname=df_megrest_first[(df_megrest_first['subjid']==subjid)]['fname'].values[0]
        trans_fname=df_megrest_first[(df_megrest_first['subjid']==subjid)]['trans_fname'].values[0]
        raw = mne.io.read_raw_fif(megfname)
        trans = mne.read_trans(trans_fname)
            
        t1w_bids_path = \
                BIDSPath(subject=subjid, session=str(row.session), root=bids_dir, suffix='T1w')
        
        subjid_bids = 'sub-' + subjid
        fs_subjects_dir = op.join('derivatives','freesurfer','subjects')

        landmarks = mne_bids.get_anat_landmarks(
                #image=op.join(row.subjects_dir, row.subjid, 'mri','T1.mgz'),
                image=row['fname'],
                info=raw.info,
                trans=trans,
                fs_subject=subjid_bids,
                fs_subjects_dir=fs_subjects_dir
                )
            
        # Write regular
        t1w_bids_path = write_anat(
                image=row['fname'],
                bids_path=t1w_bids_path,
                landmarks=landmarks,
                deface=False, 
                overwrite=True
                )
    except BaseException as e:
        print(e)
        failed_mri_bids.append(row['subjid'])
            
# =============================================================================
# Link freesurfer data
# =============================================================================
#bids_subjects_dir = op.join(bids_dir, 'derivatives', 'freesurfer','subjects')
#if not(op.join(bids_subjects_dir)):
#    os.makedirs(bids_subjects_dir)
#
#for idx, row in dframe.iterrows():
#    logger = get_subj_logger(row.subjid, session=row.session, log_dir=logdir)
#    print(row.subjid, row.session)
#    if row.session == '1':
#        try:
#            out_slink = op.join(bids_subjects_dir,'sub-'+row.subjid)
#            if not(op.exists(out_slink)):
#                os.symlink(op.join(row.subjects_dir, row.subjid), out_slink)
#        except BaseException as e:
#            logger.exception('Could not link the subject freesurfer folder: {str(e)}')
#    else:
#        logger.warning('Freesurfer linking was skipped - mostly due to session = {row.session}')
#        continue
            

