#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 28 12:55:04 2023

@author: nugenta
"""

import os, os.path as op 
import subprocess
import mne
from mne_bids import write_anat, write_raw_bids, BIDSPath
import mne_bids
import shutil as sh
import mne_bids.copyfiles

topdir = '/Volumes/EnigmaMeg/BIDS/HCP/BIDS'
rawdir = '/Volumes/EnigmaMeg/RAW_uploads/HCP'

for subjid in ['100307','106521','102816','109123','116524','146129','156334','164636',
'174841','179245','189349','192641','205119','221319','250427','287248','353740','512835','581450','662551','680957',
'725751','825048','898176','105923','111514','116726','149741','158136','166438','175237','181232',
'191033','195041','212318','223929','255639','293748','358144','555348', '599671',
'665254','706040', '735148','872764','912447','108323','113922','140117','154532','162935',
'172029','177746','187547','191841','204521','214524','248339',
'283543', '352738', '433839', '568963', '660951', '679770', '715950', '814649', '891667', '990366', '112920', '133019', '153732', '162026', 
'169040', '175540', '185442', '191437', '198653', '212823', '233326', '257845', '352132', '406836', '559053', '601127', '667056', '707749', '783462', '877168']:
    
    bids_subjid = 'sub-'+subjid

    # set freesurfer folder

    freesurfer_dir = f'{topdir}/derivatives/freesurfer'
    subjects_dir = f'{freesurfer_dir}/subjects'
    os.environ['SUBJECTS_DIR'] = subjects_dir
    
    #Directory to save html files
    QA_dir = f'{topdir}/derivatives/BIDS_ANON_QA'
    if not os.path.exists(QA_dir): os.mkdir(QA_dir)
    
    source_fsdir = f'{rawdir}/HCP_MEG/{subjid}/T1w/{subjid}'
    dest_fsdir = f'{subjects_dir}/sub-{subjid}'
    subprocess.run(['cp', '-r', source_fsdir, dest_fsdir])

    meg_fname=f'{rawdir}/data/{subjid}/3-Restin/4D/c,rfDC'
    eroom_fname=f'{rawdir}/data/{subjid}/1-Rnoise/4D/c,rfDC'

    raw=mne.io.read_raw_bti(meg_fname, head_shape_fname=None, convert=False)     # This code can be uncommented and used if the BTI/4D code
    for i in range(len(raw.info['chs'])):                                       # was processed similar to the HCP data
        raw.info['chs'][i]['coord_frame'] = 1
        
    #tmp_path = raw._init_kwargs['pdf_fname'].split('c')[0]
    #raw._init_kwargs['config_fname'] = tmp_path+'config'
    
    emptyroom=mne.io.read_raw_bti(eroom_fname, head_shape_fname=None, convert=False)
    for i in range(len(emptyroom.info['chs'])): 
        raw.info['chs'][i]['coord_frame'] = 1           

    tmp_path = emptyroom._init_kwargs['pdf_fname'].split('c')[0]
    emptyroom._init_kwargs['config_fname'] = tmp_path+'config'                          # was processed similar to the HCP data
    raw.info['chs'][i]['coord_frame'] = 1

    trans_fname = f'{rawdir}/HCP_MNE_MEG/{subjid}/{subjid}-head_mri-trans.fif'
    trans = mne.read_trans(trans_fname)

    t1_path = f'{dest_fsdir}/mri/T1w_hires.nii.gz'
    t1w_bids_path = BIDSPath(subject=subjid, session='1', root=topdir, suffix='T1w')
    t1w_bids_path = write_anat(
                    image=t1_path,
                    bids_path=t1w_bids_path,
                    landmarks=None,
                    deface=False,  #Deface already done
                    overwrite=True
                    )

    trans_outpath = f'{topdir}/sub-{subjid}/ses-1/anat/sub-{subjid}-head_mri-trans.fif'
    mne.write_trans(trans_outpath, trans, overwrite=True)

    bids_path = BIDSPath(subject=subjid, session='1', task='rest',
                    run='01', root=topdir, suffix='meg')
    write_raw_bids(raw, bids_path, overwrite=True)

    eroom_bids_path = BIDSPath(subject=subjid, session='1', task='emptyroom',
                    run='01', root=topdir, suffix='meg')
    write_raw_bids(emptyroom, eroom_bids_path, overwrite=True)


                
 
