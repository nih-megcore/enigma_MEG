#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 14:39:25 2023

@author: Allison Nugent and Jeff Stout
"""

import os, os.path as op
import argparse
import enigmeg
from enigmeg.process_meg import process
from enigmeg.QA.enigma_QA_functions import gen_coreg_pngs, gen_bem_pngs, gen_src_pngs, gen_surf_pngs
from enigmeg.QA.enigma_QA_functions import gen_epo_pngs, gen_fooof_pngs
import sys
import pandas as pd
import numpy as np
import logging
import mne
from mne_bids import BIDSPath
import matplotlib.pyplot as plt
from enigmeg.process_meg import load_data

def hcp_load_data(proc_subj):
    
        if not hasattr(proc_subj, 'raw_rest'):
            
            filename = str(proc_subj.meg_rest_raw.fpath) + '/c,rf0.0Hz'
            proc_subj.raw_rest = mne.io.read_raw_bti(filename,
                                    head_shape_fname=None,convert=False,preload=True)
            for i in range(len(proc_subj.raw_rest.info['chs'])):                      # was processed similar to the HCP data
                        proc_subj.raw_rest.info['chs'][i]['coord_frame'] = 1
            proc_subj.raw_rest.pick_types(meg=True, eeg=False)
            
        if (not hasattr(proc_subj, 'raw_eroom')) and (proc_subj.meg_er_raw != None):
            
            filename_er = str(proc_subj.meg_er_raw.fpath)+ '/c,rf0.0Hz'
            proc_subj.raw_eroom = mne.io.read_raw_bti(filename_er,
                                    head_shape_fname=None,convert=False,preload=True)
            for i in range(len(proc_subj.raw_eroom.info['chs'])):                      # was processed similar to the HCP data
                        proc_subj.raw_eroom.info['chs'][i]['coord_frame'] = 1
            proc_subj.raw_eroom.pick_types(meg=True, eeg=False)

        # For subsequent reference, if raw_room not provided, set to None
        if (not hasattr(proc_subj, 'raw_eroom')):
            proc_subj.raw_eroom=None
        # figure out the MEG system vendor, note that this may be different from 
        # datatype if the datatype is .fif
        proc_subj.vendor = mne.channels.channels._get_meg_system(proc_subj.raw_rest.info)

def gen_coreg_pngs_hcp(subjstruct):
    
    from mne.viz._brain.view import views_dicts
    from mne.viz import set_3d_view
    
    subjid = subjstruct.subject
    
    print(subjstruct.meg_rest_raw.fpath)
    filename = str(subjstruct.meg_rest_raw.fpath) + '/c,rf0.0Hz'
    subjstruct.raw_rest = mne.io.read_raw_bti(filename,
                            head_shape_fname=None,convert=False,preload=True)
    for i in range(len(subjstruct.raw_rest.info['chs'])):                      # was processed similar to the HCP data
                subjstruct.raw_rest.info['chs'][i]['coord_frame'] = 1
   
    subjstruct.trans = mne.read_trans(subjstruct.fnames['rest_trans'])
   
    fig = mne.viz.plot_alignment(info=subjstruct.raw_rest.info, trans=subjstruct.trans, subject='sub-'+subjid, 
                                 subjects_dir=subjstruct.subjects_dir,meg=('sensors'))
    set_3d_view(fig,**views_dicts['both']['rostral'])
    img1=fig.plotter.screenshot()
    fig.plotter.close()
    fig = mne.viz.plot_alignment(info=subjstruct.raw_rest.info, trans=subjstruct.trans, subject='sub-'+subjid, 
                                 subjects_dir=subjstruct.subjects_dir,meg=('sensors'))
    set_3d_view(fig,**views_dicts['both']['lateral'])
    img2=fig.plotter.screenshot()
    fig.plotter.close()
    fig = mne.viz.plot_alignment(info=subjstruct.raw_rest.info, trans=subjstruct.trans, subject='sub-'+subjid, 
               subjects_dir=subjstruct.subjects_dir,meg=('sensors'))
   
    set_3d_view(fig,**views_dicts['both']['medial'])
    img3=fig.plotter.screenshot()
    fig.plotter.close()
    fig, ax = plt.subplots(1,3)
    ax[0].imshow(img1)
    tmp=ax[0].axis('off')
    ax[1].imshow(img2)
    tmp=ax[1].axis('off')
    ax[2].imshow(img3)
    tmp=ax[2].axis('off')
   
    figname_basename = subjstruct.deriv_path.update(
        root=subjstruct.bids_root,
        task = subjstruct.meg_rest_raw.task,
        datatype = 'meg',
        subject=subjstruct.subject,
        session=subjstruct.meg_rest_raw.session,
        run=subjstruct.meg_rest_raw.run,
        suffix = 'coreg',
        extension='.png'
        ).basename

    figname = op.join(subjstruct.QA_dir.directory, figname_basename)
    
    fig.savefig(figname, dpi=300,bbox_inches='tight')
    plt.close(fig)

def _prepare_QA(subjstruct):
    
    gen_coreg_pngs_hcp(subjstruct)

    gen_bem_pngs(subjstruct)

    gen_src_pngs(subjstruct)

    gen_surf_pngs(subjstruct)
    
    gen_epo_pngs(subjstruct)
    
    gen_fooof_pngs(subjstruct)

def main():
    # parse the arguments and initialize variables   

    parser = argparse.ArgumentParser()
    parser.add_argument('-bids_root', help='''BIDS root directory''')
    parser.add_argument('-subjid', help='''Define the subject id to process''')
    parser.add_argument('-session', help='''Session number''', default=None)
    parser.add_argument('-run', help='''Run number, note that 01 is different from 1''', default='1')
    parser.add_argument('-rest_tag', help= '''override if rest task name is other than rest''', default='rest')
    parser.add_argument('-emptyroom_tag', help= '''override if emptyroom task name is other than empty''', default='empty')
    parser.add_argument('-proc_from_csv', help='''Loop over all subjects in a .csv file''', default=None)
    parser.description='''This python script will compile a series of QA images for assessment of the enigma_MEG pipeline'''
    
    args = parser.parse_args()
    
    if len(sys.argv) == 1:
        parser.print_help()
        parser.exit(1)   
    
    if not args.bids_root:
        bids_root = 'bids_out'
    else:
        bids_root=args.bids_root
    
    if args.emptyroom_tag:
            if args.emptyroom_tag.lower()=='none': args.emptyroom_tag=None
    
    if not op.exists(bids_root):
        raise ValueError('Please specify a correct -bids_root')
    
    derivatives_dir = op.join(bids_root, 'derivatives')
    
    enigma_root = op.join(derivatives_dir, 'ENIGMA_MEG')          
    if not op.exists(enigma_root):
        raise ValueError('No ENIGMA_MEG directory - did you run process_meg.py?')
                
    subjects_dir = op.join(derivatives_dir,'freesurfer/subjects')  
    
    # Setup logging if there are errors
    logging.basicConfig(filename=op.join(enigma_root+'_QA', 'QA_logfile.txt'),
                        level=logging.WARNING, format='%(asctime)s :: %(message)s')
        
    # process a single subject
    
    if args.subjid:    
            
        args.subjid=args.subjid.replace('sub-','')
        subjid=args.subjid
        print(args.subjid)
                
        if args.proc_from_csv != None:
            raise ValueError("You can't specify both a subject id and a csv file, sorry")    

        subjstruct = process(subject=subjid, 
                        bids_root=bids_root, 
                        deriv_root=derivatives_dir,
                        subjects_dir=subjects_dir,
                        rest_tagname=args.rest_tag,
                        emptyroom_tagname=args.emptyroom_tag, 
                        session=args.session, 
                        mains=0,      
                        run=args.run,
                        t1_override=None,
                        fs_ave_fids=False
                        )
        _prepare_QA(subjstruct)
        
        rogue_bidspath = subjstruct.deriv_path.copy().update(extension=None)
        rogue_dir = op.join(rogue_bidspath.directory, rogue_bidspath.basename)
        if os.path.isdir(rogue_dir):
            if len(os.listdir(rogue_dir))==0: # make sure directory is empty
                os.rmdir(rogue_dir) # this won't work unless the irectory is empty
    
    elif args.proc_from_csv:
        
        logging.info('processing subject list from %s' % args.proc_from_csv)
        
        dframe = pd.read_csv(args.proc_from_csv, dtype={'sub':str, 'ses':str, 'run':str})
        dframe = dframe.astype(object).replace(np.nan,None)
        
        for idx, row in dframe.iterrows():  # iterate over each row in the .csv file
            
            print(row)
            
            subjid=row['sub']
            if pd.isna(row['ses']):
                session=None
            else:
                session=str(row['ses'])
            if pd.isna(row['run']):
                run=None
            else:
                run=str(row['run'])
            args.rest_tag = row['path'].split('task-')[1].split('_')[0]
            try:
                args.emptyroom_tag = row['eroom'].split('task-')[1].split('_')[0]
            except:
                args.emptyroom_tag = None
            
            try:
                subjstruct = process(subject=subjid, 
                            bids_root=bids_root, 
                            deriv_root=derivatives_dir,
                            subjects_dir=subjects_dir,
                            rest_tagname=args.rest_tag,
                            emptyroom_tagname=args.emptyroom_tag,  
                            session=session, 
                            mains=0,      
                            run=run,
                            t1_override=None,
                            fs_ave_fids=False
                            )
    
                _prepare_QA(subjstruct)
    
                rogue_bidspath = subjstruct.deriv_path.copy().update(extension=None)
                rogue_dir = op.join(rogue_bidspath.directory, rogue_bidspath.basename)
                if os.path.isdir(rogue_dir):
                    if len(os.listdir(rogue_dir))==0: # make sure directory is empty
                        os.rmdir(rogue_dir) # this won't work unless the irectory is empty
            except BaseException as e:
                logging.warning(f':: {subjid} :: {str(e)}')

    
if __name__=='__main__':
    main()

                
    
    
