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




def _prepare_QA(subjstruct):
    
    gen_coreg_pngs(subjstruct)

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

                
    
    
