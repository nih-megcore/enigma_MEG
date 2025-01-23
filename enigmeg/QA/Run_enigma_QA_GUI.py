#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 22 09:15:11 2023

@author: Allison Nugent and Jeff Stout
"""

import os.path as op
import sys
import argparse
import glob
from enigmeg.QA.enigma_QA_GUI_functions import initialize, sub_qa_info, get_last_review, build_status_dict, run_gui



PROJECT = 'ENIGMA_MEG_QA'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-bids_root', help='''Location of bids directory, default=bids_out''')
    parser.add_argument('-QAtype', help='''QA type to run. Options are 'coreg', 'ica', 'bem','src','surf','spectra','alpha','sourcepsd''')
    parser.add_argument('-rows', help='''number of rows in QA browser, default=4''')
    parser.add_argument('-columns', help='''number of columns in QA browser, default=2''')
    parser.add_argument('-imgsize', help='''make images smaller or larger, default=200''')
    
    args = parser.parse_args()
    
    if not args.bids_root:
        bids_root='bids_out'
    else:
        bids_root=args.bids_root
    if not op.exists(bids_root):
        parser.print_help()
        raise ValueError('specified BIDS directory %s does not exist' % bids_root)
        
    deriv_root = op.join(bids_root, 'derivatives') 
    QA_root = op.join(deriv_root,PROJECT)
    if not op.exists(bids_root):
        parser.print_help()
        raise ValueError('No ENIGMA_MEG_QA directory. Did you run prepare_QA?')
    
    if not args.QAtype:
        parser.print_help()
        raise ValueError('You must choose a QA type')
    else:
        if args.QAtype not in ['coreg','ica','surf','bem','src','spectra','alpha','cleaned','sourcepsd']:
            raise ValueError("QAtype not valid, must be one of 'coreg', 'ica','surf','bem','src','spectra','alpha','cleaned','sourcepsd'")
        QAtype = args.QAtype
    
    if QAtype == 'coreg' or QAtype == 'bem' or QAtype == 'src' or QAtype == 'ica':
        column_default = 3
        row_default = 4
        size_default = 400
    elif QAtype == 'surf':
        column_default = 2
        row_default = 3
        size_default = 300
    elif QAtype == 'spectra' or QAtype == 'alpha' or QAtype == 'cleaned' or QAtype == 'sourcepsd':
        column_default = 3
        row_default = 3
        size_default = 300   
        
    if not args.imgsize:
        imgsize=size_default
    else:
        imgsize=int(args.imgsize)
    if not args.rows:
        rows=row_default
    else:
        rows=int(args.rows)    
    if not args.columns:
        columns=column_default
    else:
        columns=int(args.columns)

    deriv_root = op.join(bids_root, 'derivatives') 
    QA_root = op.join(deriv_root,PROJECT)
    subjects_dir = op.join(bids_root,'derivatives/freesurfer/subjects')
    
    history_log = initialize(bids_root, QAtype)        
        
    # search for QA images matching the requested QAtype

    image_list = glob.glob(op.join(QA_root,'*/*/*/*' + QAtype + '*.png'))
    if len(image_list) == 0:
        image_list = glob.glob(op.join(QA_root,'*/*/*' + QAtype + '*.png'))
    sub_obj_list = [sub_qa_info(i, fname) for i,fname in enumerate(image_list)]
    
    #Update status based on previous log
    if history_log is not None:
        last_review = get_last_review(history_log) 
        stat_dict = build_status_dict(last_review)
        for sub_qa in sub_obj_list:
            if sub_qa.subject in stat_dict.keys():
                sub_qa.set_status(stat_dict[sub_qa.subject])
    else: # if there is  no previous log, initialize the type of QA for each subject            
        for sub_qa in sub_obj_list:
            sub_qa.qa_type=QAtype
    
    run_gui(sub_obj_list,rows,columns,imgsize,QAtype)          
    
    
if __name__ == '__main__':
    main()
    
          
                           