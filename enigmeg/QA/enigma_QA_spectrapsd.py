#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 13:42:44 2025

@author: nugenta
"""

import argparse
import mne_bids
import pandas as pd
import matplotlib.pyplot as plt
import glob

def main():
    # parse the arguments and initialize variables   

    parser = argparse.ArgumentParser()
    parser.add_argument('-bids_root', help='''BIDS root directory''')
    parser.description='''This python script will compile spectra images from beamformed data'''
    
    args = parser.parse_args()
    
    bids_root=args.bids_root
    print(bids_root)
    QA_root = f'{bids_root}/derivatives/ENIGMA_MEG_QA'
    print(QA_root)
    spectra_files = glob.glob(f'{bids_root}/derivatives/ENIGMA_MEG/*/*/*/*spectra.csv')
    if len(spectra_files) == 0:
        spectra_files = glob.glob(f'{bids_root}/derivatives/ENIGMA_MEG/*/*/*spectra.csv')

    print(len(spectra_files))
    for file in spectra_files:
        
        entities = mne_bids.get_entities_from_fname(file)
        print('working on subject %s, session %s, run %s, task %s' %
              (entities['subject'],entities['session'],entities['run'],entities['task']))

        
        if entities["session"]== None:
            QA_dir = f'{QA_root}/sub-{entities["subject"]}/meg'
            if entities["run"] == None:
                QA_file = f'{QA_dir}/sub-{entities["subject"]}_task-{entities["task"]}_sourcepsd.png'
            else:
                QA_file = f'{QA_dir}/sub-{entities["subject"]}_task-{entities["task"]}_run-{entities["run"]}_sourcepsd.png'
        else:
            QA_dir = f'{QA_root}/sub-{entities["subject"]}/ses-{entities["session"]}/meg/'
            if entities["run"] == None:
                QA_file = f'{QA_dir}/sub-{entities["subject"]}_ses-{entities["session"]}_task-{entities["task"]}_sourcepsd.png'
            else:
                QA_file = f'{QA_dir}/sub-{entities["subject"]}_ses-{entities["session"]}_task-{entities["task"]}_run-{entities["run"]}_sourcepsd.png'

        df = pd.read_csv(file)
        # drop the "unknown-lh" and "unknown-rh" parcels
        df = df.drop([225,449])
        x=df.columns.astype(float).values
        
        fig, ax = plt.subplots()
        ax.plot(x, df.values.T)
        
        fig.savefig(QA_file, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
if __name__=='__main__':
    main()