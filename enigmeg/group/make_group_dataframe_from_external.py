#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 19:37:16 2025

@author: nugenta
"""
import numpy as np
import pandas as pd
import mne
import mne_bids
import argparse
import glob
from os import path as op
from datetime import datetime
import re
import munch

def check_sex(sex):
    
    if (sex == 'f') | (sex == 'F') | (sex == 'female') | (sex == 'FEMALE'):
        sex_out = 'F'
    elif (sex == 'm') | (sex == 'M') | (sex == 'male') | (sex == 'MALE'):
        sex_out = 'M'
    else:
        sex_out = 'N'
        
    return sex_out

def check_task(task):
    
    if (task == 'resteyesclosed') | (task == 'eyesclosed') | (task == 'restEC') | (task == 'RestEC') | ('EC' in task):
        out_task = 'eyesclosed'
    elif (task == 'resteyesopen') | (task == 'eyesopen') | (task == 'restEO') | (task == 'RestEO') | ('EO' in task) | (task == 'ArcaraResting'):
        out_task = 'eyesopen'
    else:
        out_task = task
        
    return out_task

def check_hand(hand):
    
    if (hand == 'r') | (hand == 'right') | (hand == 'R') | (hand == 'RIGHT'):
        out_hand = 'R'
    elif (hand == 'l') | (hand == 'left') | (hand == 'L') | (hand == 'LEFT'):
        out_hand = 'L'
    else:
        out_hand = 'N'
        
    return out_hand

def check_group(group):
    
    if (group == 'Control') | (group == 'HV') | (group == 'HC') | (group == 'healthy'):
        out_group = 'Control'
    else:
        out_group = group

    return out_group    

def parse_line(line, next_lines):
    
    info={}
    components_to_reject = None
    parts = line.split(' :: ')
    if parts[2].startswith('do_classify_ics'):
        components_to_reject = parts[-1].strip().split(': ')[1][1:-1].split(', ')
    if len(parts) > 3:
        if parts[2].startswith('_proc_epochs') & (parts[3].startswith('START')):
            part_line0 = next_lines[0].split(' :: ')
            part_line1 = next_lines[1].split(' :: ')
            part_line2 = next_lines[2].split(' :: ')
            total_time = float(part_line2[-1].split(': ')[1].strip('\n'))
            final_epochs = int(part_line1[-1].split(': ')[1].strip('\n'))
            original_epochs = int(part_line0[-1].split(': ')[1].strip('\n'))
            info['epochs'] = {'original': original_epochs,
                          'final' : final_epochs,
                          'total_time' : total_time }
    elif parts[2].startswith('eTIV'):
        pattern = r"eTIV: (\d+) lh_holes: (\d+) rh_holes: (\d+) avg_holes: (\d+\.\d+)"
        match = re.search(pattern, line)
        info['segstats'] = {'eTIV': match.group(1),
                            'lh_holes': match.group(2),
                            'rh_holes': match.group(3),
                            'avg_holes': match.group(4) }
    
    return components_to_reject, info

def parse_logfile(fname):
    
    with open(fname, 'r') as file:
        lines = file.readlines()
    
    recent_data = []
    
    for idx, line in enumerate(lines):
        
        parts = line.split(' :: ') 
        if len(parts) < 2:
            continue
        if len(parts) > 3:
            if parts[2].startswith('do_mri_segstats') & parts[3].startswith('COMPLETED'):
                break
        next_lines = lines[idx+1:idx+4]
        
        components_to_reject, info = parse_line(line, next_lines)        

    # parse the arguments and initialize variable

        if (components_to_reject != None):
            recent_data.append(components_to_reject)
        if (info != {}):
            recent_data.append(info)
    
    return recent_data

def standardize_sub(sub):
    print(sub)
    return 'sub-'+ sub if not sub.startswith('sub-') else sub

def dframe_tomanifest(scan_dframe, bids_root=None, outfile='manifext.txt'):
    
    fname = f'{bids_root}/{outfile}'
    participants_df = pd.read_csv(f'{bids_root}/participants.tsv', sep="\t" )
    participants_df['sub_standardized'] = participants_df['participant_id'].apply(standardize_sub)
    scan_dframe['sub_standardized'] = scan_dframe['sub'].apply(standardize_sub)
    participants_df = participants_df.drop(['participant_id'], axis=1)
    
    merged_df = pd.merge(participants_df, scan_dframe, left_on='sub_standardized',right_on='sub_standardized')
    merged_df = merged_df.rename(columns={'sub_standardized':'participant_id'})
    print(merged_df)
    
    selected_columns = ['participant_id','task','ses','run']
    for field in ['age', 'sex', 'hand','group','Diagnosis_TP1', 'Diagnosis_TP2','Diagnosis','diagnosis']:
        if field in participants_df:
            selected_columns.append(field)
            
    print(selected_columns)
    
    duplicate_columns = merged_df.columns[merged_df.columns.duplicated()]
    # If there are duplicate columns, drop one of them
    if len(duplicate_columns) > 0:
        merged_df = merged_df.drop(columns=duplicate_columns[0])
    
    manifest_tofile = merged_df[selected_columns]

    manifest_tofile.to_csv(fname)
    
    return merged_df


if __name__=='__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-root', help='''bids root''')
    parser.add_argument('-scanner', help='''The scanning platform''')
    parser.add_argument('-group_name', help='''the GROUP name for the output file''')

    args = parser.parse_args()
    root = args.root
    group_name = args.group_name
    
    logfile_root = op.join(root,'logs')

    if args.scanner not in ['megin','4d','kit','ctf']:
        print('Scanner must be megin, 4d, kit, ctf')
    scanner=args.scanner

    # make a dataframe from all the available output files
    
    power_fnames = []
    df_list = []
    
    power_fnames = glob.glob(f'{root}/*power.csv')
    
    for fname in power_fnames:
        
        ents = mne_bids.get_entities_from_fname(fname)
        sub = ents['subject']
        task = ents['task']
        ses = ents['session']
        run = ents['run']

        df_obj = munch.Munch()
        df_obj.sub = sub
        df_obj.task = task
        df_obj.ses = ses
        df_obj.run = run
        df_obj.power_fname = fname
        df_obj.spectra_fname = f'{root}/sub-{sub}_ses-{ses}_meg_label_task-{task}_run-{run}_spectra.csv'
        df_obj.logfile_fname = f'{root}/logs/{sub}_ses-{ses}_task-{task}_run-{run}_log.txt'
        df_list.append(df_obj)

    df_list = munch.unmunchify(df_list)     
    df = pd.DataFrame(df_list)
    
    manifest = dframe_tomanifest(df, bids_root=root, outfile='manifext.txt')
    
    manifest = manifest.replace({'None':None})

    # Now, put all the csv files into one big giant csv file

    list_of_power_dframes = []
    list_of_spectra_dframes = []

    for index, row in manifest.iterrows():
        
        print('Working on subject:')
        
        print(row['participant_id'])
        subjid = row['participant_id']
        
        # parse the log file for this subject to add additional items to the dataframe
        
        log_data = parse_logfile(row['logfile_fname'])
        
        # check to see if there are one or two listings for proc_epoch 
        # if there are two, it means one is from an empty room
        
        first_epochs_dict = None
        second_epochs_dict = None
        segstat_dict = None
        for item in log_data:
            if 'epochs' in item:
                if first_epochs_dict is None:
                    first_epochs_dict = item['epochs']
                else:
                    second_epochs_dict = item['epochs']
            if 'segstats' in item:
                segstat_dict = item['segstats']
        
        # read in the .csv file for each subject
        
        subj_power_dframe = pd.read_csv(row['power_fname'],sep='\t')
        subj_power_dframe['subject'] = f'{group_name}_{subjid}'
        
        subj_spectra_dframe = pd.read_csv(row['spectra_fname'],sep='\t')
        subj_spectra_dframe['subject'] = f'{group_name}_{subjid}'
        
        # drop the corpus callosum parcel
        
        subj_power_dframe = subj_power_dframe.rename(columns={'Unnamed: 0':'Parcel'})
        subj_spectra_dframe['Parcel'] = subj_power_dframe['Parcel']
        
        subj_power_dframe = subj_power_dframe[subj_power_dframe['Parcel'] != 'unknown-lh']
        subj_power_dframe = subj_power_dframe[subj_power_dframe['Parcel'] != 'unknown-rh']
        subj_spectra_dframe = subj_spectra_dframe[subj_spectra_dframe['Parcel'] != 'unknown-lh']
        subj_spectra_dframe = subj_spectra_dframe[subj_spectra_dframe['Parcel'] != 'unknown-rh']
        
        # check ages and replace ranges
        
        if 'age' in row:
            subj_power_dframe['age'] = row['age']
            subj_spectra_dframe['age'] = row['age']
         
        # check sex
        
        if 'sex' in row:
            
            subj_power_dframe['sex'] = check_sex(row['sex'])
            subj_spectra_dframe['sex'] = check_sex(row['sex'])
                  
        # check handedness
        
        if 'hand' in row:
            subj_power_dframe['hand'] = check_hand(row['hand'])
            subj_spectra_dframe['hand'] =check_hand(row['hand'])
        else:
            subj_power_dframe['hand'] = 'N'
            subj_spectra_dframe['hand'] = 'N'
            
        # check task
            
        subj_power_dframe['task'] = check_task(row['task'])
        subj_spectra_dframe['task'] = check_task(row['task'])
        
        # check group
        
        if 'group' in row:
            subj_power_dframe['group'] = check_group(row['group'])
            subj_spectra_dframe['group'] = check_group(row['group'])                     
        elif 'Diagnosis_TP1' in row:
            if 'control' in row['Diagnosis_TP1']:
                subj_power_dframe['group'] = 'Control'
                subj_spectra_dframe['group'] = 'Control'
            else:
                subj_power_dframe['group'] = row['Diagnosis_TP1']
                subj_spectra_dframe['group'] = row['Diagnosis_TP1']
        elif 'Diagnosis' in row:
            subj_power_dframe['group'] = check_group(row['Diagnosis'])
            subj_spectra_dframe['group'] = check_group(row['Diagnosis'])                     
        elif 'diagnosis' in row:
            subj_power_dframe['group'] = check_group(row['diagnosis'])
            subj_spectra_dframe['group'] = check_group(row['diagnosis'])                     
        else:
            subj_power_dframe['group'] = 'Control'
            subj_spectra_dframe['group'] = 'Control'

        subj_power_dframe['ses'] = row['ses']
        subj_power_dframe['run'] = row['run']        
        subj_power_dframe['scanner'] = scanner
        
        subj_spectra_dframe['ses'] = row['ses']
        subj_spectra_dframe['run'] = row['run']        
        subj_spectra_dframe['scanner'] = scanner         
           
        # populate data from logfile
        
        subj_power_dframe['eTIV'] = segstat_dict['eTIV']
        subj_power_dframe['lh_holes'] = segstat_dict['lh_holes']
        subj_power_dframe['rh_holes'] = segstat_dict['rh_holes']
        subj_power_dframe['avg_holes'] = segstat_dict['avg_holes']
        
        subj_spectra_dframe['eTIV'] = segstat_dict['eTIV']
        subj_spectra_dframe['lh_holes'] = segstat_dict['lh_holes']
        subj_spectra_dframe['rh_holes'] = segstat_dict['rh_holes']
        subj_spectra_dframe['avg_holes'] = segstat_dict['avg_holes']    
        
        subj_power_dframe['epochs_orig'] = first_epochs_dict['original']
        subj_power_dframe['epochs_final'] = first_epochs_dict['final']
        subj_power_dframe['total_time'] = first_epochs_dict['total_time']
        subj_spectra_dframe['epochs_orig'] = first_epochs_dict['original']
        subj_spectra_dframe['epochs_final'] = first_epochs_dict['final']
        subj_spectra_dframe['total_time'] = first_epochs_dict['total_time']
        
        subj_power_dframe['QA_FINAL'] = 'GOOD'
        subj_spectra_dframe['QA_FINAL'] = 'GOOD'
            
        if (subj_power_dframe['group'][0] != 'EXCLUDE'):
            list_of_power_dframes.append(subj_power_dframe)
            list_of_spectra_dframes.append(subj_spectra_dframe) 
            
    group_power_dframe = pd.concat(list_of_power_dframes)
    group_spectra_dframe = pd.concat(list_of_spectra_dframes)
    
    group_power_dframe = group_power_dframe.rename(columns={'Unnamed: 0':'Parcel'})
    group_spectra_dframe = group_spectra_dframe.rename(columns={'Unnamed: 0':'Parcel'})
        
    group_power_dframe.to_csv(f'{group_name}_Power_dataframe.csv')
    group_spectra_dframe.to_csv(f'{group_name}_Spectra_dataframe.csv')


