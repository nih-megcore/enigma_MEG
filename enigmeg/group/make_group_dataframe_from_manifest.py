#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 13:36:36 2024

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

def check_age(age):
    
    if age == '22-25':
        age_out = 23.5
    if age == '26-30':
        age_out = 28
    if age == '31-35':
        age_out = 33
    if age == '36-40':
        age_out = 38
    if age == '41-45':
        age_out = 43
    if age == '46-50':
        age_out = 48
    if age == '51-55':
        age_out = 53
    if age == '56-60':
        age_out = 58
    if age == '61-65':
        age_out = 63
    if age == '66-70':
        age_out = 68
    if age == '71-75':
        age_out = 73
    if age == '76-80':
        age_out = 78
 
    return age_out

def check_sex(sex):
    
    if (sex == 'f') | (sex == 'F') | (sex == 'female') | (sex == 'FEMALE'):
        sex_out = 'F'
    elif (sex == 'm') | (sex == 'M') | (sex == 'male') | (sex == 'MALE'):
        sex_out = 'M'
    else:
        sex_out = 'N'
        
    return sex_out

def check_task(task):
    
    if (task == 'resteyesclosed') | (task == 'eyesclosed') | (task == 'restEC') | (task == 'RestEC'):
        out_task = 'eyesclosed'
    elif (task == 'resteyesopen') | (task == 'eyesopen') | (task == 'restEO') | (task == 'RestEO') | (task == 'eyesNA'):
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
    
    if (group == 'Control') | (group == 'HV') | (group == 'HC'):
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

def parse_qafile(fname):
    
    allsubj_df = pd.DataFrame(columns=['subj','ses','run','task','QA'])
    
    with open(fname, 'r') as file:
        lines = file.readlines()
        
    for idx, line in enumerate(lines):
        
        parts = line.split('::')
        if parts[2].startswith('REVIEW'):
            continue
        
        imagefilename=parts[2].split(':')[1]
        entities=mne_bids.get_entities_from_fname(imagefilename)
        dfobject = munch.Munch()
        dfobject.subj = entities['subject']
        dfobject.ses = entities['session']
        dfobject.run = entities['run']
        dfobject.task = entities['task']
        dfobject.QA = parts[2].split(':')[5].split('\n')[0]
        dfobject = munch.unmunchify(dfobject)     
        subj_df = pd.DataFrame([dfobject])
    
        # concatenate the subject dataframe with the global datafram for all subjects
        allsubj_df = pd.concat([allsubj_df, subj_df])
        
    return allsubj_df

if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-bids_root', help='''bids root''')
    parser.add_argument('-group_root', help='''The name of the GROUP directory''')
    parser.add_argument('-group_name', help='''The name for the group''')
    parser.add_argument('-scanner', help='''The scanning platform''')

    args = parser.parse_args()
    bids_root = args.bids_root
    group_root = args.group_root
    group_name = args.group_name
    
    print(bids_root)
    
    logfile_root = op.join(bids_root,'derivatives/ENIGMA_MEG/logs/')
    print(logfile_root)
    
    if args.scanner not in ['megin','4d','kit','ctf']:
        print('Scanner must be megin, 4d, kit, ctf')
    scanner=args.scanner

    manifest = pd.read_csv(f'{bids_root}/manifest.txt',na_filter=False,dtype=str)
    manifest = manifest.replace({'None':None})

    # first, update the manifest to inidcate whether an output dataset was produced or not

    manifest.insert(len(manifest.columns), "power_fname",value='')
    manifest.insert(len(manifest.columns), "spectra_fname",value='')
    manifest.insert(len(manifest.columns), "logfile_fname",value='')

    print('Figuring out which MEG datasets in the manifest have output data')

    for index, row in manifest.iterrows():
        
        print(row)
        subject = row["participant_id"]
        subject_strip = subject.strip('sub-')
        session = row["ses"]
        run = row["run"]
        task = row["task"]
        print((subject, session, run, task))
        power_fname = []
        spectra_fname = []
    
        if session != None:
            if run != None:
                power_fname = glob.glob(f'{group_root}/{subject}*{session}*{task}*{run}*power.csv')
                spectra_fname = glob.glob(f'{group_root}/{subject}*{session}*{task}*{run}*spectra.csv')
                logfile_fname = f'{logfile_root}/{subject_strip}_ses-{session}_task-{task}_run-{run}_log.txt'
            else:
                power_fname = glob.glob(f'{group_root}/{subject}*{session}*{task}*power.csv')
                spectra_fname = glob.glob(f'{group_root}/{subject}*{session}*{task}*spectra.csv')
                logfile_fname = f'{logfile_root}/{subject_strip}_ses-{session}_task-{task}_run-None_log.txt'
        else:
            if run != None:
                power_fname = glob.glob(f'{group_root}/{subject}*{task}*{run}*power.csv')
                spectra_fname = glob.glob(f'{group_root}/{subject}*{task}*{run}*spectra.csv')
                logfile_fname = f'{logfile_root}/{subject_strip}_ses-None_task-{task}_run-{run}_log.txt'
            else:
                power_fname = glob.glob(f'{group_root}/{subject}*{task}*power.csv')
                spectra_fname = glob.glob(f'{group_root}/{subject}*{task}*spectra.csv')
                logfile_fname = f'{logfile_root}/{subject_strip}_ses-None_task-{task}_run-None_log.txt'
            
        if len(power_fname) == 0:
            manifest.at[index, 'power_fname'] = ''
        elif len(power_fname) == 1:
            manifest.at[index, 'power_fname'] = power_fname[0]
            manifest.at[index, 'spectra_fname'] = spectra_fname[0]
            manifest.at[index, 'logfile_fname'] = logfile_fname
        else:
            print("Problem - redundancy detected")

    # choose only rows that have existing output
        
    manifest_exists = manifest.drop(manifest[manifest['power_fname'] == ''].index)
 
    # Also, choose only rows that haven't been excluded in QA
    
    # first load in the QA logfiles we are using
    QA_root = f'{bids_root}/derivatives/ENIGMA_MEG_QA'
    coreg_qa = parse_qafile(op.join(QA_root,'coreg_QA_logfile.txt'))
    spectra_qa = parse_qafile(op.join(QA_root,'spectra_QA_logfile.txt'))
    alpha_qa = parse_qafile(op.join(QA_root,'alpha_QA_logfile.txt'))
    surf_qa = parse_qafile(op.join(QA_root,'surf_QA_logfile.txt'))
    if op.isfile(op.join(QA_root,'auxillary_QA_logfile.txt')):
        aux_qa = parse_qafile(op.join(QA_root,'auxillary_QA_logfile.txt'))
        aux_qa = aux_qa.rename(columns={'QA':'QA_X'})
    
    coreg_qa=coreg_qa.rename(columns={'QA':'QA_C'})
    spectra_qa=spectra_qa.rename(columns={'QA':'QA_P'})
    alpha_qa=alpha_qa.rename(columns={'QA':'QA_A'})
    surf_qa=surf_qa.rename(columns={'QA':'QA_S'})
    
    # merge 
    try: 
        full_qa = pd.merge(pd.merge(pd.merge(pd.merge(coreg_qa,spectra_qa, 
                    on=['subj','ses','run','task']), alpha_qa, 
                    on=['subj','ses','run','task']), surf_qa,
                    on=['subj','ses','run','task']), aux_qa,
                    on=['subj','ses','run','task'])
    except:
        full_qa = pd.merge(pd.merge(pd.merge(coreg_qa,spectra_qa, 
                    on=['subj','ses','run','task']), alpha_qa, 
                    on=['subj','ses','run','task']), surf_qa,
                    on=['subj','ses','run','task'])
    
    # compose a final good/bad rating. All must be good to be included
    try:
        full_qa['QA_FINAL'] = full_qa[['QA_C', 'QA_P', 'QA_A', 'QA_S', 'QA_X']].eq('GOOD').all(axis=1).map({True: 'GOOD', False: 'BAD'})
    except:
        full_qa['QA_FINAL'] = full_qa[['QA_C', 'QA_P', 'QA_A', 'QA_S']].eq('GOOD').all(axis=1).map({True: 'GOOD', False: 'BAD'})
    
    try:
        final_qa = full_qa.drop(columns=['QA_C', 'QA_P', 'QA_A', 'QA_S', 'QA_X'],inplace=False)
    except:
        final_qa = full_qa.drop(columns=['QA_C', 'QA_P', 'QA_A', 'QA_S'],inplace=False)
        
    final_qa['subj'] = 'sub-' + final_qa['subj'].astype(str)
    
    final_qa=final_qa.rename(columns={'subj':'participant_id'})
    manifest_exists=pd.merge(manifest_exists, final_qa, how='left', on=['participant_id','ses','run','task'])

    # Now, put all the csv files into one big giant csv file

    list_of_power_dframes = []
    list_of_spectra_dframes = []

    for index, row in manifest_exists.iterrows():
        
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
            
            if '-' in str(row['age']):
                subj_power_dframe['age'] = check_age(row['age'])
                subj_spectra_dframe['age'] = check_age(row['age'])
            else:
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
        
        subj_power_dframe['QA_FINAL'] = row['QA_FINAL']
        subj_spectra_dframe['QA_FINAL'] = row['QA_FINAL']        
            
        if (subj_power_dframe['group'][0] != 'EXCLUDE'):
            list_of_power_dframes.append(subj_power_dframe)
            list_of_spectra_dframes.append(subj_spectra_dframe) 
            
    group_power_dframe = pd.concat(list_of_power_dframes)
    group_spectra_dframe = pd.concat(list_of_spectra_dframes)
    
    group_power_dframe = group_power_dframe.rename(columns={'Unnamed: 0':'Parcel'})
    group_spectra_dframe = group_spectra_dframe.rename(columns={'Unnamed: 0':'Parcel'})
        
    group_power_dframe.to_csv(f'{group_name}_Power_dataframe.csv')
    group_spectra_dframe.to_csv(f'{group_name}_Spectra_dataframe.csv')
