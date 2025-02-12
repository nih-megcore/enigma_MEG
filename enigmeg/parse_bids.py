#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 17:28:12 2023

@author: nugenta
"""

import os, os.path as op
import re
import sys
import mne_bids
from mne_bids import BIDSPath
import glob
import munch
import pandas as pd
import argparse
import shutil

def get_subdirectories(path: str, stem: str) -> list:
    """Get a list of subdirectorty names that start with the requested stem; return strings without stem"""
    dirs = []
    for item in os.listdir(path):
        if os.path.isdir(os.path.join(path, item)) and item.startswith(stem):
            dirs.append(item[len(stem):])
    return dirs

def get_runnumbers(path: str, stem: str) -> list:
    """ Get a list of run numbers for a given 'task-stem' string"""
    run_numbers = set()
    searchpath = f'{path}/*{stem}*_meg*'
    for file in glob.glob(searchpath):
        tmpdict = mne_bids.get_entities_from_fname(file)
        if tmpdict['run'] == None:
            run_numbers.add('None')
            print('adding none')
        else:
            run_numbers.add(tmpdict['run'])
            print('adding runnumber')
    return list(run_numbers)

def assemble_cmd(row, bids_root=None):
    '''
    Use the pandas series to generate the process_meg command

    Parameters
    ----------
    row : pd.Series
        Row from Dataframe
    bids_root : str
        DESCRIPTION. The default is None.

    Returns
    -------
    cmd_ : str
        Line in the swarmfile command.

    '''
    cmd_ = f'process_meg.py -bids_root {bids_root}'
    tag_dict = {
        'sub':'-subject',
        'ses':'-session',
        'run':'-run',
        'emptyroom_run':'-emptyroom_run',
        'rest_tag':'-rest_tag',
        'eroom_tag':'-emptyroom_tag',
                }
    
    if row['eroom'] != None :
        row['eroom_tag']=op.basename(row.eroom).split('_task')[1].split('_')[0][1:]
    else:
        row['eroom_tag']=None

    if len(row['path']) != 0 and (row['type'] != 'c,rf'): 
        row['rest_tag']=op.basename(row.path).split('_task')[1].split('_')[0][1:]
    elif len(row['path']) != 0 and (row['type'] == 'c,rf'):
        row['rest_tag']=op.basename(op.dirname(row.path)).split('_task')[1].split('_')[0][1:]
    else:
        raise ValueError('Could not find MEG rest dataset')
    
    for row_tag, cmd_flag in tag_dict.items():
        if (row[row_tag]==None) or (len(row[row_tag])==0):
            cmd_ += f' {cmd_flag} None'
        else:
            cmd_ += f' {cmd_flag} {row[row_tag]}'
    cmd_ += ' -n_jobs 6'
    cmd_ += '\n'
    return cmd_

def dframe_toswarm(dframe, bids_root=None, outfile='enigma_swarm.sh'):
    "Convert the dataframe to swarm file"
    swarm = []
    for idx, row in dframe.iterrows():
        print(idx, row)
        swarm.append(assemble_cmd(row, bids_root=bids_root))
    with open(outfile, 'w') as f:
        f.writelines(swarm)

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
    
    result_df = merged_df[selected_columns]

    result_df.to_csv(fname)

def main():
    # parse the arguments and initialize variables   

    parser = argparse.ArgumentParser()
    parser.add_argument('-bids_root', help='''The name of the BIDS directory to be parsed''')
    parser.add_argument('-rest_tag',help='''The filename stem to find rest datasets''')
    parser.add_argument('-emptyroom_tag',help='''The filename stem to find emptyroom datasets''')
    parser.add_argument('-swarmfile', 
                        help='''For internal use at NIH.  Write a swarmfile for biowulf''',
                        default=False, action='store_true')
    parser.add_argument('-makemanifest', 
                        help='''For internal use at NIH.  Make a manifest of scans''',
                        default=False, action='store_true')
    parser.add_argument('-make_link', 
                        help='''If the MRI is in a separate session from the MEG, create a symbolic link''',
                        default=False, action='store_true')
    parser.description='''This python script parses a BIDS directory into a CSV file with one line per MEG to process'''
    
    
    args = parser.parse_args()
    
    if len(sys.argv) == 1:
        parser.print_help()
        parser.exit(1)            
    
    if not args.bids_root:
        bids_root = 'bids_out'
    else:
        bids_root = args.bids_root
        
    if not os.path.exists(bids_root):
        raise ValueError('BIDS root directory %s does not exist' % bids_root)
                   
    if args.rest_tag:
        rest_tag = args.rest_tag
    else:
        rest_tag = 'rest'
        
    if args.emptyroom_tag:
        emptyroom_tag = args.emptyroom_tag
    else:
        emptyroom_tag = 'emptyroom'

    # define list of MEG file format types to search for       
    type_list = ['.fif','.ds','4d','.con','.sqd','c,rf']

    # initialize the BIDSPath object
    bids_path = BIDSPath(root=bids_root)

    # compile subject list and set up a dataframe to hold subject data
    print('Compiling subject list. This may take time with lots of subjects')
    #subject_list = mne_bids.get_entity_vals(bids_root, 'subject')
    
    subject_list = get_subdirectories(bids_root, 'sub-')
    
    allsubj_df = pd.DataFrame(columns=['sub','ses','run','type','path','mripath','eroom','emptyroom_run'], 
                              dtype=str)

    for subject in subject_list:
    
        print('Working on subject %s' % subject)
        subj_dir = os.path.join(bids_root,'sub-' + subject)
    
        # initialize subject lists
        subrestlist = []
        mrilist = []
    
        rest_count = 0
        mri_count = 0
        
        # to exclude all the other subjects when searching for datasets from this subject, we'll
        # need to have a list of all other subjects besides the present one
        #other_subjects = list(set(subject_list) - set([subject]))

        # list of sessions for this subject
        #session_list = mne_bids.get_entity_vals(bids_root, 'session', ignore_subjects = other_subjects)
        
        session_list = get_subdirectories(subj_dir, 'ses-')
        
        if session_list == []:
            print("No ses- directories, will try going straight to meg/anat directories")
            session_list = ['None']

        # first, we are going to make a list of all available anatomical MRI scans        
        print('%d sessions for subject %s' % (len(session_list),subject))
        
        for session in session_list:
            if (session != 'None'):
                anatpath = bids_root + '/sub-' + subject + '/ses-' + session + '/anat/'   
            elif (session == 'None'):
                anatpath = bids_root + '/sub-' + subject + '/anat/'
            
            mrifiles = glob.glob(f'{anatpath}*T1w.nii*')    
                
            if len(mrifiles) > 0:   # if mri files found in current session, add to mri object list
                
                mri_count = mri_count+1
                mri_object = munch.Munch()
                mri_object.ses = session
                mri_object.path = mrifiles[0]
                mrilist.append(mri_object)
           
            if len(mrifiles) > 1:   # if there's more than one MRI in a session, just pick the first one
                print("More than one MRI in session %s, picking first one" % session)
                
        print("Identified %d T1w MRI scans for subject %s" % (len(mrilist), subject))
                
        # now, we are going to make a list of all the resting state MEG scans
        
        for session in session_list:   
            
            subrestlist_ses = []
            eroom_run_list = []
            
            if (session != 'None'):
                session_dir = os.path.join(subj_dir,'ses-' + session)
                megpath = bids_root + '/sub-' + subject + '/ses-' + session + '/meg'
            elif (session == 'None'):
                megpath = bids_root + '/sub-' + subject + '/meg'
    
            # is there an MEG in the session?
            if(os.path.exists(megpath)):
                print('found a megpath')
                print(megpath)
        
                for dattype in type_list:
                    
                    # are there any rest files for this datatype?
                    if dattype == 'c,rf':
                        restfiles = glob.glob(f'{megpath}/*{rest_tag}*/*{dattype}*')
                    else:
                        restfiles = glob.glob(f'{megpath}/*{rest_tag}*{dattype}')
                    if len(restfiles) > 0:
                    
                        # look for emptyroom datasets in the same session
                        if dattype == 'c,rf':
                            emptyfiles = glob.glob(f'{megpath}/*{emptyroom_tag}*/*{dattype}*')
                        else:
                            emptyfiles = glob.glob(f'{megpath}/*{emptyroom_tag}*{dattype}')
                        numemptyfiles = len(emptyfiles)
                        if numemptyfiles == 0: 
                            eroom = None
                            emptyroom_run = None
                            eroom_run_list = []
                            
                        if numemptyfiles > 0:         
                            # if there are more than one emptyroom datasets, move them away.
                            # this was really just a hack to accomodate Omega
                            # if (numemptyfiles > 1):   
                            #    for i in range(1,numemptyfiles):
                            #        epathextra = mne_bids.get_bids_path_from_fname(emptyfiles[i])
                            #        epathextra.update(suffix=None,extension=None)
                            #        epathextra_flist = glob.glob(f'{epathextra.directory}/{epathextra.basename}*')
                            #        for efile in epathextra_flist:
                            #            efile_basename = os.path.basename(efile)
                            #            print(efile_basename)
                            #            print(f'{bids_root}/extra_empty_rooms/{efile_basename}')
                            #            shutil.move(efile,f'{bids_root}/extra_empty_rooms/{efile_basename}')                              
                            eroom_run_list = get_runnumbers(megpath, emptyroom_tag)          
                           
                        # make a list of the runs, and make a new entry for each run
                        run_list = get_runnumbers(megpath, rest_tag)
                        if run_list == []:
                            run_list = ['None']
                        print('found %d rest runs for session %s' % (len(run_list), session))
                        for run in run_list:
                        # we'll have to search for available MEGs looping over all datatypes
                            
                            if run == 'None':
                                if dattype == 'c,rf':
                                    restfiles = glob.glob(f'{megpath}/*{rest_tag}*/*{dattype}*')
                                else:
                                    restfiles = glob.glob(f'{megpath}/*{rest_tag}*{dattype}')    
                                if len(eroom_run_list) > 0:
                                    if run in eroom_run_list:
                                        emptyroom_run = run
                                        eroom = glob.glob(f'{megpath}/*{emptyroom_tag}*{dattype}')
                                        eroom = eroom[0]
                                    else:
                                        eroom = emptyfiles[0]
                                        emptyroom_run = mne_bids.get_entities_from_fname(eroom)['run']
                            else:
                                if dattype == 'c,rf':
                                    restfiles = glob.glob(f'{megpath}/*{rest_tag}*run-{run}*/*{dattype}*')
                                else:
                                    restfiles = glob.glob(f'{megpath}/*{rest_tag}*run-{run}*{dattype}')
                                if len(eroom_run_list) > 0:
                                    if run in eroom_run_list:
                                        emptyroom_run = run
                                        if dattype == 'c,rf':
                                            eroom = glob.glob(f'{megpath}/*{emptyroom_tag}*run-{run}*/*{dattype}*')
                                        else:
                                            eroom = glob.glob(f'{megpath}/*{emptyroom_tag}*run-{run}*{dattype}')
                                        eroom = eroom[0]
                                                  
                                    else:
                                        eroom = emptyfiles[0]
                                        emptyroom_run = mne_bids.get_entities_from_fname(eroom)['run']
                                
                        
                            # if an MEG dataset is found, create an object 
                            if len(restfiles) > 0:
                                print('found an MEG')
                                record_count = rest_count + 1
                                proc_object = munch.Munch()
                                proc_object.sub = subject
                                proc_object.ses = session
                                proc_object.run = run
                                proc_object.type = dattype
                                proc_object.path = restfiles[0]
                                proc_object.mripath = ''
                                proc_object.eroom = eroom
                                proc_object.emptyroom_run = emptyroom_run
                                proc_object.task = rest_tag
                                subrestlist_ses.append(proc_object)
                                
                            # check if there's a processing tag in the file, make a link without it if so
                            tmp_ents = mne_bids.get_entities_from_fname(restfiles[0])
                            if ((tmp_ents['processing'] != None) & (args.make_link == True)):
                                bids_path_orig = mne_bids.get_bids_path_from_fname(restfiles[0])
                                bids_path_out = bids_path_orig.copy().update(processing=None)
                                if not os.path.islink(bids_path_out.fpath):
                                  print('making a symbolic link to make a same session MRI')
                                  os.symlink(bids_path_orig.fpath, bids_path_out.fpath)
                            proc_object.task = tmp_ents['task']
                                           
                subrestlist.extend(subrestlist_ses)
       
        # OKAY. So at this point we have two lists. One has all the MRIs, and one has all the 
        # MpdEGs. We have to match them up. 

        if len(mrilist) != 0:
            for restmeg in subrestlist:

                print('going through list of MEGs for subject %s and matching with MRI' % subject)
                # look at the session of the MEG and see if there is an MRI in the same session
                for mri in mrilist:                
                    if mri.ses == restmeg.ses:                    
                        restmeg.mripath = mri.path
                        print('Session %s match for MEG and MRI' % mri.ses)
              
                # what happens if there is no mri in the same sesssion as the meg           
                if restmeg.mripath == '':
                    restmeg.mripath = mrilist[0].path
                    print('No MRI MEG session match, using an MRI from same subject, different session')
                    if args.make_link == True:
                        anat_bidspath = mne_bids.get_bids_path_from_fname(restmeg.mripath)
                        all_anat_files = glob.glob(f'{anat_bidspath.directory}/*')
                        new_anat_bidspath = anat_bidspath.copy().update(session = restmeg.ses)
                        if not os.path.isdir(new_anat_bidspath.directory):
                            os.mkdir(new_anat_bidspath.directory)
                        for file in all_anat_files:
                            file_bidspath = mne_bids.get_bids_path_from_fname(file,check=False)
                            new_file_bidspath = file_bidspath.copy().update(session=restmeg.ses,check=False,run=None)
                            if not os.path.islink(new_file_bidspath.fpath):
                                print('making a symbolic link to make a same session MRI')
                                os.symlink(file_bidspath.fpath, new_file_bidspath.fpath)
                            else:
                                print('link already exists to anatomical, doing nothing')
                        restmeg.mripath=new_anat_bidspath.fpath
                        mri_count = mri_count+1
                        mri_object = munch.Munch()
                        mri_object.ses = restmeg.ses
                        mri_object.path = new_anat_bidspath.fpath
                        mrilist.append(mri_object)
            
            # now, merge all the proc objects into a single dictionary, and make that a dataframe. 
        else:
            print('No MRI for this participant, removing from list')
            subrestlist = []
    
        # once we've finished with the subject, unmunchify the list of objects and make a dataframe for the subject
        subrestlist = munch.unmunchify(subrestlist)     
        subj_df = pd.DataFrame(subrestlist)
    
        # concatenate the subject dataframe with the global datafram for all subjects
        allsubj_df = pd.concat([allsubj_df, subj_df])

    if args.swarmfile == True:
        dframe_toswarm(allsubj_df, bids_root=bids_root,
                       outfile=op.join(os.getcwd(),'enigma_swarmfile.sh'))
    
    if args.makemanifest == True:
        dframe_tomanifest(allsubj_df, bids_root=bids_root,
                       outfile='manifest.txt')
        
    else:
        # save out the dataframe as a .csv file
        allsubj_df.to_csv('ParsedBIDS_dataframe.csv', index=False)
    

if __name__=='__main__':
    main()


    
