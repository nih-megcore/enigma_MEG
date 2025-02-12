#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 13:35:37 2024

@author: nugenta
"""

import pandas as pd
import argparse
import glob
import os


if __name__=='__main__':

    # parse the arguments and initialize variables   

    parser = argparse.ArgumentParser()
    parser.add_argument('-group_root', help='''The name of the GROUP directory''')

    args = parser.parse_args()
    group_root = args.group_root

    powercsvlist = glob.glob(f'{group_root}/*Power_dataframe.csv')
    spectracsvlist = glob.glob(f'{group_root}/*Spectra_dataframe.csv')
    extracsvlist = glob.glob(f'{group_root}/*Extras_dataframe.csv')
    list_of_powerdframes_norepeats = []
    list_of_spectradframes_norepeats = []
    list_of_extradframes_norepeats = []
    list_of_powerdframes_all = []
    list_of_spectradframes_all = []
    list_of_extradframes_all = []

    for i in range(len(powercsvlist)):

        print('working on %s' % (powercsvlist[i]))
        
        # read in each dataframe
        powerdataframe = pd.read_csv(powercsvlist[i], na_filter=False,dtype=str)
        spectradataframe = pd.read_csv(spectracsvlist[i], na_filter=False,dtype=str)
        extradataframe = pd.read_csv(extracsvlist[i], na_filter=False,dtype=str)
        
        print('Initial number of scans: %d' % (len(powerdataframe)/448))
        
        # do some filtering - remove the datasets with < 10 epochs
        
        powerdataframe = powerdataframe[powerdataframe['epochs_final'].astype(int)>=10]
        spectradataframe = spectradataframe[spectradataframe['epochs_final'].astype(int)>=10]
        extradataframe= extradataframe[extradataframe['epochs_final'].astype(int)>=10]
        
        print('Number of scans with >= 10 epochs: %d' % (len(powerdataframe)/448))
        
        # I'm taking this section out for now - - given that we visually looked at every surface, and defects don't necessarily matter for our purposes.
        
        # and remove the ones with high Euler numbers (low quality freesurfer)
        # the threshold is EulerNumber > -220 from Rosen, et al. NI 2018
        
        #powerdataframe = powerdataframe[powerdataframe['avg_holes'].astype(float)<111]
        #spectradataframe =spectradataframe[spectradataframe['avg_holes'].astype(float)<111]
        #extradataframe = extradataframe[extradataframe['avg_holes'].astype(float)<111]
        
        #print('Number of scans with holes < 111: %d' % (len(powerdataframe)/448))
        
        # finally remove any where the QA was bad
        
        powerdataframe = powerdataframe[powerdataframe['QA_FINAL']=='GOOD']
        spectradataframe = spectradataframe[spectradataframe['QA_FINAL']=='GOOD']
        extradataframe = extradataframe[extradataframe['QA_final']=='GOOD']
        
        print('Final number of scans after QA removal: %d' % (len(powerdataframe)/448))
            
        # get the study name from 0he             
        StudyName = os.path.basename(powercsvlist[i]).split('_')[0]
        print(StudyName)
        if 'NIH' in StudyName:
            SiteName = 'NIH'
        elif 'Aston' in StudyName:
            SiteName = 'Aston'
        elif 'CEA' in StudyName:
            SiteName = 'CEA'
        elif 'SanCam' in StudyName:
            SiteName = 'SANCAMILLO'
        elif 'MRN' in StudyName:
            SiteName = 'MRN'
        else:
            SiteName = StudyName       
            
        # adding eyesopen and eyesclosed designations based on dataset
        
        if (StudyName == 'GOTTINGEN') | (StudyName == 'CAMCAN') | (StudyName == 'WakeForest') | (StudyName == 'NatMEGPD') | (StudyName == 'REMPE') | (StudyName == 'NIHHV')| (StudyName == 'NIHHV2') | (StudyName == 'NIHZARATE') | (StudyName == 'NIHBERMAN') | (StudyName == 'UAB'):
            powerdataframe['task'] = 'eyesclosed'
            spectradataframe['task'] = 'eyesclosed'
            extradataframe['task'] = 'eyesclosed'
        if (StudyName == 'MOUS') | ('MRN' in StudyName) | (StudyName == 'CHOP') | (StudyName == 'HCP') | (StudyName == 'OMEGA') | (StudyName == 'Aston1') | (StudyName == 'NIHSTRINGARIS') | (StudyName == 'NIHYANOVSKI') | (SiteName == 'CEA') | (StudyName == 'NYUAD') | (StudyName == 'NYUNY') | (StudyName == 'UPMCDS'):
            powerdataframe['task'] = 'eyesopen'
            spectradataframe['task'] = 'eyesopen'
            extradataframe['task'] = 'eyesopen'
            
        # add in the site and study names to the main dataframe    
            
        powerdataframe['site'] = SiteName
        powerdataframe['study']= StudyName
        spectradataframe['site'] = SiteName
        spectradataframe['study']= StudyName
        extradataframe['site'] = SiteName
        extradataframe['study']= StudyName
        
        # sort the datasets based on subject, task, session, and run
        
        powersort = powerdataframe.sort_values(by=['subject','task','ses','run'])
        spectrasort = spectradataframe.sort_values(by=['subject','task','ses','run'])
        extrasort = extradataframe.sort_values(by=['subject','task','ses','run'])
        
        # make a temporary dataset that has just the first entry for each subject based on subject, session, task, and run
        
        powerfirst = powersort.groupby(['subject','task']).first().reset_index(drop=False)
        spectrafirst = spectrasort.groupby(['subject','task']).first().reset_index(drop=False)
        extrafirst = extrasort.groupby(['subject','task']).first().reset_index(drop=False)
        
        # trim these down to just the columns to be merged on

        powerfirst = powerfirst[['subject','task','ses','run']]
        spectrafirst = spectrafirst[['subject','task','ses','run']]
        extrafirst = extrafirst[['subject','task','ses','run']]
        
        # now merge to get a dataset with only the first entry for each participant, but with all the data
        
        powermerge = pd.merge(powerdataframe, powerfirst,how='inner',
                              on=['subject','task','ses','run']).reset_index(drop=True)
        spectramerge = pd.merge(spectradataframe, spectrafirst,how='inner',
                              on=['subject','task','ses','run']).reset_index(drop=True)
        extramerge = pd.merge(extradataframe, extrafirst,how='inner',
                              on=['subject','task','ses','run']).reset_index(drop=True)
        
        print('length of dataset with repeat runs %d length of datasets with repeats removed %d' % (len(powerfirst), len(powersort)/448))
        
        list_of_powerdframes_norepeats.append(powermerge) 
        list_of_spectradframes_norepeats.append(spectramerge) 
        list_of_extradframes_norepeats.append(extramerge) 
        
        list_of_powerdframes_all.append(powersort) 
        list_of_spectradframes_all.append(spectrasort) 
        list_of_extradframes_all.append(extrasort)
    
        
    powermegaframe_all = pd.concat(list_of_powerdframes_all).reset_index(drop=True)
    spectramegaframe_all = pd.concat(list_of_spectradframes_all).reset_index(drop=True)
    extramegaframe_all = pd.concat(list_of_extradframes_all).reset_index(drop=True)
    
    powermegaframe_norepeats = pd.concat(list_of_powerdframes_norepeats).reset_index(drop=True)
    spectramegaframe_norepeats = pd.concat(list_of_spectradframes_norepeats).reset_index(drop=True)
    extramegaframe_norepeats = pd.concat(list_of_extradframes_norepeats).reset_index(drop=True)
    
    print('Total datasets %d' % (len(powermegaframe_all)/448))
    print('Total datasets without repeat runs %d' % (len(powermegaframe_norepeats)/448))
    
    powersingle = powermegaframe_norepeats.groupby(['subject']).first().reset_index(drop=False)
    
    print('Total number of unique subjects %d' % (len(powersingle)))
    
    print('Total eyesopen datasets %d' % (len(powermegaframe_all[powermegaframe_all['task']=='eyesopen'])/448))
    print('Total eyesclosed datasets %d' % (len(powermegaframe_all[powermegaframe_all['task']=='eyesclosed'])/448))
    
    print('Total eyesopen datasets with no repeat runs %d' % (len(powermegaframe_norepeats[powermegaframe_norepeats['task']=='eyesopen'])/448))
    print('Total eyesclosed datasets with no repeat runs %d' % (len(powermegaframe_norepeats[powermegaframe_norepeats['task']=='eyesclosed'])/448))
   
    controls_powermegaframe_all = powermegaframe_all[powermegaframe_all['group'] == 'Control']
    controls_spectramegaframe_all = spectramegaframe_all[spectramegaframe_all['group'] == 'Control']
    controls_extramegaframe_all = extramegaframe_all[extramegaframe_all['group'] == 'Control']
    
    controls_powermegaframe_norepeats = powermegaframe_norepeats[powermegaframe_norepeats['group'] == 'Control']
    controls_spectramegaframe_norepeats = spectramegaframe_norepeats[spectramegaframe_norepeats['group'] == 'Control']
    controls_extramegaframe_norepeats = extramegaframe_norepeats[extramegaframe_norepeats['group'] == 'Control']
    
    controlsingle = controls_powermegaframe_norepeats.groupby(['subject']).first().reset_index(drop=False)
    
    print('Total control datasets %d' % (len(controls_powermegaframe_all)/448))
    print('Total control datasets with no repeats %d' % (len(controls_powermegaframe_norepeats)/448))
    
    print('Total number of unique control subjects %d' % (len(controlsingle)))
    
    print('Total eyesopen control datasets %d' % (len(controls_powermegaframe_all[controls_powermegaframe_all['task']=='eyesopen'])/448))
    print('Total eyesclosed control datasets %d' % (len(controls_powermegaframe_all[controls_powermegaframe_all['task']=='eyesclosed'])/448))
    
    print('Total eyesopen control datasets with no repeat runs %d' % (len(controls_powermegaframe_norepeats[controls_powermegaframe_norepeats['task']=='eyesopen'])/448))
    print('Total eyesclosed control datasets with no repeat runs %d' % (len(controls_powermegaframe_norepeats[controls_powermegaframe_norepeats['task']=='eyesclosed'])/448))
      
    controls_powermegaframe_all.to_csv('Controls_Power_dataframe_all.csv')
    controls_powermegaframe_all.to_csv('Controls_Spectra_dataframe_all.csv')
    controls_powermegaframe_all.to_csv('Controls_Extra_dataframe_all.csv')
    
    controls_powermegaframe_norepeats.to_csv('Controls_Power_dataframe_norepeats.csv')
    controls_powermegaframe_norepeats.to_csv('Controls_Spectra_dataframe_norepeats.csv')
    controls_powermegaframe_norepeats.to_csv('Controls_Extra_dataframe_norepeats.csv')   
    
    powermegaframe_all.to_csv('Power_dataframe_all.csv')
    spectramegaframe_all.to_csv('Spectra_dataframe_all.csv')
    extramegaframe_all.to_csv('Extra_dataframe_all.csv')
    
    powermegaframe_norepeats.to_csv('Power_dataframe_norepeats.csv')
    spectramegaframe_norepeats.to_csv('Spectra_dataframe_norepeats.csv')
    extramegaframe_norepeats.to_csv('Extra_dataframe_norepeats.csv')   
    

    