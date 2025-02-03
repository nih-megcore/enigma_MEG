#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 12 14:34:01 2023

@author: jstout
"""

import os, os.path as op
demo_dir = '/home/jstout/src/enigma_OHBM/RESULTS_OHBM_FIN/DEMOG'
topdir = demo_dir
os.chdir(topdir)

import pandas as pd

# =============================================================================
# Get age/sex from participants.tsv
# Download the all_demographics folder from biowulf enigma/bids/all_demographics
# =============================================================================
# demo_dir = '/tmp/all_demographics'
demo=dict(ICA_1='NIHstr_participants.tsv',
      results_mous='mous_participants.tsv',
      NIH_y='NIHyan_participants.tsv',
      NIH_hvprotocol='NIHhv_participants.tsv', 
      Omega = 'omega_participants.tsv')

def load_demo(fname):
    return pd.read_csv(op.join(demo_dir, fname), sep='\t')

demo_list = []
column_sel = ['participant_id', 'age', 'sex','SheetName']

#NIHstr data
tmp = load_demo('NIHstr_participants.tsv')
tmp['SheetName']='ICA_1'
tmp.rename(columns=dict(participant_age='age'), inplace=True)
demo_list.append(tmp.loc[:,column_sel])

#NIHy data
# tmp = load_demo('NIHyan_participants.tsv')
# tmp['SheetName']='NIH_y'
# tmp.rename(columns=dict(participant_age='age'), inplace=True)
# demo_list.append(tmp.loc[:,column_sel])

#NIHhv data
tmp = load_demo('NIHhv_participants.tsv')
tmp['SheetName']='NIH_hvprotocol'
demo_list.append(tmp.loc[:,column_sel])

#Mous data
tmp = load_demo('mous_participants.tsv')
tmp['SheetName']='results_mous'
demo_list.append(tmp.loc[:,column_sel])

#Camcan data
tmp = load_demo('cam_participants.tsv')
tmp['SheetName']='results_camcan'
demo_list.append(tmp.loc[:,column_sel])

#Omega data
tmp = load_demo('omega_participants.tsv')
tmp['SheetName'] = 'omega'
demo_list.append(tmp.loc[:,column_sel])

#Cardiff data
tmp = pd.read_csv('meguk_cardiff_participants.csv')
tmp['SheetName'] = 'cardiff'
tmp.rename(columns={'MEG-PART-ID':'participant_id',
            'Sex':'sex',
            'Age':'age'}, inplace=True)
demo_list.append(tmp.loc[:,column_sel])


demo_final = pd.concat(demo_list)
demo_final.reset_index(drop=True, inplace=True)



# =============================================================================
# Merge demographic and ICA results
# =============================================================================
#ICAColumns
#['idx', 'sub', 'type', 'eyeblink', 'Saccade', 'EKG', 'other',
#       'SheetName', 'Scanner', 'Site', 'TaskType', 'Unnamed: 6']

def munge_subjid(subj):
    subj=str(subj)
    if subj[0:4]!='sub-':
        subj='sub-'+subj
    return subj

def fix_nihy_subjids(dframe):
    'Specifically parse datasets from NIH_y - subject IDs need to conform'
    for idx, row in dframe.iterrows():
        if row.SheetName=='NIH_y':
            print(f"Fixing ID {dframe.loc[idx]['participant_id']}")
            if len(str(dframe.loc[idx]['participant_id']))==1:
                dframe.loc[idx, 'participant_id']='sub-000'+str(dframe.loc[idx]['participant_id'])
            elif  len(str(dframe.loc[idx,'participant_id']))==2:
                dframe.loc[idx, 'participant_id']='sub-00'+str(dframe.loc[idx]['participant_id'])
            else:
                print(f'Error with {dframe.loc[idx]["participant_id"]}')
    

import copy
all_dat = copy.deepcopy(demo_final)

reject_idx=[]
for idx,i in enumerate(all_dat.age):
    try:
        int(i)
    except:
        reject_idx.append(idx)
        continue
    if int(i) < 0:
        reject_idx.append(idx)
        
    
    
all_dat.drop(reject_idx, inplace=True)


# all_dat[['participant_id', 'type']]
#Cleanup
#Drop subjects without age
all_dat=all_dat.dropna(subset=['age'])
all_dat['age']=all_dat['age'].astype(int)
all_dat['age'].hist()



#Change coding of M/F for hv protocol
hv_m_idx = all_dat[(all_dat.SheetName=='NIH_hvprotocol') & (all_dat.sex==1)].index
hv_f_idx = all_dat[(all_dat.SheetName=='NIH_hvprotocol') & (all_dat.sex==2)].index
all_dat.loc[hv_m_idx,'sex']='M'
all_dat.loc[hv_f_idx,'sex']='F'


#Use single letter for sex 
all_dat.sex.dropna(inplace=True)
all_dat.drop(index=all_dat[all_dat.sex==-999].index, inplace=True)
all_dat['sex']=all_dat['sex'].str[0].apply(str.upper)


# #Remove run number from mmi task id
# all_dat.loc[all_dat.type.str[0:4]=='mmi3','type']='mmi3'


# =============================================================================
# Demographic level stats
# =============================================================================
all_dat=all_dat.drop_duplicates(subset=['participant_id','SheetName'])
# group_demo_data.age.hist(density=True)
# group_demo_data.sex.hist()

rename_dict=dict(ICA_1='NIHstr',
              results_mous='MOUS',
              NIH_hvprotocol='NIHhv',
              results_camcan='CAMCAN',
              omega = 'OMEGA') 

all_dat.SheetName.replace(rename_dict, inplace=True)
all_dat.rename(columns=dict(SheetName='Site'), inplace=True)
all_dat.reset_index(inplace=True, drop=True)
