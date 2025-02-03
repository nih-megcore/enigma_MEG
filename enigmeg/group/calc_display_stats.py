#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 07:41:59 2020

@author: stoutjd
"""

'''TODO:
    use parc as input so that subjects without full parc will not be added
    
    Fix testing
    Add commandline functionality
    '''


import pandas as pd
import numpy as np
import os, os.path as op
import glob
import seaborn as sns
import pylab
import nibabel as nib
from mne.viz import Brain


import statsmodels.api as sm
import statsmodels.formula.api as smf


def compile_group_outputs(enigma_outputs_path=None):
    '''Find all single subject enigma csv files and compile into a single
    dataframe'''

    os.chdir(enigma_outputs_path)
    filenames = glob.glob('*/Band_rel_power.csv')
    return filenames

def return_multisubject_dframe(filenames):
    '''Take a list of filenames 
    Load the csv filenames into a dataframe
    Stitch together single subject dataframes into one large dataframe'''
    
    list_of_dframes = list()
    for fname in filenames:
        print(fname)
        subj_dframe = pd.read_csv(fname, sep='\t')       
        subj_dframe['subject'] = fname.split('/')[0]
        list_of_dframes.append(subj_dframe)
        
    group_dframe = pd.concat(list_of_dframes)
    group_dframe = group_dframe.rename(columns={'Unnamed: 0':'Parcel'})
    return group_dframe
        
        
def merge_in_demographics(subjs_vars=None,
                          parcel_dat=None,
                          subjs_vars_merge_idx=None,
                          parcel_merge_idx=None):
    '''Check for filenames and load dataframes
    Merge dataframes'''
    
    if subjs_vars_merge_idx==None: subjs_vars_merge_idx='subject'
    if parcel_merge_idx==None: parcel_merge_idx='subject'
    
    ## Determine if demographics and enigma data are dataframes or files
    if type(subjs_vars) == str:
        try: 
            subjs_vars = pd.read_csv(subjs_vars, delimiter='\t')
        except:
            print("Cannot determine subjs_vars - Does not appear to be tsv \
                  file or dataframe")
            raise ValueError()
    if type(parcel_dat) == str:
        try:
            parcel_dat = pd.read_csv(parcel_dat, delimiter='\t')
        except:
            print("Cannot determine enigma_dat - Does not appear to be tsv \
                  file or dataframe")
            raise ValueError()

    demogr_subjs = subjs_vars[subjs_vars_merge_idx].unique()
    parc_subjs = parcel_dat[parcel_merge_idx].unique()
    num_demogr_subjs = demogr_subjs.__len__()
    num_parc_subjs = parc_subjs.__len__()
    diff_subjs = set(demogr_subjs).difference(parc_subjs)

    print(f'Demographic_subjs: {num_demogr_subjs}\
          \nParcel_subjs: {num_parc_subjs}')

    #! #### FIX
    ###### FIX need to log the subjects that were not merged ################################
    #####
    
    dframe=pd.merge(subjs_vars, 
                    parcel_dat, 
                    left_on=subjs_vars_merge_idx,
                    right_on=parcel_merge_idx) 
    return dframe
 

def filter_parcel_data(dframe, parcel_name=None, column_name=None):
    '''Return single parcel information for stats testing'''
    tmp=dframe[dframe.Parcel==parcel_name]
    return tmp[[column_name, 'age', 'SEX']].dropna()
    # return tmp[[column_name, 'age', 'sex_text']].dropna()


def proc_parcel_regression(dframe):
    '''Takes merged dataframe (subj_vars + parcel data)
    and performs regression of variables using statsmodels'''
    
    #Initialize regression dataframe
    rdframe = pd.DataFrame(columns=['parcel_name', 'coeff','rsquared_adj'])
    rdframe.parcel_name = dframe.Parcel.unique()

    for idx, row in rdframe.iterrows():
        parc = row['parcel_name']
        tmp=filter_parcel_data(dframe, parcel_name=parc, 
                               column_name='AlphaPeak')
        results = smf.ols('AlphaPeak ~ age', data=tmp).fit()
        rdframe.loc[idx, 'rsquared_adj']=results.rsquared_adj
        rdframe.loc[idx, 'coeff']=results.params['age']
    
    return rdframe



##################  <<<<<<<<<<<<<<<<  CHECK

# dframe.parcel_name.values

# tmp=filter_parcel_data(dframe, parcel_name='inferiortemporal_1-lh', column_name='AlphaPeak')
# # results = smf.ols('AlphaPeak ~ age', data=tmp).fit()
# # print(results.summary())


# pylab.scatter(tmp['age'], tmp['AlphaPeak'])
# pylab.xlabel('age'); pylab.ylabel('AlphaPeak')

# pylab.hist(demographics.age)

# pylab.hist(dframe.age)

# rdframe.to_csv('Alpha_vs_age.csv', index=False)


def display_regression_coefs(stats_dframe, 
                             subject_id='fsaverage',
                             parc_name='aparc',
                             subjects_dir=None,
                             image_outpath=None):
    '''Plot the statistics on the brain
    parc_name:
        aparc or aparc_sub currently supported
    
    '''
   
    if subjects_dir != None: os.environ['SUBJECTS_DIR']=subjects_dir
    hemi = "lh"
    surf = "inflated"
    
    brain = Brain(subject_id, hemi, surf, background="white")
    
    aparc_file = os.path.join(os.environ["SUBJECTS_DIR"],
                              subject_id, "label",
                              hemi + f".{parc_name}.annot") 
    labels, ctab, names = nib.freesurfer.read_annot(aparc_file)
    
    names2=[i.decode() for i in names] #convert from binary
    if 'corpuscallosum' in names2: names2.remove('corpuscallosum')
    if 'unknown' in names2: names2.remove('unknown')
    
    # Placeholder - must be larger than number of ROIs
    roi_data = np.zeros(600)
    
    for idx,name in enumerate(names2):
        if stats_dframe.loc[idx, 'parcel_name'].split('-')[0] != name:
            print(f'{stats_dframe.loc[idx, "parcel_name"].split("-")[0]} \
                  != {name}')
            raise(ValueError)
        roi_data[idx]=stats_dframe.loc[idx, 'coeff']
    	
    vtx_data = roi_data[labels]
    vtx_data[labels == -1] = 0
    
    thresh=.001
    vtx_data[np.abs(vtx_data)<thresh]=0
    
    brain.add_data(vtx_data, fmin=-0.05, fmax=0.05, colormap="coolwarm", alpha=1) 
    
    if image_outpath != None:
        brain.save_image(image_outpath)




def test_merge_in_demographics():
    subjs_vars='/home/stoutjd/data/test_data/GROUP/demographics/age_regressors.tsv'
    parcel_dat='/data/test_data/GROUP/enigma_outputs/group_outputs.tsv'
    subjs_vars_merge_idx='subject',
    parcel_merge_idx='subject' 
    
    #Temporary_patch
    subjs_vars = pd.read_csv(subjs_vars, delimiter='\t')
    subjs_vars['subject']=subjs_vars['subject']+'_fs'    
    
    dframe = merge_in_demographics(subjs_vars=subjs_vars,
                          parcel_dat=parcel_dat,
                          subjs_vars_merge_idx='subject',
                          parcel_merge_idx='subject')
    assert dframe.shape[0] == 25022 
    assert dframe.shape[1] == 10
    
    dframe=dframe.rename(columns={'AGE':'age'})
    
    stats_dframe=proc_parcel_regression(dframe)
    
    display_regression_coefs(stats_dframe, 
                             subject_id='fsaverage',
                             parc_name='aparc_sub',
                             subjects_dir=None,
                             image_outpath='/tmp/testfile.png')
    
    
    
    
    
    




def test_main():
    filenames = compile_group_outputs('/data/test_data/GROUP/enigma_outputs')
    dframe = return_multisubject_dframe(filenames)    


def test_plots(dframe):
    plot_frame = dframe[['AlphaPeak','subject', 'Parcel']]
    rois = ['pericalcarine_1-rh','postcentral_1-rh', 'bankssts_1-rh',
    'pericalcarine_1-lh','postcentral_1-lh', 'bankssts_1-lh']
    
    plot_frame=plot_frame[plot_frame.Parcel.isin(rois)]
    plot_frame.dropna(inplace=True)
    
    sns.histplot(data=plot_frame, x='Parcel', y='AlphaPeak')
    sns.displot(plot_frame, x="AlphaPeak", kind="kde", bw_adjust=2, hue='Parcel')
    

    # fig=pylab.Figure(figsize=(6,8), dpi=300)  #I dont think this is necessary
    # tmp=sns.displot(plot_frame, x="AlphaPeak", kind="kde", bw_adjust=2, hue='Parcel')
    # tmp.savefig(op.join(input_dir, 'SensoryHists.png'), dpi=300)

def test_compile_group_outputs():
    # from enigmeg import test_data    
    #Hack replace value w/ import parameter
    top_dir = '/data/test_data/GROUP/enigma_outputs'
    
    fnames = compile_group_outputs(top_dir)
    assert len(fnames) == 61
    
    # dframe.to_csv('/data/test_data/GROUP/enigma_outputs/group_outputs.tsv', index=None, sep='\t')
    

# if __name__ == '__main__':
    #dframe.to_csv(op.join(input_dir, 'Compiled_bandrel.csv'), index=False)
