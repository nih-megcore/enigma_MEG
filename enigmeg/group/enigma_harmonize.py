#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  6 09:21:19 2024

@author: nugenta
"""

import pandas as pd
import numpy as np
import neuroHarmonize
from neuroHarmonize import harmonizationLearn, saveHarmonizationModel, harmonizationApply

        
def combine_columns(df):
    
    bankssts_lh = df.filter(regex='^bankssts.*-lh$').mean(axis=1)
    bankssts_rh = df.filter(regex='^bankssts.*-rh$').mean(axis=1)
    caudalanteriorcingulate_lh = df.filter(regex='^caudalanteriorcingulate.*-lh$').mean(axis=1)
    caudalanteriorcingulate_rh = df.filter(regex='^caudalanteriorcingulate.*-rh$').mean(axis=1)
    caudalmiddlefrontal_lh = df.filter(regex='^caudalmiddlefrontal.*-lh$').mean(axis=1)
    caudalmiddlefrontal_rh = df.filter(regex='^caudalmiddlefrontal.*-rh$').mean(axis=1)
    cuneus_lh = df.filter(regex='^cuneus.*-lh$').mean(axis=1)
    cuneus_rh = df.filter(regex='^cuneus.*-rh$').mean(axis=1)
    entorhinal_lh = df.filter(regex='^entorhinal.*-lh$').mean(axis=1)
    entorhinal_rh = df.filter(regex='^entorhinal.*-rh$').mean(axis=1)
    frontalpole_lh = df.filter(regex='^frontalpole.*-lh$').mean(axis=1)
    frontalpole_rh = df.filter(regex='^frontalpole.*-rh$').mean(axis=1)
    fusiform_lh = df.filter(regex='^fusiform.*-lh$').mean(axis=1)
    fusiform_rh = df.filter(regex='^fusiform.*-rh$').mean(axis=1)
    inferiorparietal_lh = df.filter(regex='^inferiorparietal.*-lh$').mean(axis=1)
    inferiorparietal_rh = df.filter(regex='^inferiorparietal.*-rh$').mean(axis=1)
    inferiortemporal_lh = df.filter(regex='^inferiortemporal.*-lh$').mean(axis=1)
    inferiortemporal_rh = df.filter(regex='^inferiortemporal.*-rh$').mean(axis=1)
    insula_lh = df.filter(regex='^insula.*-lh$').mean(axis=1)
    insula_rh = df.filter(regex='^insula.*-rh$').mean(axis=1)
    isthmuscingulate_lh = df.filter(regex='^isthmuscingulate.*-lh$').mean(axis=1)
    isthmuscingulate_rh = df.filter(regex='^isthmuscingulate.*-rh$').mean(axis=1)
    lateraloccipital_lh = df.filter(regex='^lateraloccipital.*-lh$').mean(axis=1)
    lateraloccipital_rh = df.filter(regex='^lateraloccipital.*-rh$').mean(axis=1)
    lateralorbitofrontal_lh = df.filter(regex='^lateralorbitofrontal.*-lh$').mean(axis=1)
    lateralorbitofrontal_rh = df.filter(regex='^lateralorbitofrontal.*-rh$').mean(axis=1)
    lingual_lh = df.filter(regex='^lingual.*-lh$').mean(axis=1)
    lingual_rh = df.filter(regex='^lingual.*-rh$').mean(axis=1)
    medialorbitofrontal_lh = df.filter(regex='^medialorbitofrontal.*-lh$').mean(axis=1)
    medialorbitofrontal_rh = df.filter(regex='^medialorbitofrontal.*-rh$').mean(axis=1)
    middletemporal_lh = df.filter(regex='^middletemporal.*-lh$').mean(axis=1)
    middletemporal_rh = df.filter(regex='^middletemporal.*-rh$').mean(axis=1)
    paracentral_lh = df.filter(regex='^paracentral.*-lh$').mean(axis=1)
    paracentral_rh = df.filter(regex='^paracentral.*-rh$').mean(axis=1)
    parahippocampal_lh = df.filter(regex='^parahippocampal.*-lh$').mean(axis=1)
    parahippocampal_rh = df.filter(regex='^parahippocampal.*-rh$').mean(axis=1)
    parsopercularis_lh = df.filter(regex='^parsopercularis.*-lh$').mean(axis=1)
    parsopercularis_rh = df.filter(regex='^parsopercularis.*-rh$').mean(axis=1)
    parsorbitalis_lh = df.filter(regex='^parsorbitalis.*-lh$').mean(axis=1)
    parsorbitalis_rh = df.filter(regex='^parsorbitalis.*-rh$').mean(axis=1)
    parstriangularis_lh = df.filter(regex='^parstriangularis.*-lh$').mean(axis=1)
    parstriangularis_rh = df.filter(regex='^parstriangularis.*-rh$').mean(axis=1)
    pericalcarine_lh = df.filter(regex='^pericalcarine.*-lh$').mean(axis=1)
    pericalcarine_rh = df.filter(regex='^pericalcarine.*-rh$').mean(axis=1)
    postcentral_lh = df.filter(regex='^postcentral.*-lh$').mean(axis=1)
    postcentral_rh = df.filter(regex='^postcentral.*-rh$').mean(axis=1)
    posteriorcingulate_lh = df.filter(regex='^posteriorcingulate.*-lh$').mean(axis=1)
    posteriorcingulate_rh = df.filter(regex='^posteriorcingulate.*-rh$').mean(axis=1)
    precentral_lh = df.filter(regex='^precentral.*-lh$').mean(axis=1)
    precentral_rh = df.filter(regex='^precentral.*-rh$').mean(axis=1)
    precuneus_lh = df.filter(regex='^precuneus.*-lh$').mean(axis=1)
    precuneus_rh = df.filter(regex='^precuneus.*-rh$').mean(axis=1)
    rostralanteriorcingulate_lh = df.filter(regex='^rostralanteriorcingulate.*-lh$').mean(axis=1)
    rostralanteriorcingulate_rh = df.filter(regex='^rostralanteriorcingulate.*-rh$').mean(axis=1)
    rostralmiddlefrontal_lh = df.filter(regex='^rostralmiddlefrontal.*-lh$').mean(axis=1)
    rostralmiddlefrontal_rh = df.filter(regex='^rostralmiddlefrontal.*-rh$').mean(axis=1)
    superiorfrontal_lh = df.filter(regex='^superiorfrontal.*-lh$').mean(axis=1)
    superiorfrontal_rh = df.filter(regex='^superiorfrontal.*-rh$').mean(axis=1)
    superiorparietal_lh = df.filter(regex='^superiorparietal.*-lh$').mean(axis=1)
    superiorparietal_rh = df.filter(regex='^superiorparietal.*-rh$').mean(axis=1)
    superiortemporal_lh = df.filter(regex='^superiortemporal.*-lh$').mean(axis=1)
    superiortemporal_rh = df.filter(regex='^superiortemporal.*-rh$').mean(axis=1)
    supramarginal_lh = df.filter(regex='^supramarginal.*-lh$').mean(axis=1)
    supramarginal_rh = df.filter(regex='^supramarginal.*-rh$').mean(axis=1)
    temporalpole_lh = df.filter(regex='^temporalpole.*-lh$').mean(axis=1)
    temporalpole_rh = df.filter(regex='^temporalpole.*-rh$').mean(axis=1)
    transversetemporal_lh = df.filter(regex='^transversetemporal.*-lh$').mean(axis=1)
    transversetemporal_rh = df.filter(regex='^transversetemporal.*-rh$').mean(axis=1)

    df['bankssts_lh'] = bankssts_lh
    df['bankssts_rh'] = bankssts_rh
    df['caudalanteriorcingulate_lh'] = caudalanteriorcingulate_lh
    df['caudalanteriorcingulate_rh'] = caudalanteriorcingulate_rh
    df['caudalmiddlefrontal_lh'] = caudalmiddlefrontal_lh
    df['caudalmiddlefrontal_rh'] = caudalmiddlefrontal_rh
    df['cuneus_lh'] = cuneus_lh
    df['cuneus_rh'] = cuneus_rh
    df['entorhinal_lh'] = entorhinal_lh
    df['entorhinal_rh'] = entorhinal_rh
    df['frontalpole_lh'] = frontalpole_lh
    df['frontalpole_rh'] = frontalpole_rh
    df['fusiform_lh'] = fusiform_lh
    df['fusiform_rh'] = fusiform_rh
    df['inferiorparietal_lh'] = inferiorparietal_lh
    df['inferiorparietal_rh'] = inferiorparietal_rh
    df['inferiortemporal_lh'] = inferiortemporal_lh
    df['inferiortemporal_rh'] = inferiortemporal_rh
    df['insula_lh'] = insula_lh
    df['insula_rh'] = insula_rh
    df['isthmuscingulate_lh'] = isthmuscingulate_lh
    df['isthmuscingulate_rh'] = isthmuscingulate_rh
    df['lateraloccipital_lh'] = lateraloccipital_lh
    df['lateraloccipital_rh'] = lateraloccipital_rh
    df['lateralorbitofrontal_lh'] = lateralorbitofrontal_lh
    df['lateralorbitofrontal_rh'] = lateralorbitofrontal_rh
    df['lingual_lh'] = lingual_lh
    df['lingual_rh'] = lingual_rh
    df['medialorbitofrontal_lh'] = medialorbitofrontal_lh
    df['medialorbitofrontal_rh'] = medialorbitofrontal_rh
    df['middletemporal_lh'] = middletemporal_lh
    df['middletemporal_rh'] = middletemporal_rh
    df['paracentral_lh'] = paracentral_lh
    df['paracentral_rh'] = paracentral_rh
    df['parahippocampal_lh'] = parahippocampal_lh
    df['parahippocampal_rh'] = parahippocampal_rh
    df['parsopercularis_lh'] = parsopercularis_lh
    df['parsopercularis_rh'] = parsopercularis_rh
    df['parsorbitalis_lh'] = parsorbitalis_lh
    df['parsorbitalis_rh'] = parsorbitalis_rh
    df['parstriangularis_lh'] = parstriangularis_lh
    df['parstriangularis_rh'] = parstriangularis_rh
    df['pericalcarine_lh'] = pericalcarine_lh
    df['pericalcarine_rh'] = pericalcarine_rh
    df['postcentral_lh'] = postcentral_lh
    df['postcentral_rh'] = postcentral_rh
    df['posteriorcingulate_lh'] = posteriorcingulate_lh
    df['posteriorcingulate_rh'] = posteriorcingulate_rh
    df['precentral_lh'] = precentral_lh
    df['precentral_rh'] = precentral_rh
    df['precuneus_lh'] = precuneus_lh
    df['precuneus_rh'] = precuneus_rh
    df['rostralanteriorcingulate_lh'] = rostralanteriorcingulate_lh
    df['rostralanteriorcingulate_rh'] = rostralanteriorcingulate_rh
    df['rostralmiddlefrontal_lh'] = rostralmiddlefrontal_lh
    df['rostralmiddlefrontal_rh'] = rostralmiddlefrontal_rh
    df['superiorfrontal_lh'] = superiorfrontal_lh
    df['superiorfrontal_rh'] = superiorfrontal_rh
    df['superiorparietal_lh'] = superiorparietal_lh
    df['superiorparietal_rh'] = superiorparietal_rh
    df['superiortemporal_lh'] = superiortemporal_lh
    df['superiortemporal_rh'] = superiortemporal_rh
    df['supramarginal_lh'] = supramarginal_lh
    df['supramarginal_rh'] = supramarginal_rh
    df['temporalpole_lh'] = temporalpole_lh
    df['temporalpole_rh'] = temporalpole_rh
    df['transversetemporal_lh'] = transversetemporal_lh
    df['transversetemporal_rh'] = transversetemporal_rh

    df_out = df[['subject','bankssts_lh','bankssts_rh',
       'caudalanteriorcingulate_lh','caudalanteriorcingulate_rh',
       'caudalmiddlefrontal_rh', 'caudalmiddlefrontal_lh',
       'cuneus_lh','cuneus_rh', 'entorhinal_lh', 'entorhinal_rh',
       'frontalpole_lh', 'frontalpole_rh', 'fusiform_lh','fusiform_rh',
       'inferiorparietal_lh', 'inferiorparietal_rh',
       'inferiortemporal_lh', 'inferiortemporal_rh',
       'insula_lh', 'insula_rh', 'isthmuscingulate_lh','isthmuscingulate_rh',
       'lateraloccipital_lh','lateraloccipital_rh',
       'lateralorbitofrontal_lh','lateralorbitofrontal_rh',
       'lingual_lh','lingual_rh',
       'medialorbitofrontal_lh','medialorbitofrontal_rh',
       'middletemporal_lh','middletemporal_rh',
       'paracentral_lh','paracentral_rh',
       'parahippocampal_lh','parahippocampal_rh',
       'parsopercularis_rh', 'parsopercularis_lh',
       'parsorbitalis_lh', 'parsorbitalis_rh',
       'parstriangularis_lh','parstriangularis_rh', 
       'pericalcarine_lh', 'pericalcarine_rh', 
       'postcentral_lh', 'postcentral_rh', 
       'posteriorcingulate_lh','posteriorcingulate_rh',
       'precentral_lh', 'precentral_rh', 
       'precuneus_lh', 'precuneus_rh',
       'rostralanteriorcingulate_lh', 'rostralanteriorcingulate_rh',
       'rostralmiddlefrontal_lh', 'rostralmiddlefrontal_rh',
       'superiorfrontal_lh', 'superiorfrontal_rh',
       'superiorparietal_lh', 'superiorparietal_rh',
       'superiortemporal_lh', 'superiortemporal_rh',
       'supramarginal_lh', 'supramarginal_rh', 
       'temporalpole_lh','temporalpole_rh', 
       'transversetemporal_lh','transversetemporal_rh'
    ]]
    
    return df_out

def make_power_dataframes(dataframe):
    
    delta = dataframe.pivot(index=['subject','task'], columns='Parcel', values='[1, 3]').sort_index().reset_index()
    theta = dataframe.pivot(index=['subject','task'], columns='Parcel', values='[3, 6]').sort_index().reset_index()
    alpha = dataframe.pivot(index=['subject','task'], columns='Parcel', values='[8, 12]').sort_index().reset_index()
    beta = dataframe.pivot(index=['subject','task'], columns='Parcel', values='[13, 35]').sort_index().reset_index()
    gamma = dataframe.pivot(index=['subject','task'], columns='Parcel', values='[35, 45]').sort_index().reset_index()
    aperoffset = dataframe.pivot(index=['subject','task'], columns='Parcel', values='AperiodicOffset').sort_index().reset_index()
    aperexp = dataframe.pivot(index=['subject','task'], columns='Parcel', values='AperiodicExponent').sort_index().reset_index()
    
    return delta, theta, alpha, beta, gamma, aperoffset, aperexp


def prepare_to_harmonize(dataframe, single, num_parcels):
    
    dataframe=pd.merge(
        dataframe.drop(columns=['age','sex','site','study'], errors='ignore'),
        single[['subject', 'task', 'age', 'sex', 'site', 'study']],
        on=['subject', 'task'],
        how='left'
    )
    
    parcel_cols = [col for col in dataframe.columns if col not in ['subject', 'task','age','sex','site','study']]
    data_to_harmonize = dataframe[parcel_cols].to_numpy()
    
    return dataframe, data_to_harmonize, parcel_cols

def process_adjusted(data, parcel_cols, single, num_parcels):
        
    data_dframe = pd.DataFrame(data, columns=parcel_cols)
    data_dframe['subject'] = single['subject'].values
    data_dframe['age'] = single['age'].values
    data_dframe['sex'] = single['sex'].values
    data_dframe['site'] = single['site'].values
    data_dframe['task'] = single['task'].values
    data_dframe['study'] = single['study'].values

    return data_dframe

def harmonize(filename, do_subparc, harm_task, prefix, groupbyvar, band):
    
    if do_subparc == True:
        num_parcels = 448
    else:
        num_parcels = 68
     
    dataframe = pd.read_csv(filename)
    
    print('Dataframe read, number of subjects is %d' % (int(len(dataframe)/448)))
 
    dataframe = dataframe.dropna(subset=['age'])
    dataframe = dataframe[dataframe['sex'].isin(['M','F'])]
    
    print('Age NaN and sex N values dropped, number of subjects is %d' % (int(len(dataframe)/448)), flush=True)
 
    # extract column headings 
    dataframe['study'] = dataframe['study'].astype('category')
    dataframe['site'] = dataframe['site'].astype('category')  
    dataframe['sex'] = dataframe['sex'].astype('category')
    dataframe['task'] = dataframe['task'].astype('category')
    
    dataframe=dataframe.sort_values(['subject','task']).reset_index(drop=True)
 
    delta, theta, alpha, beta, gamma, aperoffset, aperexp = make_power_dataframes(dataframe)
    single = dataframe[dataframe['Parcel'] == 'bankssts_1-lh'].copy()
    
    expected_n = dataframe[['subject', 'task']].drop_duplicates().shape[0]
    actual_n = single.shape[0]

    if expected_n != actual_n:
        print(f"WARNING: Expected {expected_n} subject-task pairs, found {actual_n} in 'single'")
 
    # select the band to harmonize
    
    if band == 'delta':
        selected = delta
        savename = 'delta'
    elif band == 'theta':
        selected = theta
        savename = 'theta'
    elif band == 'alpha':
        selected = alpha
        savename = 'alpha'
    elif band == 'beta':
        selected = beta
        savename = 'beta'
    elif band == 'gamma':
        selected = gamma
        savename = 'gamma'
    elif band == 'aperoffset':
        selected = aperoffset
        savename = 'aperoffset'
    elif band == 'aperexp':
        selected = aperexp
        savename = 'aperexp'   
        
    # combine the subparcellation columns
 
    if do_subparc == False: 
     
        print('Combining subparcellation',flush=True)    

        selected = combine_columns(selected)

    # prepare the datasets for harmonization
            
    print('Preparing selected datasets for Harmonization',flush=True)
      
    single = single.sort_values(['subject', 'task']).reset_index(drop=True)           
    selected = selected.sort_values(['subject', 'task']).reset_index(drop=True)     
    
    selected, selected_data, parcel_cols = prepare_to_harmonize(selected, single, num_parcels)
    
    single['study'] = single['study'].astype('category')
    single['site'] = single['site'].astype('category')  
    single['sex'] = single['sex'].astype('category')
    single['task'] = single['task'].astype('category') 
    
    if (harm_task == False):
        covars = pd.DataFrame.from_dict({'SITE': single[groupbyvar].cat.codes,
                                  'age': single.age.values,
                                  'sex': single.sex.cat.codes})
    else:
        covars = pd.DataFrame.from_dict({'SITE': single[groupbyvar].cat.codes,
                                  'age': single.age.values,
                                  'sex': single.sex.cat.codes,
                                  'task': single.task.cat.codes})
  
    # ComBatGam neuroHarmonize

    print('Harmonizing',flush=True)
    model, adjusted_gamcombat = harmonizationLearn(np.array(selected_data),
                                    covars, smooth_terms=['age'])
    

    if do_subparc == False:
            saveHarmonizationModel(model, f'{prefix}_{savename}_full_neuroharmonizemodel')
    else:
            saveHarmonizationModel(model, f'{prefix}_{savename}_full_neuroharmonizemodel_subparc')
     
    adjusted_gamcombat_dframe = process_adjusted(adjusted_gamcombat, parcel_cols, single, num_parcels)
 
    # Apply the model to the holdout data, if necessary
  
    print('Saving output data',flush=True)
 
    if do_subparc == False:
            adjusted_gamcombat_dframe.to_csv(f'{prefix}_gamComBatAdjusted_{savename}_dataframe.csv')
            selected.to_csv(f'{prefix}_Orig_{savename}_dataframe.csv')
    else:
            adjusted_gamcombat_dframe.to_csv(f'{prefix}_gamComBatAdjusted_{savename}_dataframe_subparc.csv')
            selected.to_csv(f'{prefix}_Orig_{savename}_dataframe_subparc.csv')

    print('Done, thanks for harmonizing!')

def main():
    
    import argparse  
    parser = argparse.ArgumentParser()

    parser.add_argument('-input_file', help='''The name of the group output file to be harmonized''')
    parser.add_argument('-process_full_subparc', help='''Process the full subparcellation''',
                        action='store_true',
                        default=0)
    parser.add_argument('-harmonize_task', help='''Use task as a variable to retain post-harmonization''',
                        action='store_true',
                        default=0)
    parser.add_argument('-groupby_var', help='''The grouping variable to harmonize on - site or study''')
    parser.add_argument('-outputfile', help='''Prefix for outputfiles''')
    parser.add_argument('-band',help='''What band to harmonize''')

    args = parser.parse_args()
    filename = args.input_file
    do_subparc = args.process_full_subparc
    harm_task = args.harmonize_task
    prefix = args.outputfile
    groupbyvar = args.groupby_var
    band = args.band
    
    harmonize(filename, do_subparc, harm_task, prefix, groupbyvar, band)
    
if __name__=='__main__':
    main()



