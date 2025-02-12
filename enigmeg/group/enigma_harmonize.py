#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  6 09:21:19 2024

@author: nugenta
"""

import pandas as pd
import numpy as np
import scipy as sp
import scipy.stats as stats
import neuroCovHarmonize
from neuroCovHarmonize.harmonizationLearn import harmonizationCovLearn, saveHarmonizationModel, saveHarmonizationModelNeuroCombat
import neuroHarmonize
from neuroHarmonize import harmonizationLearn
import statsmodels.api as sm
from statsmodels.formula.api import ols
from sklearn.preprocessing import PolynomialFeatures
import statsmodels.formula.api as smf
from neuroCombat.neuroCombat import neuroCombat

reliefSource = '/home/nugenta/src/RELIEF.R'
import rpy2
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
from rpy2.robjects.vectors import StrVector
import rpy2.robjects.packages as rpackages

import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()

## import R packages
base = importr('base')
utils = importr('utils')

denoiseR = importr('denoiseR')
MASS = importr('MASS')
Matrix = importr('Matrix')

## source the RELIEF functions from local .R file
r = robjects.r
r.source(reliefSource)   ## path to the local file

## bring the function into python
relief = robjects.globalenv['relief']
# print(relief.r_repr())

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
    
    delta = dataframe.pivot(index=['subject','task'], columns='Parcel', values='[1, 3]').reset_index()
    theta = dataframe.pivot(index=['subject','task'], columns='Parcel', values='[3, 6]').reset_index()
    alpha = dataframe.pivot(index=['subject','task'], columns='Parcel', values='[8, 12]').reset_index()
    beta = dataframe.pivot(index=['subject','task'], columns='Parcel', values='[13, 35]').reset_index()
    gamma = dataframe.pivot(index=['subject','task'], columns='Parcel', values='[35, 45]').reset_index()
    
    delta = delta.drop(columns='task')
    theta = theta.drop(columns='task')
    alpha = alpha.drop(columns='task')
    beta = beta.drop(columns='task')
    gamma = gamma.drop(columns='task')
    
    return delta, theta, alpha, beta, gamma

def prepare_to_harmonize(dataframe, single, num_parcels):
    
    dataframe['age'] = single['age']
    dataframe['sex'] = single['sex']
    dataframe['site'] = single['site']
    dataframe['task'] = single['task']
    dataframe['study'] = single['study']
    
    data_to_harmonize = dataframe.iloc[:, 1:(num_parcels+1)].to_numpy()
    
    # also add age bins for later
    age_bins = [0, 20, 40, 60, 120]  # 0-20, 20-40, 40-60, 60-80
    age_labels = ['0-20', '20-40', '40-60', '60up']

    # Create a new column with the age bins
    dataframe['age_group'] = pd.cut(dataframe['age'], bins=age_bins, labels=age_labels, right=False)
    
    return dataframe, data_to_harmonize

def process_adjusted(data, tmpidx, single, num_parcels):
        
    data_dframe = pd.DataFrame(data, columns=tmpidx[1:(num_parcels+1)])
    data_dframe['subject'] = single['subject']
    data_dframe['age'] = single['age']
    data_dframe['sex'] = single['sex']
    data_dframe['site'] = single['site']
    data_dframe['task'] = single['task']
    data_dframe['study'] = single['study']
        
    # add age bins
    age_bins = [0, 20, 40, 60, 120]  # 0-20, 20-40, 40-60, 60-80
    age_labels = ['0-20', '20-40', '40-60', '60up']

    # Create a new column with the age bins
    data_dframe['age_group'] = pd.cut(data_dframe['age'], bins=age_bins, labels=age_labels, right=False)

    return data_dframe

def process_adjusted_covbat(data, tmpidx, single, num_parcels):
    
    dataframes = []    
    numframes = np.shape(data)[2]
    
    # add age bins
    age_bins = [0, 20, 40, 60, 120]  # 0-20, 20-40, 40-60, 60-80
    age_labels = ['0-20', '20-40', '40-60', '60up']
    
    for i in range(numframes):
    
        data_dframe = pd.DataFrame(np.squeeze(data[:,:,i]), columns=tmpidx[1:(num_parcels+1)])
        data_dframe['subject'] = single['subject']
        data_dframe['age'] = single['age']
        data_dframe['sex'] = single['sex']
        data_dframe['site'] = single['site']
        data_dframe['task'] = single['task']
        data_dframe['study'] = single['study']
        data_dframe['age_group'] = pd.cut(data_dframe['age'], bins=age_bins, labels=age_labels, right=False)
        
        dataframes.append(data_dframe)

    return dataframes

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
    parser.add_argument('-outputfile', help='''Prefix for outputfiles''')

    args = parser.parse_args()
    filename = args.input_file
    do_subparc = args.process_full_subparc
    harm_task = args.harmonize_task
    prefix = args.outputfile
    
    if do_subparc == True:
        num_parcels = 448
    else:
        num_parcels = 68
        
    dataframe = pd.read_csv(filename)
    print('Dataframe read, number of subjects is %d' % (int(len(dataframe)/448)))
    
    dataframe = dataframe.dropna(subset='age').reset_index()
    print('Age NaN values dropped, number of subjects is %d' % (int(len(dataframe)/448)))
    
    # extract column headings 
    dataframe['study'] = dataframe['study'].astype('category')
    dataframe['site'] = dataframe['site'].astype('category')  
    dataframe['sex'] = dataframe['sex'].astype('category')
    dataframe['task'] = dataframe['task'].astype('category')
    
    dataframe=dataframe.sort_values(['subject','task'])
    
    delta, theta, alpha, beta, gamma = make_power_dataframes(dataframe)
    
    single = dataframe[dataframe['Parcel'] == 'bankssts_1-lh'].reset_index()
    num_subjects = len(single)
    
    # combine the subparcellation columns
    
    if do_subparc == False: 
        delta = combine_columns(delta)
        theta = combine_columns(theta)
        alpha = combine_columns(alpha)
        beta = combine_columns(beta)
        gamma = combine_columns(gamma)
    
    # prepare the datasets for harmonization
    
    delta, delta_data = prepare_to_harmonize(delta, single, num_parcels)
    theta, theta_data = prepare_to_harmonize(theta, single, num_parcels)
    alpha, alpha_data = prepare_to_harmonize(alpha, single, num_parcels)
    beta, beta_data = prepare_to_harmonize(beta, single, num_parcels)
    gamma, gamma_data = prepare_to_harmonize(gamma, single, num_parcels)

    if (harm_task == False):
        covars = pd.DataFrame.from_dict({'SITE': single.study.cat.codes,
                                     'age': single.age.values,
                                     'sex': single.sex.cat.codes,
                                     'age2': single.age.values*single.age.values,
                                     'age3': single.age.values*single.age.values*single.age.values})
    else:
        covars = pd.DataFrame.from_dict({'SITE': single.study.cat.codes,
                                     'age': single.age.values,
                                     'sex': single.sex.cat.codes,
                                     'task': single.task.cat.codes,
                                     'age2': single.age.values*single.age.values,
                                     'age3': single.age.values*single.age.values*single.age.values})
    
    tmpidx = delta.columns.to_flat_index().tolist()
    
    studyid = np.array(single.study.cat.codes)
    
    # perform the harmonizations
  
    # oh what the hell, let's add one more, let's do RELIEF too

    delta_data_r = robjects.r['matrix'](delta_data.T, num_parcels, num_subjects)
    theta_data_r = robjects.r['matrix'](theta_data.T, num_parcels, num_subjects)
    alpha_data_r = robjects.r['matrix'](alpha_data.T, num_parcels, num_subjects)
    beta_data_r = robjects.r['matrix'](beta_data.T, num_parcels, num_subjects)
    gamma_data_r = robjects.r['matrix'](gamma_data.T, num_parcels, num_subjects)

    studyid_r = robjects.IntVector(studyid)
        
    output_delta = relief(delta_data_r, studyid_r)
    output_theta = relief(theta_data_r, studyid_r)
    output_alpha = relief(alpha_data_r, studyid_r)
    output_beta = relief(beta_data_r, studyid_r)
    output_gamma = relief(gamma_data_r, studyid_r)
        
    delta_dat_relief = output_delta.rx2['dat.relief']
    theta_dat_relief = output_theta.rx2['dat.relief']
    alpha_dat_relief = output_alpha.rx2['dat.relief']
    beta_dat_relief = output_beta.rx2['dat.relief']
    gamma_dat_relief = output_gamma.rx2['dat.relief']
        
    delta_adjusted_relief_dframe = process_adjusted(delta_dat_relief.T, tmpidx, single, num_parcels)
    theta_adjusted_relief_dframe = process_adjusted(theta_dat_relief.T, tmpidx, single, num_parcels)
    alpha_adjusted_relief_dframe = process_adjusted(alpha_dat_relief.T, tmpidx, single, num_parcels)
    beta_adjusted_relief_dframe = process_adjusted(beta_dat_relief.T, tmpidx, single, num_parcels)
    gamma_adjusted_relief_dframe = process_adjusted(gamma_dat_relief.T, tmpidx, single, num_parcels)

    if do_subparc == False:
            delta_adjusted_relief_dframe.to_csv(f'{prefix}_ReliefAdjusted_delta_dataframe.csv')
            theta_adjusted_relief_dframe.to_csv(f'{prefix}_ReliefAdjusted_theta_dataframe.csv')
            alpha_adjusted_relief_dframe.to_csv(f'{prefix}_ReliefAdjusted_alpha_dataframe.csv')
            beta_adjusted_relief_dframe.to_csv(f'{prefix}_ReliefAdjusted_beta_dataframe.csv')
            gamma_adjusted_relief_dframe.to_csv(f'{prefix}_ReliefAdjusted_gamma_dataframe.csv')
    else:
            delta_adjusted_relief_dframe.to_csv(f'{prefix}_ReliefAdjusted_delta_dataframe_subparc.csv')
            theta_adjusted_relief_dframe.to_csv(f'{prefix}_ReliefAdjusted_theta_dataframe_subparc.csv')
            alpha_adjusted_relief_dframe.to_csv(f'{prefix}_ReliefAdjusted_alpha_dataframe_subparc.csv')
            beta_adjusted_relief_dframe.to_csv(f'{prefix}_ReliefAdjusted_beta_dataframe_subparc.csv')
            gamma_adjusted_relief_dframe.to_csv(f'{prefix}_ReliefAdjusted_gamma_dataframe_subparc.csv')

    # First - just do ComBat normalization, inculding age2 as a covariate
    if (harm_task == False):
        output_combat_delta = neuroCombat(delta_data.T, covars, 'SITE', ['sex'], eb=True)
        output_combat_theta = neuroCombat(theta_data.T, covars, 'SITE', ['sex'], eb=True)
        output_combat_alpha = neuroCombat(alpha_data.T, covars, 'SITE', ['sex'], eb=True)
        output_combat_beta = neuroCombat(beta_data.T, covars, 'SITE', ['sex'], eb=True)
        output_combat_gamma = neuroCombat(gamma_data.T, covars, 'SITE', ['sex'], eb=True)
        
    else:
            output_combat_delta = neuroCombat(delta_data.T, covars, 'SITE', ['sex','task'], eb=True)
            output_combat_theta = neuroCombat(theta_data.T, covars, 'SITE', ['sex','task'], eb=True)
            output_combat_alpha = neuroCombat(alpha_data.T, covars, 'SITE', ['sex','task'], eb=True)
            output_combat_beta = neuroCombat(beta_data.T, covars, 'SITE', ['sex','task'], eb=True)
            output_combat_gamma = neuroCombat(gamma_data.T, covars, 'SITE', ['sex','task'], eb=True)

    delta_adjusted_combat_dframe = process_adjusted(output_combat_delta['data'].T, tmpidx, single, num_parcels)
    theta_adjusted_combat_dframe = process_adjusted(output_combat_theta['data'].T, tmpidx, single, num_parcels)
    alpha_adjusted_combat_dframe = process_adjusted(output_combat_alpha['data'].T, tmpidx, single, num_parcels)
    beta_adjusted_combat_dframe = process_adjusted(output_combat_beta['data'].T, tmpidx, single, num_parcels)
    gamma_adjusted_combat_dframe = process_adjusted(output_combat_gamma['data'].T, tmpidx, single, num_parcels)
   
    if do_subparc == False:
        delta_adjusted_combat_dframe.to_csv(f'{prefix}_ComBatAdjusted_delta_dataframe.csv')
        theta_adjusted_combat_dframe.to_csv(f'{prefix}_ComBatAdjusted_theta_dataframe.csv')
        alpha_adjusted_combat_dframe.to_csv(f'{prefix}_ComBatAdjusted_alpha_dataframe.csv')
        beta_adjusted_combat_dframe.to_csv(f'{prefix}_ComBatAdjusted_beta_dataframe.csv')
        gamma_adjusted_combat_dframe.to_csv(f'{prefix}_ComBatAdjusted_gamma_dataframe.csv')
    else:
        delta_adjusted_combat_dframe.to_csv(f'{prefix}_ComBatAdjusted_delta_dataframe_subparc.csv')
        theta_adjusted_combat_dframe.to_csv(f'{prefix}_ComBatAdjusted_theta_dataframe_subparc.csv')
        alpha_adjusted_combat_dframe.to_csv(f'{prefix}_ComBatAdjusted_alpha_dataframe_subparc.csv')
        beta_adjusted_combat_dframe.to_csv(f'{prefix}_ComBatAdjusted_beta_dataframe_subparc.csv')
        gamma_adjusted_combat_dframe.to_csv(f'{prefix}_ComBatAdjusted_gamma_dataframe_subparc.csv')

    if (harm_task == False):
         covars = pd.DataFrame.from_dict({'SITE': single.study.cat.codes,
                                      'age': single.age.values,
                                      'sex': single.sex.cat.codes})
    else:
         covars = pd.DataFrame.from_dict({'SITE': single.study.cat.codes,
                                      'age': single.age.values,
                                      'sex': single.sex.cat.codes,
                                      'task': single.task.cat.codes})
         
   # Second - do ComBatGam neuroHarmonize
   
    if (harm_task == False):
       
       print('Harmonizing Delta')
       delta_model, delta_adjusted_gamcombat = harmonizationLearn(delta_data,
                                       covars, smooth_terms=['age','sex'])
       print('Harmonizing Theta')
       theta_model, theta_adjusted_gamcombat = harmonizationLearn(np.array(theta_data),
                                        covars, smooth_terms=['age','sex'])
       print('Harmonizing Alpha')
       alpha_model, alpha_adjusted_gamcombat = harmonizationLearn(np.array(alpha_data),
                                       covars, smooth_terms=['age','sex'])
       print('Harmonizing Beta')
       beta_model, beta_adjusted_gamcombat = harmonizationLearn(np.array(beta_data),
                                        covars, smooth_terms=['age','sex'])
       print('Harmonizing Gamma')
       gamma_model, gamma_adjusted_gamcombat = harmonizationLearn(np.array(gamma_data),
                                       covars, smooth_terms=['age','sex'])
       
    else:
       
       print('Harmonizing Delta')
       delta_model, delta_adjusted_gamcombat = harmonizationLearn(delta_data,
                                       covars, smooth_terms=['age','sex','task'])
       print('Harmonizing Theta')
       theta_model, theta_adjusted_gamcombat = harmonizationLearn(np.array(theta_data),
                                        covars, smooth_terms=['age','sex','task'])
       print('Harmonizing Alpha')
       alpha_model, alpha_adjusted_gamcombat = harmonizationLearn(np.array(alpha_data),
                                       covars, smooth_terms=['age','sex','task'])
       print('Harmonizing Beta')
       beta_model, beta_adjusted_gamcombat = harmonizationLearn(np.array(beta_data),
                                        covars, smooth_terms=['age','sex','task'])
       print('Harmonizing Gamma')
       gamma_model, gamma_adjusted_gamcombat = harmonizationLearn(np.array(gamma_data),
                                       covars, smooth_terms=['age','sex','task'])
 
    delta_adjusted_gamcombat_dframe = process_adjusted(delta_adjusted_gamcombat, tmpidx, single, num_parcels)
    theta_adjusted_gamcombat_dframe = process_adjusted(theta_adjusted_gamcombat, tmpidx, single, num_parcels)
    alpha_adjusted_gamcombat_dframe = process_adjusted(alpha_adjusted_gamcombat, tmpidx, single, num_parcels)
    beta_adjusted_gamcombat_dframe = process_adjusted(beta_adjusted_gamcombat, tmpidx, single, num_parcels)
    gamma_adjusted_gamcombat_dframe = process_adjusted(gamma_adjusted_gamcombat, tmpidx, single, num_parcels)
    
    if do_subparc == False:
        delta_adjusted_gamcombat_dframe.to_csv(f'{prefix}_gamComBatAdjusted_delta_dataframe.csv')
        theta_adjusted_gamcombat_dframe.to_csv(f'{prefix}_gamComBatAdjusted_theta_dataframe.csv')
        alpha_adjusted_gamcombat_dframe.to_csv(f'{prefix}_gamComBatAdjusted_alpha_dataframe.csv')
        beta_adjusted_gamcombat_dframe.to_csv(f'{prefix}_gamComBatAdjusted_beta_dataframe.csv')
        gamma_adjusted_gamcombat_dframe.to_csv(f'{prefix}_gamComBatAdjusted_gamma_dataframe.csv')
    else:
        delta_adjusted_gamcombat_dframe.to_csv(f'{prefix}_gamComBatAdjusted_delta_dataframe_subparc.csv')
        theta_adjusted_gamcombat_dframe.to_csv(f'{prefix}_gamComBatAdjusted_theta_dataframe_subparc.csv')
        alpha_adjusted_gamcombat_dframe.to_csv(f'{prefix}_gamComBatAdjusted_alpha_dataframe_subparc.csv')
        beta_adjusted_gamcombat_dframe.to_csv(f'{prefix}_gamComBatAdjusted_beta_dataframe_subparc.csv')
        gamma_adjusted_gamcombat_dframe.to_csv(f'{prefix}_gamComBatAdjusted_gamma_dataframe_subparc.csv')

    if do_subparc == False:
        saveHarmonizationModel(delta_model, f'{prefix}_delta_neuroharmonizemodel')
        saveHarmonizationModel(theta_model, f'{prefix}_theta_neuroharmonizemodel')
        saveHarmonizationModel(alpha_model, f'{prefix}_alpha_neuroharmonizemodel')
        saveHarmonizationModel(beta_model, f'{prefix}_beta_neuroharmonizemodel')
        saveHarmonizationModel(gamma_model, f'{prefix}_gamma_neuroharmonizemodel')    
    else:
        saveHarmonizationModel(delta_model, f'{prefix}_delta_neuroharmonizemodel_subparc')
        saveHarmonizationModel(theta_model, f'{prefix}_theta_neuroharmonizemodel_subparc')
        saveHarmonizationModel(alpha_model, f'{prefix}_alpha_neuroharmonizemodel_subparc')
        saveHarmonizationModel(beta_model, f'{prefix}_beta_neuroharmonizemodel_subparc')
        saveHarmonizationModel(gamma_model, f'{prefix}_gamma_neuroharmonizemodel_subparc')    

    # Now, finally, do ComBatGam Harmonization + CovBat
         
    if (harm_task == False):
        
        print('Harmonizing Delta')
        delta_model, delta_level2model, delta_adjusted, delta_adjusted_combat = harmonizationCovLearn(delta_data,
                                        covars, smooth_terms=['age','sex'], pct_var=[0.9,0.95,1] )
        print('Harmonizing Theta')
        theta_model, theta_level2model, theta_adjusted, theta_adjusted_combat = harmonizationCovLearn(np.array(theta_data),
                                         covars, smooth_terms=['age','sex'], pct_var=[0.9,0.95,1])
        print('Harmonizing Alpha')
        alpha_model, alpha_level2model, alpha_adjusted, alpha_adjusted_combat = harmonizationCovLearn(np.array(alpha_data),
                                        covars, smooth_terms=['age','sex'], pct_var=[0.9,0.95,1])
        print('Harmonizing Beta')
        beta_model, beta_level2model, beta_adjusted, beta_adjusted_combat = harmonizationCovLearn(np.array(beta_data),
                                         covars, smooth_terms=['age','sex'], pct_var=[0.9,0.95,1])
        print('Harmonizing Gamma')
        gamma_model, gamma_level2model, gamma_adjusted, gamma_adjusted_combat = harmonizationCovLearn(np.array(gamma_data),
                                        covars, smooth_terms=['age','sex'], pct_var=[0.9,0.95,1])
        
    else:
        
        print('Harmonizing Delta')
        delta_model, delta_level2model, delta_adjusted, delta_adjusted_combat = harmonizationCovLearn(delta_data,
                                        covars, smooth_terms = ['age', 'sex', 'task'], pct_var=[0.9,0.95,1])
        print('Harmonizing Theta')
        theta_model, theta_level2model, theta_adjusted, theta_adjusted_combat = harmonizationCovLearn(np.array(theta_data),
                                         covars, smooth_terms = ['age', 'sex', 'task'], pct_var=[0.9,0.95,1])
        print('Harmonizing Alpha')
        alpha_model, alpha_level2model, alpha_adjusted, alpha_adjusted_combat = harmonizationCovLearn(np.array(alpha_data),
                                        covars, smooth_terms = ['age', 'sex', 'task'], pct_var=[0.9,0.95,1])
        print('Harmonizing Beta')
        beta_model, beta_level2model, beta_adjusted, beta_adjusted_combat = harmonizationCovLearn(np.array(beta_data),
                                         covars, smooth_terms = ['age', 'sex', 'task'], pct_var=[0.9,0.95,1])
        print('Harmonizing Gamma')
        gamma_model, gamma_level2model, gamma_adjusted, gamma_adjusted_combat = harmonizationCovLearn(np.array(gamma_data),
                                        covars, smooth_terms = ['age', 'sex', 'task'], pct_var=[0.9,0.95,1])

    # fix up the adjusted data
        
    delta_adjusted_dframe = process_adjusted_covbat(delta_adjusted, tmpidx, single, num_parcels)
    theta_adjusted_dframe = process_adjusted_covbat(theta_adjusted, tmpidx, single, num_parcels)
    alpha_adjusted_dframe = process_adjusted_covbat(alpha_adjusted, tmpidx, single, num_parcels)
    beta_adjusted_dframe = process_adjusted_covbat(beta_adjusted, tmpidx, single, num_parcels)
    gamma_adjusted_dframe = process_adjusted_covbat(gamma_adjusted, tmpidx, single, num_parcels)
        
    # output the adjusted dataframes
    
    pct_var = [0.9,0.95,1]
    
    varidx = 0
    for var in pct_var:
    
        if do_subparc == False:
            delta_adjusted_dframe[varidx].to_csv(f'{prefix}_CovBatAdjusted{var}_delta_dataframe.csv')
            theta_adjusted_dframe[varidx].to_csv(f'{prefix}_CovBatAdjusted{var}_theta_dataframe.csv')
            alpha_adjusted_dframe[varidx].to_csv(f'{prefix}_CovBatAdjusted{var}_alpha_dataframe.csv')
            beta_adjusted_dframe[varidx].to_csv(f'{prefix}_CovBatAdjusted{var}_beta_dataframe.csv')
            gamma_adjusted_dframe[varidx].to_csv(f'{prefix}_CovBatAdjusted{var}_gamma_dataframe.csv')

            saveHarmonizationModelNeuroCombat(delta_level2model[varidx], f'{prefix}_delta_level2neurocombatmodel{var}')
            saveHarmonizationModelNeuroCombat(theta_level2model[varidx], f'{prefix}_theta_level2neurocombatmodel{var}')
            saveHarmonizationModelNeuroCombat(alpha_level2model[varidx], f'{prefix}_alpha_level2neurocombatmodel{var}')
            saveHarmonizationModelNeuroCombat(beta_level2model[varidx], f'{prefix}_beta_level2neurocombatmodel{var}')
            saveHarmonizationModelNeuroCombat(gamma_level2model[varidx], f'{prefix}_gamma_level2neurocombatmodel{var}')

        else:
            delta_adjusted_dframe[varidx].to_csv(f'{prefix}_CovBatAdjusted_delta_dataframe_subparc.csv')
            theta_adjusted_dframe[varidx].to_csv(f'{prefix}_CovBatAdjusted_theta_dataframe_subparc.csv')
            alpha_adjusted_dframe[varidx].to_csv(f'{prefix}_CovBatAdjusted_alpha_dataframe_subparc.csv')
            beta_adjusted_dframe[varidx].to_csv(f'{prefix}_CovBatAdjusted_beta_dataframe_subparc.csv')
            gamma_adjusted_dframe[varidx].to_csv(f'{prefix}_CovBatAdjusted_gamma_dataframe_subparc.csv')    

            saveHarmonizationModelNeuroCombat(delta_level2model[varidx], f'{prefix}_delta_level2neurocombatmodel{var}_subparc')
            saveHarmonizationModelNeuroCombat(theta_level2model[varidx], f'{prefix}_theta_level2neurocombatmodel{var}_subparc')
            saveHarmonizationModelNeuroCombat(alpha_level2model[varidx], f'{prefix}_alpha_level2neurocombatmodel{var}_subparc')
            saveHarmonizationModelNeuroCombat(beta_level2model[varidx], f'{prefix}_beta_level2neurocombatmodel{var}_subparc')
            saveHarmonizationModelNeuroCombat(gamma_level2model[varidx], f'{prefix}_gamma_level2neurocombatmodel{var}_subparc')
          
        varidx += 1    
        
    if do_subparc == False:
        delta.to_csv(f'{prefix}_Orig_delta_dataframe.csv')
        theta.to_csv(f'{prefix}_Orig_theta_dataframe.csv')
        alpha.to_csv(f'{prefix}_Orig_alpha_dataframe.csv')
        beta.to_csv(f'{prefix}_Orig_beta_dataframe.csv')
        gamma.to_csv(f'{prefix}_Orig_gamma_dataframe.csv')
    else:
        delta.to_csv(f'{prefix}_Orig_delta_dataframe_subparc.csv')
        theta.to_csv(f'{prefix}_Orig_theta_dataframe_subparc.csv')
        alpha.to_csv(f'{prefix}_Orig_alpha_dataframe_subparc.csv')
        beta.to_csv(f'{prefix}_Orig_beta_dataframe_subparc.csv')
        gamma.to_csv(f'{prefix}_Orig_gamma_dataframe_subparc.csv')

    if do_subparc == False:
        saveHarmonizationModel(delta_model, f'{prefix}_delta_neurocov_neuroharmonizemodel')
        saveHarmonizationModel(theta_model, f'{prefix}_theta_neurocov_neuroharmonizemodel')
        saveHarmonizationModel(alpha_model, f'{prefix}_alpha_neurocov_neuroharmonizemodel')
        saveHarmonizationModel(beta_model, f'{prefix}_beta_neurocov_neuroharmonizemodel')
        saveHarmonizationModel(gamma_model, f'{prefix}_gamma_neurocov_neuroharmonizemodel')    
    
    else:
        saveHarmonizationModel(delta_model, f'{prefix}_delta_neurocov_neuroharmonizemodel_subparc')
        saveHarmonizationModel(theta_model, f'{prefix}_theta_neurocov_neuroharmonizemodel_subparc')
        saveHarmonizationModel(alpha_model, f'{prefix}_alpha_neurocov_neuroharmonizemodel_subparc')
        saveHarmonizationModel(beta_model, f'{prefix}_beta_neurohcov_neuroharmonizemodel_subparc')
        saveHarmonizationModel(gamma_model, f'{prefix}_gamma_neurocov_neuroharmonizemodel_subparc')    

if __name__=='__main__':
    main()



