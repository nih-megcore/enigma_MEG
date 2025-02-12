#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 24 13:25:18 2024

@author: nugenta
"""
import numpy as np
import pandas as pd
import mne_bids
import argparse
import glob
import os
from fooof import FOOOF

def get_max_peak(results, minfreq, maxfreq, single_peak_only):
    
    try:
        potential_idx = np.where((minfreq <= results.peak_params[:,0] ) & \
                                                (results.peak_params[:,0] <= maxfreq ) )[0]
        if len(potential_idx) == 0:     # no peak in the given frequency range
            peak = np.nan 
            power = np.nan
            bandwidth = np.nan
        elif len(potential_idx) == 1:   # one peak in the given frequency range, easy case
            peak = results.peak_params[potential_idx[0]][0]
            power = results.peak_params[potential_idx[0]][1]
            bandwidth = results.peak_params[potential_idx[0]][2]
        else:   # more than one peak in the given frequency range: what to do? 
            if single_peak_only == False:   # if we allow more than one peak, find the biggest
                maxlist=[]
                for i in potential_idx:
                    maxlist.append(results.peak_params[i][1])
                max_idx = np.argmax(np.array(maxlist))
                max_peak_idx = potential_idx[max_idx]
                peak = results.peak_params[max_peak_idx][0]
                power = results.peak_params[max_peak_idx][1]
                bandwidth = results.peak_params[max_peak_idx][2]    
            else:   # sorry, only one peak allowed in the frequency range
                peak = np.nan
                power = np.nan
                bandwidth = np.nan
    except:
        peak = np.nan  # case where no peak is identified in range - set to np.nan
        power = np.nan
        bandwidth = np.nan

    return peak, power, bandwidth

if __name__=='__main__':

    # parse the arguments and initialize variables   

    parser = argparse.ArgumentParser()
    parser.add_argument('-power_file', help='''The name of the power file''')
    parser.add_argument('-spectra_file', help='''The name of the spectra file''')
    parser.add_argument('-output_file', help='''The name of the output file''')

    args = parser.parse_args()
    power_fname = args.power_file
    spectra_fname = args.spectra_file
    output_fname = args.output_file
    
    powerframe = pd.read_csv(power_fname, na_filter=False, dtype=str, sep=',')
    spectraframe = pd.read_csv(spectra_fname, na_filter=False, dtype=str, sep=',')

    if len(powerframe) != len(spectraframe):
        raise ValueError('Different number of entries in csv files, abort')
        
    freqsstring=spectraframe.columns.tolist()[1].split(',')
    freqs = np.array([float(string) for string in freqsstring]) 

    fm = FOOOF(verbose = False, aperiodic_mode = 'fixed', peak_width_limits= (1, 8),
                         min_peak_height = 0.05, max_n_peaks = 6, peak_threshold = 0.5)
        
    num_spectra = len(spectraframe)
  
    theta_peak = np.zeros(num_spectra)
    theta_power = np.zeros(num_spectra)
    theta_bandwidth = np.zeros(num_spectra)
    alpha_peak = np.zeros(num_spectra)
    alpha_power = np.zeros(num_spectra)
    alpha_bandwidth = np.zeros(num_spectra)
    beta_peak = np.zeros(num_spectra)
    beta_power = np.zeros(num_spectra)
    beta_bandwidth = np.zeros(num_spectra)
    gamma_peak = np.zeros(num_spectra)
    gamma_power = np.zeros(num_spectra)
    gamma_bandwidth = np.zeros(num_spectra)
    offset = np.zeros(num_spectra)
    exponent = np.zeros(num_spectra)
    peak_peak = np.zeros(num_spectra)
    peak_power = np.zeros(num_spectra)
    peak_bandwidth = np.zeros(num_spectra)
    R2 = np.zeros(num_spectra)
        
    for j in range(num_spectra):
        
        if j%448 == 0:
            print('Working on recording %d' % (j/448))
        
        psdstring = np.asarray(spectraframe.iloc[j].iloc[1].split(','))
        psd = np.array([float(string) for string in psdstring])
        fm.fit(freqs, psd, freq_range=[1, 45])
        results = fm.get_results()
            
        # get out the alpha peak param first for fixed - only accept if there is a single alpha peak
      
        theta_peak[j], theta_power[j], theta_bandwidth[j] = get_max_peak(results, 3.0, 6.0, True)
        alpha_peak[j], alpha_power[j], alpha_bandwidth[j] = get_max_peak(results, 8.0, 12.0, False)
        beta_peak[j], beta_power[j], beta_bandwidth[j] = get_max_peak(results, 13.0, 35.0, False)
        gamma_peak[j], gamma_power[j], gamma_bandwidth[j] = get_max_peak(results, 35.0, 45.0, False)
        peak_peak[j], peak_power[j], peak_bandwidth[j] = get_max_peak(results, 1.0, 45.0, False)
        
        offset[j] = results.aperiodic_params[0]
        exponent[j] = results.aperiodic_params[1]
            
        R2[j] = results.r_squared
        
    print('found %d total theta peaks' % np.sum(~np.isnan(theta_peak)))
    print('found %d total alpha peaks' % np.sum(~np.isnan(alpha_peak)))
    print('found %d total beta peaks' % np.sum(~np.isnan(beta_peak)))
    print('found %d total gamma peaks' % np.sum(~np.isnan(gamma_peak)))
     
    output_dframe = pd.DataFrame()
    
    # add in all the columns from the power dataframe
   
    output_dframe['subject'] = powerframe['subject']
    output_dframe['Parcel'] = powerframe['Parcel']
    output_dframe['age'] = powerframe['age']
    output_dframe['sex'] = powerframe['sex']
    output_dframe['ses'] = powerframe['ses']
    output_dframe['run'] = powerframe['run']
    output_dframe['hand'] = powerframe['hand']
    output_dframe['task'] = powerframe['task']
    output_dframe['group'] = powerframe['group']
    output_dframe['scanner'] = powerframe['scanner']
    output_dframe['eTIV'] = powerframe['eTIV']
    output_dframe['lh_holes'] = powerframe['lh_holes']
    output_dframe['rh_holes'] = powerframe['rh_holes']
    output_dframe['avg_holes'] = powerframe['avg_holes']
    output_dframe['epochs_orig'] = powerframe['epochs_orig']
    output_dframe['epochs_final'] = powerframe['epochs_final']
    output_dframe['total_time'] = powerframe['total_time']   
    output_dframe['QA_final'] = powerframe['QA_FINAL']
        
    # add in all the columns we just calculated
   
    output_dframe['ThetaPeak'] = theta_peak
    output_dframe['Theta_power'] = theta_power
    output_dframe['Theta_bandwidth'] = theta_bandwidth
    output_dframe['AlphaPeak'] = alpha_peak
    output_dframe['Alpha_power'] = alpha_power
    output_dframe['Alpha_bandwidth'] = alpha_bandwidth
    output_dframe['BetaPeak'] = beta_peak
    output_dframe['Beta_power'] = beta_power
    output_dframe['Beta_bandwidth'] = beta_bandwidth
    output_dframe['GammaPeak'] = gamma_peak
    output_dframe['Gamma_power'] = gamma_power
    output_dframe['Gamma_bandwidth'] = gamma_bandwidth
    output_dframe['PeakPeak'] = peak_peak
    output_dframe['Peak_power'] = peak_power
    output_dframe['Peak_bandwidth'] = peak_bandwidth
    output_dframe['AperiodicOffset'] = offset
    output_dframe['AperiodicExponent'] = exponent
    output_dframe['R2'] = R2
        
    output_dframe.to_csv(output_fname, sep=',')  

        
        