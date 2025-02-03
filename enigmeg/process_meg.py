#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: Jeff Stout and Allison Nugent
"""
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)

import os
import os.path as op
import sys
import mne
import re
import glob
import numpy as np
import pandas as pd
import enigmeg
from enigmeg.spectral_peak_analysis import calc_spec_peak
from enigmeg.QA.enigma_QA_GUI_functions import build_status_dict
from enigmeg.utils import get_head_mri_trans_bti
from enigmeg import mod_label_extract
import logging
import munch 
import subprocess
import mne_bids
from mne_bids import get_head_mri_trans
from mne.beamformer import make_dics, apply_dics_csd
from mne.beamformer import make_lcmv, apply_lcmv_epochs
from mne.bem import FIFF
import scipy as sp
from mne_bids import BIDSPath
import functools
from scipy.stats import zscore, trim_mean
from mne.preprocessing import maxwell_filter
from io import StringIO

# Set tensorflow to use CPU
# The import is performed in the sub-functions to delay the unsuppressable warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Define some MNE settings
os.environ['MNE_3D_OPTION_ANTIALIAS'] = 'false' # necessary for headless operation


# define some variables

fmin = 1
fmax = 45
sfreq = 300
epoch_len = 4.0
mt_bandwidth = 2 # bandwidth for multitaper
n_bins = 177

# parameters for rejecting bad epochs
magthresh = 5000e-15
gradthresh = 5000e-13
flatmagthresh = 10e-15
flatgradthresh = 10e-13
std_thresh = 15


# Make a string buffer logger before establishing final log filename
  
global log_dir
logger = logging.getLogger()
logger.setLevel(logging.INFO)
buffer_logstream = StringIO('')
ch = logging.StreamHandler(stream=buffer_logstream)
logger.addHandler(ch)

# Function to retrieve the subject/session specific logger

def get_subj_logger(subjid, session, task, run, log_dir=None):
     '''Return the subject specific logger.
     This is particularly useful in the multiprocessing where logging is not
     necessarily in order'''
     fmt = '%(asctime)s :: %(levelname)s :: %(message)s'
     sub_ses = f'{subjid}_ses_{session}_task_{task}_run_{run}'
     subj_logger = logging.getLogger(sub_ses)
     if subj_logger.handlers != []: # if not first time requested, use the file handler already defined
         tmp_ = [type(i) for i in subj_logger.handlers ]
         if logging.FileHandler in tmp_:
             return subj_logger
     else: # first time requested, add the file handler
         fileHandle = logging.FileHandler(f'{log_dir}/{subjid}_ses-{session}_task-{task}_run-{run}_log.txt')
         fileHandle.setLevel(logging.INFO)
         fileHandle.setFormatter(logging.Formatter(fmt)) 
         subj_logger.addHandler(fileHandle)
         subj_logger.info('Initializing subject level enigma log')
     return subj_logger   


#Decorator for logging functions

def log(function):
    def wrapper(*args, **kwargs):  
        logger.info(f"{function.__name__} :: START")
        try:
            output = function(*args, **kwargs)
        except BaseException as e:
            logger.exception(f"{function.__name__} :: " + str(e))
            raise
        logger.info(f"{function.__name__} :: COMPLETED")
        return output
    return wrapper


# define a class that holds all the information about a single subject/session/run dataset for processing

class process():
    def __init__(
            self, 
            subject=None, 
            bids_root=None, 
            deriv_root=None,
            subjects_dir=None,
            rest_tagname='rest',
            emptyroom_tagname='emptyroom',
            emptyroom_run=None,
            session='1', 
            mains=60,
            run='1',
            t1_override=None,
            megin_ignore=None,
            fs_ave_fids=False, 
            check_paths=True,
            do_dics=False,
            csv_info=None,
            ):
        
# =============================================================================
#         # Initialize variables and directories
# =============================================================================
        
        # Establish subject level logger and flush string buffer into file
        log_dir = f'{bids_root}/derivatives/ENIGMA_MEG/logs' 
        if not op.exists(log_dir): os.makedirs(log_dir)
        _buffer = None
        global logger
        if len(logger.handlers) > 0:
            if hasattr(logger.handlers[0], 'stream.getvalue'):
                _buffer = logger.handlers[0].stream.getvalue()
        logger = get_subj_logger(subject, session, rest_tagname, run, log_dir)
        if _buffer != None:
            logger.info(_buffer)
        
        self.subject=subject.replace('sub-','')  # Strip sub- if present
        self.session = session
        self.run = run
        self.bids_root=bids_root
        if deriv_root is None:                   # Derivatives directory
            self.deriv_root = op.join(
                bids_root, 
                'derivatives'
                )
        else:
            self.deriv_root = deriv_root
            
        self.enigma_root = op.join( # enigma output directory
            self.deriv_root,
            'ENIGMA_MEG'
            )
        
        if subjects_dir is None:    # Freesurfer subjects directory
            self.subjects_dir = op.join(
                self.deriv_root,
                'freesurfer',
                'subjects'
                )
        else:
            self.subjects_dir = subjects_dir
        self.proc_vars=munch.Munch()    # set some parameters
        self.proc_vars['fmin'] = fmin
        self.proc_vars['fmax'] = fmax
        self.proc_vars['sfreq'] = sfreq
        self.proc_vars['mains'] = mains
        self.proc_vars['epoch_len']=epoch_len  #seconds
        
        # t1_override is used when you are processing a single subject (so csv_info=None) but the 
        # name of the anatomical does not follow the same subj/ses/run hierarchy as the meg
        
        self._t1_override = t1_override   
        if fs_ave_fids==True:
            self._use_fsave_coreg=True
        else:
            self._use_fsave_coreg=False
            
        self._megin_ignore = megin_ignore
        
        try:
            #The commandline args have been assigned into os.environ['n_jobs']
            #so it will be used here if set
            self._n_jobs = int(os.environ['n_jobs'])
        except:
            self._n_jobs = 1
        
        self.do_dics = do_dics
            
# =============================================================================
#             Configure paths and filenames
# =============================================================================

        self.bids_path = BIDSPath(
            root=bids_root, 
            subject=subject, 
            session=session)
        
        self.deriv_path=self.bids_path.copy().update(
            root=self.enigma_root,
            check=False,
            extension='.fif',
            datatype='meg', 
            suffix='meg'
            )
        
        # QA output directory
        QA_root = op.join( # enigma output directory
            self.deriv_root,
            'ENIGMA_MEG_QA'
            )
        self.QA_dir = self.deriv_path.copy().update(root=QA_root,
                                            datatype='meg', 
                                            extension=None, 
                                            suffix='QA')
        os.makedirs(self.QA_dir.directory, exist_ok=True)
        
        if not op.exists(self.deriv_path.directory): 
            self.deriv_path.directory.mkdir(parents=True)
        
        self.rest_derivpath = self.deriv_path.copy().update(
            task=rest_tagname, 
            run=run
            )
        
        if emptyroom_tagname == None: # and not csv_info['eroom']:
            self.eroom_derivpath = None
        else:
            self.eroom_derivpath = self.deriv_path.copy().update(
                task=emptyroom_tagname,
                run=emptyroom_run
                )
        
        self.meg_rest_raw = self.bids_path.copy().update(
            datatype='meg', 
            task=rest_tagname, 
            run=run
            )
        
        if emptyroom_tagname == None: # and not csv_info['eroom']:
            self.meg_er_raw = None
        else:
            self.meg_er_raw = self.bids_path.copy().update(
                datatype='meg',
                task=emptyroom_tagname, 
                run=emptyroom_run
                )
        
        if(self._t1_override == None):
            self.anat_bidspath = self.bids_path.copy().update(root=self.subjects_dir,
                                                          session=None,
                                                          check=False)
        else:
            self.anat_bidspath = mne_bids.get_bids_path_from_fname(self._t1_override)
        
        self.fnames=self.initialize_fnames(rest_tagname, emptyroom_tagname)
        if check_paths:
            self.check_paths()

            
        self.anat_vars=munch.Munch()
        self.anat_vars.fsdict = get_fs_filedict(self.subject,self.bids_root)
        self.anat_vars.process_list = compile_fs_process_list(self)  
        
        if check_paths:
            self.check_for_files()         

    
    def initialize_fnames(self, rest_tagname, emptyroom_tagname):
        '''Use the bids paths to generate output names'''
        _tmp=munch.Munch()
        rest_deriv = self.rest_derivpath.copy().update(extension='.fif')
        if emptyroom_tagname!=None:
            eroom_deriv = self.eroom_derivpath.copy().update(extension='.fif')
        
        ## Setup bids paths for all 
        # Conversion to actual paths at end
        
        if(self._t1_override == None):
            _tmp['anat']=self.bids_path.copy().update(datatype='anat',extension='.nii')
            if not os.path.exists(_tmp['anat'].fpath):
                _tmp['anat']=self.bids_path.copy().update(datatype='anat',extension='.nii.gz')
        else:
            _tmp['anat'] = mne_bids.get_bids_path_from_fname(self._t1_override)

        _tmp['raw_rest']=self.meg_rest_raw
        if emptyroom_tagname!=None:
            _tmp['raw_eroom']=self.meg_er_raw
        
        _tmp['rest_filt']=rest_deriv.copy().update(processing='filt')
        if emptyroom_tagname!=None:
            _tmp['eroom_filt']=eroom_deriv.copy().update(processing='filt')
        
        # MEGNET post
        #_tmp['rest_clean']=rest_deriv.copy().update(processing='clean')
        #_tmp['eroom_clean']=eroom_deriv.copy().update(processing='clean')
        
        _tmp['rest_epo']=rest_deriv.copy().update(suffix='epo')
        if emptyroom_tagname!=None:
            _tmp['eroom_epo']=eroom_deriv.copy().update(suffix='epo')
        
        #_tmp['rest_epo_clean']=_tmp['rest_epo'].copy().update(processing='clean')
        #_tmp['eroom_epo_clean']=_tmp['eroom_epo'].copy().update(processing='clean')
        
        if self.do_dics:
            _tmp['rest_csd']=rest_deriv.copy().update(suffix='csd', extension='.h5')
            if emptyroom_tagname!=None:
                _tmp['eroom_csd']=eroom_deriv.copy().update(suffix='csd', extension='.h5')
        else:
            _tmp['rest_cov']=rest_deriv.copy().update(suffix='cov', extension='.fif')
            if emptyroom_tagname!=None:
                _tmp['eroom_cov']=eroom_deriv.copy().update(suffix='cov', extension='.fif')
        
        _tmp['rest_fwd']=rest_deriv.copy().update(suffix='fwd') 
        _tmp['rest_trans']=rest_deriv.copy().update(suffix='trans')
        _tmp['bem'] = self.deriv_path.copy().update(suffix='bem', extension='.fif')
        _tmp['src'] = self.deriv_path.copy().update(suffix='src', extension='.fif')
        _tmp['ica'] = self.deriv_path.copy().update(suffix='ica', extension='.fif')
        if self.do_dics:
            _tmp['dics'] = self.deriv_path.copy().update(suffix='dics', 
                                                     run=self.meg_rest_raw.run,
                                                     extension='.h5')  
        else:
            _tmp['lcmv'] = self.deriv_path.copy().update(suffix='lcmv',
                                                     run=self.meg_rest_raw.run,
                                                     extension='.h5')
        self.fooof_dir = self.deriv_path.directory / \
            self.deriv_path.copy().update(datatype=None, extension=None).basename
    
        # Cast all bids paths to paths and save as dictionary
        path_dict = {key:str(i.fpath) for key,i in _tmp.items()}
        
        # Additional non-bids path files
        path_dict['parc'] = op.join(self.subjects_dir, 'morph-maps', 
                               f'sub-{self.subject}-fsaverage-morph.fif') 
        
        outfolder = self.deriv_path.directory / \
            self.deriv_path.copy().update(datatype=None, extension=None).basename
        
        if self.run != None:
            path_dict['spectra'] = str(outfolder) + f'_label_task-{rest_tagname}_run-{self.run}_spectra.csv'
            path_dict['power'] = str(outfolder) + f'_band_task-{rest_tagname}_run-{self.run}_rel_power.csv'
        else:
            path_dict['spectra'] = str(outfolder) + f'_label_task-{rest_tagname}_spectra.csv'
            path_dict['power'] = str(outfolder) + f'_band_task-{rest_tagname}_rel_power.csv'
            
        return munch.Munch(path_dict)

    @log
    def check_for_files(self):
        if not os.path.exists(self.fnames['raw_rest']):
            raise ValueError('Raw MEG file not present')
        if self.meg_er_raw != None:
             if not os.path.exists(self.fnames['raw_eroom']):
                 raise ValueError('EmptyRoom dataset not present')
        if not os.path.isfile(str(self.fnames['anat'])):
            raise ValueError('Anatomical MRI not present')     
        
        
        
# =============================================================================
#       Load data
# =============================================================================

    @log
    def load_data(self):
        if not hasattr(self, 'raw_rest'):
            self.raw_rest = load_data(self.meg_rest_raw.fpath) 
            self.raw_rest.pick_types(meg=True, eeg=False, exclude=[])
        if (not hasattr(self, 'raw_eroom')) and (self.meg_er_raw != None):
            self.raw_eroom = load_data(self.meg_er_raw.fpath) 
            self.raw_eroom.pick_types(meg=True, eeg=False, exclude=[])
        # For subsequent reference, if raw_room not provided, set to None
        if (not hasattr(self, 'raw_eroom')):
            self.raw_eroom=None
        # figure out the MEG system vendor, note that this may be different from 
        # datatype if the datatype is .fif
        self.vendor = mne.channels.channels._get_meg_system(self.raw_rest.info)
    
    @log
    def check_paths(self):
        '''Verify that the raw data is present and can be found'''
        try:                            # Errors if MEG is not present
            self.meg_rest_raw.fpath  
            self.datatype = check_datatype(str(self.meg_rest_raw.fpath))
            
        except:
            logging.exception(f'Could not find rest dataset:{self.meg_rest_raw.fpath}\n')
            
        if hasattr(self, 'meg_er_raw'): 
            if self.meg_er_raw != None:
                try:                    # Error if empty room is specified but not present
                    self.meg_er_raw.fpath
                except:
                    logging.exception(f'Could not find emptyroom dataset:{self.meg_er_raw.fpath}\n')
            else:
                self.raw_eroom = None
        else:
            self.meg_er_raw = None
            self.raw_eroom = None
        
        # check if freesurfer directory is present for the subject
        subj_fsdir= op.join(f'{self.subjects_dir}', f'sub-{self.subject}')
        if not op.exists(subj_fsdir):
            errtxt = f'There is no freesurfer folder for {self.subject}:\
                              \n{subj_fsdir}'.replace('  ',' ')
            logging.exception(errtxt)

# =============================================================================
#       Vendor specific prep
# =============================================================================

    @log
    def vendor_prep(self, megin_ignore=None):
        
        # check for channels with no locations
        badlocnames=self.raw_rest.info['bads']
        for chan in self.raw_rest.info['chs']:
            if np.isnan(sum(chan['loc'])):
                badlocnames.append(chan['ch_name'])
        self.raw_rest.info['bads']=badlocnames
        
        '''Different vendor types require special cleaning / initialization'''
        print('vendor = %s' % self.vendor[0])
        ## Apply 3rd order gradient for CTF datasets  
        if self.vendor[0] == 'CTF_275':
            if self.raw_rest.compensation_grade != 3:
                logging.info('Applying 3rd order gradient to rest data')
                self.raw_rest.apply_gradient_compensation(3)
            if self.raw_eroom != None: 
                    if self.raw_eroom.compensation_grade != 3:
                        logging.info('Applying 3rd order gradient to emptyroom data')
                        self.raw_eroom.apply_gradient_compensation(3)
         
        # check to see if data has already been maxfiltered before further processing

        if (_check_maxfilter(self.raw_rest)):
            logging.info('Maxfilter already applied, skipping ahead')
            self.bad_channels = []
            
        # run bad channel assessments on rest and emptyroom (if present)
        else: 
            print(self.vendor[0])
            rest_bad, rest_flat = assess_bads(self.meg_rest_raw.fpath, self.vendor[0])
            if hasattr(self, 'raw_eroom'):
                if self.raw_eroom != None:
                    er_bad, er_flat = assess_bads(self.meg_er_raw.fpath, self.vendor[0], is_eroom=True)
            else:
                er_bad = []
                er_flat =[]
            if hasattr(self, 'raw_eroom'):
                if self.raw_eroom != None:
                    all_bad = self.raw_rest.info['bads'] + self.raw_eroom.info['bads'] + \
                        rest_bad + rest_flat + er_bad + er_flat
                else:
                    all_bad = self.raw_rest.info['bads'] + rest_bad + rest_flat #This may be redundant to below
            else:
                all_bad = self.raw_rest.info['bads'] + rest_bad + rest_flat 
            # remove duplicates
            all_bad = list(set(all_bad))
                
            # mark bad/flat channels as such in datasets
            self.bad_channels = all_bad
            
            # if the vendor is MEGIN, we don't want to drop the channels, only declare them as bad
            # in the dataset header - Maxfilter will interpolate them
            if ((self.vendor[0] == '306m') | (self.vendor[0] == '122m')):
                self.raw_rest.info['bads'] = all_bad
                if self.raw_eroom != None:
                    self.raw_eroom.info['bads'] = all_bad
                print('declared all bad or flat channels in info[bads]')
                self.bad_channels = []
                print(all_bad) 
            
            # if it's not an elekta scan, just drop all the bad channels
            else:                
                for chan in all_bad:
                    if chan in self.raw_rest.info['ch_names']:
                        self.raw_rest.drop_channels(chan)
                if self.raw_eroom != None: 
                    for chan in all_bad:
                        if chan in self.raw_rest.info['ch_names']:
                            self.raw_eroom.drop_channels(all_bad)
                print('dropped all bad or flat channels')
                print(all_bad) 
            
            # Movement correction and Maxwell filtering for Elekta systems
            
            if ((self.vendor[0] == '306m') | (self.vendor[0] == '122m')):
                # Get the calibration files - check global variable to see if they 
                # were passed on the commandline
                if 'megin_cal_files' in globals().keys():
                    ct_sparse_path, sss_cal_path = find_cal_files(args=megin_cal_files, 
                                                                  bids_path=None)
                else:
                    ct_sparse_path, sss_cal_path = find_cal_files(args=None, 
                                                                  bids_path=self.bids_path)
                self.ct_sparse = ct_sparse_path
                self.sss_cal = sss_cal_path         
                
                if (megin_ignore != True):
                
                    # Check for and run the movement correction on the dataset
                    chpi_info = mne.chpi.get_chpi_info(self.raw_rest.info)
                    if hasattr(chpi_info[0], '__len__'):
                        if len(chpi_info[0]) > 0:
                            self._movement_comp()
                
                else:
  
                    self._tsss()
                   
# =============================================================================
#       Preprocessing
# =============================================================================

    @log
    def _preproc(self,          # resampling, mains notch filtering, bandpass filtering
                raw_inst=None,
                deriv_path=None):
        raw_inst.resample(self.proc_vars['sfreq'], n_jobs=self._n_jobs)
        raw_inst.notch_filter(self.proc_vars['mains'], n_jobs=self._n_jobs) 
        raw_inst.filter(self.proc_vars['fmin'], self.proc_vars['fmax'], n_jobs=self._n_jobs)
        raw_inst.save(deriv_path.copy().update(processing='filt', extension='.fif'), 
                      overwrite=True)
    
    @log
    def _movement_comp(self,
                       ):
        '''
        Perform movement correction - currently restricted to MEGIN data
        '''
        deriv_path = self.deriv_path
        raw_inst = self.raw_rest
        chpi_amplitudes = mne.chpi.compute_chpi_amplitudes(raw_inst)
        chpi_locs = mne.chpi.compute_chpi_locs(raw_inst.info, chpi_amplitudes)
        head_pos = mne.chpi.compute_head_pos(raw_inst.info, chpi_locs,verbose=False)
        np.save(str(deriv_path.fpath)[:-4]+'_run-'+str(self.run)+'_headpos.npy', head_pos)
        raw_tsss = maxwell_filter(raw_inst, head_pos=head_pos,
                                 cross_talk=self.ct_sparse,
                                 calibration=self.sss_cal, 
                                 st_duration=10.0)
        # raw_tsss = mne.chpi.filter_chpi(raw_tsss)
        # Note that filtering out the chpi channels was tested and removed 
        # because it sometimes injected peaks in the 25-40Hz range, particularly
        # for scans collected with powerline frequency = 50Hz. Because we are only
        # interested in 1-45 Hz, and the chpi coils are substantially higher frequency,
        # this step can be omitted.
        raw_tsss.save(deriv_path.copy().update(processing='mcorr', run=self.run, extension='.fif'),
                                               overwrite=True)
        self.raw_rest=raw_tsss
        if self.raw_eroom != None:
            eroom_inst = self.raw_eroom
            try:
                eroom_prep = mne.preprocessing.maxwell_filter_prepare_emptyroom(eroom_inst,
                                raw=raw_tsss, bads='keep')
                eroom_tsss = maxwell_filter(eroom_prep, head_pos=None, 
                                cross_talk=self.ct_sparse,
                                calibration=self.sss_cal, 
                                st_duration=10.0)
            except: 
                  raw_hspmod = raw_tsss.copy()
                  for d in raw_hspmod.info['dig']:
                      if d['kind'] == FIFF.FIFFV_POINT_EEG:
                          d['kind'] = FIFF.FIFFV_POINT_EXTRA
                  eroom_prep = mne.preprocessing.maxwell_filter_prepare_emptyroom(eroom_inst,
                                    raw=raw_hspmod, bads='keep')
                  eroom_tsss = maxwell_filter(eroom_prep, head_pos=None, 
                                    cross_talk=self.ct_sparse,
                                    calibration=self.sss_cal, 
                                    st_duration=10.0)    
                
            self.raw_eroom = eroom_tsss
            
    @log
    def _tsss(self,
                         ):
          '''
          Perform tsss without movement correction - currently restricted to MEGIN data
          '''
          deriv_path = self.deriv_path
          raw_inst = self.raw_rest
          raw_tsss = maxwell_filter(raw_inst,
                                   cross_talk=self.ct_sparse,
                                   calibration=self.sss_cal, 
                                   st_duration=10.0)
          raw_tsss.save(deriv_path.copy().update(processing='tsss', run=self.run, extension='.fif'),
                                                 overwrite=True)
          self.raw_rest=raw_tsss
          if self.raw_eroom != None:
              eroom_inst = self.raw_eroom
              try: 
                  eroom_prep = mne.preprocessing.maxwell_filter_prepare_emptyroom(eroom_inst,
                                  raw=raw_tsss, bads='keep')
                  eroom_tsss = maxwell_filter(eroom_prep, head_pos=None, 
                                  cross_talk=self.ct_sparse,
                                  calibration=self.sss_cal, 
                                  st_duration=10.0)
              except: 
                  print('initial hsp fit did not work, trying to copy any eeg points from digitization')
                  raw_hspmod = raw_tsss.copy()
                  for d in raw_hspmod.info['dig']:
                      if d['kind'] == FIFF.FIFFV_POINT_EEG:
                          d['kind'] = FIFF.FIFFV_POINT_EXTRA
                  eroom_prep = mne.preprocessing.maxwell_filter_prepare_emptyroom(eroom_inst,
                                    raw=raw_hspmod, bads='keep')
                  eroom_tsss = maxwell_filter(eroom_prep, head_pos=None, 
                                    cross_talk=self.ct_sparse,
                                    calibration=self.sss_cal, 
                                    st_duration=10.0)    
                  
              self.raw_eroom = eroom_tsss
  
    @log
    def do_ica(self):           # perform the 20 component ICA using functions from megnet
        from MEGnet.prep_inputs.ICA import main as ICA
        ica_basename = self.meg_rest_raw.basename + '_ica'
        bad_channels = [i for i in self.bad_channels if i in self.raw_rest.info['ch_names']] #Prevent drop channels from erroring
        ICA(self.raw_rest,mains_freq=float(self.proc_vars['mains']), 
            save_preproc=True, save_ica=True, results_dir=self.deriv_path.directory, 
            outbasename=ica_basename, do_assess_bads=False, bad_channels=bad_channels)  
        self.fnames.ica_folder = self.deriv_path.directory  / ica_basename
        self.fnames.ica = self.fnames.ica_folder / (ica_basename + '_0-ica.fif')
        self.fnames.ica_megnet_raw =self.fnames.ica_folder / (ica_basename + '_250srate_meg.fif')

    def prep_ica_qa(self):      # if desired, create QA images for ICA components
        ica_fname = self.fnames.ica
        raw_fname = self.fnames.ica_megnet_raw
        
        prep_fcn_path = op.join(enigmeg.__path__[0], 'QA/make_ica_qa.py')
        
        output_path = str(self.deriv_path.directory).replace('ENIGMA_MEG','ENIGMA_MEG_QA')
        output_path = op.dirname(output_path)
        
        subprocess.run(['python', prep_fcn_path, '-ica_fname', ica_fname, '-raw_fname', raw_fname,
                        '-vendor', self.vendor[0], '-results_dir', output_path, '-basename', self.meg_rest_raw.basename])
    @log           
    def do_classify_ica(self):  # use the MEGNET model to automatically classify ICA components as artifactual
        from scipy.io import loadmat
        import MEGnet
        from MEGnet.megnet_utilities import fPredictChunkAndVoting_parrallel
        from tensorflow import keras
        model_path = op.join(MEGnet.__path__[0] ,  'model_v2')
        # This is set to use CPU in initial import
        kModel=keras.models.load_model(model_path)
        arrSP_fnames = [op.join(self.fnames.ica_folder, f'component{i}.mat') for i in range(1,21)]
        arrTS = loadmat(op.join(self.fnames.ica_folder, 'ICATimeSeries.mat'))['arrICATimeSeries'].T
        arrSP = np.stack([loadmat(i)['array'] for i in arrSP_fnames])
        preds, probs = fPredictChunkAndVoting_parrallel(kModel, arrTS, arrSP)
        self.meg_rest_ica_classes = preds.argmax(axis=1)
        self.ica_comps_toremove = [index for index, value in enumerate(self.meg_rest_ica_classes) if value in [1, 2, 3]]
        logstring = 'MEGNET classifications: ' + str(self.meg_rest_ica_classes)
        logger.info(logstring)
        logstring = 'Components to reject: ' + str(self.ica_comps_toremove)
        logger.info(logstring)

    def set_ica_comps_manual(self): # If components were selected using manual QA, parse log files to generate list
        # lots of filenames have to be redefined here, if we've picked up after doing manual QA
        ica_basename = self.meg_rest_raw.basename + '_ica'
        self.fnames.ica_folder = self.deriv_path.directory  / ica_basename
        self.fnames.ica = self.fnames.ica_folder / (ica_basename + '_0-ica.fif')
        self.fnames.ica_megnet_raw =self.fnames.ica_folder / (ica_basename + '_250srate_meg.fif')
        newdict = parse_manual_ica_qa(self)
        self.ica_comps_toremove = np.asarray(newdict[self.meg_rest_raw.basename]).astype(int)
        logstring = 'Components to reject: ' + str(self.ica_comps_toremove)
        logger.info(logstring)
        
    @log
    def do_clean_ica(self):         # Remove identified ICA components    
        print("removing ica components")
        print("self.fnames.ica_folder %s" % self.fnames.ica_folder)
        print("self.fnames.ica %s" % self.fnames.ica)
        QAsubjdir = str(self.QA_dir.directory)
        QAfname = self.QA_dir.copy().update(suffix='cleaned',extension='png')
        figname_icaoverlay = QAsubjdir + QAfname.basename
        ica=mne.preprocessing.read_ica(op.join(self.fnames.ica))
        ica.exclude = self.ica_comps_toremove #meg_rest_raw.icacomps
        self.load_data()
        try:
            fig=ica.plot_overlay(self.raw_rest, exclude=self.ica_comps_toremove)
            fig.savefig(figname_icaoverlay)
        except:
            #Hack to prevent issues with headless servers
            logger.warning('Could not produce ICA images - possibly a server display issue')
        ica.apply(self.raw_rest)
        
    @log
    def do_preproc(self):           # proc both rest and empty room
        '''Preprocess both datasets'''
        self._preproc(raw_inst=self.raw_rest, deriv_path=self.rest_derivpath)
        if self.raw_eroom != None:
            self._preproc(raw_inst=self.raw_eroom, deriv_path=self.eroom_derivpath)

    @log
    def _proc_epochs(self,          # divide the rest data into epochs
                     raw_inst=None,
                     deriv_path=None):
        '''Create and save epochs'
        Create and save cross-spectral density'''
        evts = mne.make_fixed_length_events(raw_inst, duration=self.proc_vars['epoch_len'])
        logstring = 'Original number of epochs: ' + str(len(evts))
        logger.info(logstring)
        tmax = self.proc_vars['epoch_len'] - 1/self.proc_vars['sfreq']
        chtypes=self.raw_rest.get_channel_types()
        if 'grad' in chtypes:
            if 'mag' in chtypes: 
                reject_dict = dict(mag=magthresh, grad=gradthresh)
                flat_dict = dict(mag=flatmagthresh, grad=flatgradthresh)
            else:
                reject_dict = dict(grad=gradthresh)
                flat_dict = dict(grad=flatgradthresh)
        else:
            reject_dict = dict(mag=magthresh)
            flat_dict = dict(mag=flatmagthresh)
        epochs = mne.Epochs(raw_inst, evts, reject=reject_dict, flat=flat_dict,
                            preload=True, baseline=None, tmin=0, tmax=tmax)
        #epochs = mne.make_fixed_length_epochs(raw_inst, 
        #                                      duration=self.proc_vars['epoch_len'], 
        #                                      preload=True)
        z = zscore(np.std(epochs._data, axis=2), axis=0)
        bad_epochs = np.where(z>std_thresh)[0]
        epochs.drop(indices=bad_epochs)
        logstring = 'Final number of epochs: ' + str(epochs.__len__())
        logger.info(logstring)
        logstring = 'Total time in seconds: ' + str(epochs.__len__() * self.proc_vars['epoch_len'])
        logger.info(logstring)
        epochs_fname = deriv_path.copy().update(suffix='epo', extension='.fif')
        epochs.save(epochs_fname, overwrite=True)
        
        if self.do_dics:
            # compute the cross spectral density for the epoched data
            # multitaper better but slower, use fourier for testing
            #csd = mne.time_frequency.csd_fourier(epochs,fmin=fmin,fmax=fmax,n_jobs=self._n_jobs)
            csd = mne.time_frequency.csd_multitaper(epochs,fmin=fmin,fmax=fmax,n_jobs=self._n_jobs)
            csd_fname = deriv_path.copy().update(suffix='csd', extension='.h5')
            csd.save(str(csd_fname.fpath), overwrite=True)
        else:
            cov = mne.compute_covariance(epochs)
            cov_fname = deriv_path.copy().update(suffix='cov', extension='.fif')
            cov.save(cov_fname, overwrite=True)
    
    @log
    def do_proc_epochs(self):   # epoch both the rest and the empty room
        self._proc_epochs(raw_inst=self.raw_rest,
                          deriv_path=self.rest_derivpath)
        if self.raw_eroom != None:
            self._proc_epochs(raw_inst=self.raw_eroom, 
                              deriv_path=self.eroom_derivpath)
    
    @log                        # Process the anatomical MRI
    def proc_mri(self, t1_override=None,redo_all=False,volume='T1',preflood=None, gcaatlas=True):
        
        # if not provided with a separate T1 MRI filename, extract it from the BIDSpath objects
        if t1_override is not None:
            t1_bids_path = self.anat_bidspath
        else:
            t1_bids_path = self.bids_path.copy().update(datatype='anat', 
                                                    suffix='T1w')
                                                    
        # Configure filepaths
        os.environ['SUBJECTS_DIR']=self.subjects_dir
        deriv_path = self.rest_derivpath.copy()
        
        fs_subject = 'sub-' + self.subject
        
        fs_subject_dir = os.path.join(self.subjects_dir,fs_subject)
        
        # check to see if Freesurfer is running (or if it crashed out)
        if 'IsRunning.lh+rh' in os.listdir(os.path.join(fs_subject_dir, 'scripts')):
            del_run_file=input('''The IsRunning.lh+rh file is present.  Could be from a broken process \
                  or the process is currently running.  Do you want to delete to continue?(y/n)''')
            if del_run_file.lower()=='y':
                os.remove(os.path.join(fs_subject_dir, 'scripts','IsRunning.lh+rh'))
                
        # do all the remaining freesurfer processing steps        
        for proc in self.anat_vars.process_list:
            subcommand(proc)
        
        # Update path so src/bem are not saved as task/run specific
        src_fname = deriv_path.copy().update(suffix='src', extension='.fif',
                                             task=None, run=None)
        bem_fname = deriv_path.copy().update(suffix='bem', extension='.fif',
                                             task=None, run=None)
        
        # Specific to task = rest / run = run#
        fwd_fname = deriv_path.copy().update(suffix='fwd', extension='.fif')
        trans_fname = deriv_path.copy().update(suffix='trans',extension='.fif')

        # check to see if stuff is there, and if it isn't, make it                
        if fwd_fname.fpath.exists() and (redo_all is False):        # forward solution
            fwd = mne.read_forward_solution(fwd_fname)
            self.rest_fwd = fwd
        
        watershed_check_path=op.join(self.subjects_dir,             # watershed bem
                                     f'sub-{self.subject}',
                                     'bem',
                                     'inner_skull.surf'
                                     )
        if (not os.path.exists(watershed_check_path)) or (redo_all is True):
            mne.bem.make_watershed_bem(f'sub-{self.subject}',
                                       self.subjects_dir,
                                       overwrite=True, 
                                       gcaatlas=gcaatlas,
                                       volume=volume,
                                       preflood=preflood,
                                       )
        if (not bem_fname.fpath.exists()) or (redo_all is True):
            bem = mne.make_bem_model(f'sub-{self.subject}', 
                                     subjects_dir=self.subjects_dir, 
                                     conductivity=[0.3])
            bem_sol = mne.make_bem_solution(bem)
            
            mne.write_bem_solution(bem_fname, bem_sol, overwrite=True)
        else:
            bem_sol = mne.read_bem_solution(bem_fname)
            
        if (not src_fname.fpath.exists()) or (redo_all is True):    # source space
            src = mne.setup_source_space(f'sub-{self.subject}',
                                         spacing='oct6', add_dist='patch',
                                 subjects_dir=self.subjects_dir)
            src.save(src_fname.fpath, overwrite=True)
        else:
            src = mne.read_source_spaces(src_fname.fpath)
        
        if (not trans_fname.fpath.exists()) or (redo_all is True):  # transformation
            if self._use_fsave_coreg==True:
                trans = self.do_auto_coreg()
            else:
                if(self.datatype == 'fif'):
                    trans = get_head_mri_trans(self.meg_rest_raw, {'allow_maxshield' : True},
                                           t1_bids_path=t1_bids_path,
                                           fs_subject='sub-'+self.bids_path.subject) 
                elif(self.datatype == 'ctf'):
                    trans = get_head_mri_trans(self.meg_rest_raw, {'system_clock' : 'ignore'},
                                           t1_bids_path=t1_bids_path,
                                           fs_subject='sub-'+self.bids_path.subject)
                elif(self.datatype == '4d'):
                    trans = get_head_mri_trans_bti(self.meg_rest_raw,
                                           t1_bids_path=t1_bids_path,
                                           fs_subject='sub-'+self.bids_path.subject,
                                           fs_subjects_dir=self.subjects_dir)
                else:
                    trans = get_head_mri_trans(self.meg_rest_raw,
                                           t1_bids_path=t1_bids_path,
                                           fs_subject='sub-'+self.bids_path.subject)
            mne.write_trans(trans_fname.fpath, trans, overwrite=True)
        else:
            trans = mne.read_trans(trans_fname.fpath)
        # make the forward solution
        fwd = mne.make_forward_solution(self.raw_rest.info, trans, src, bem_sol, eeg=False, 
                                        n_jobs=self._n_jobs)
        self.rest_fwd=fwd
        mne.write_forward_solution(fwd_fname.fpath, fwd, overwrite=True)
    
    @log
    def do_auto_coreg(self):
        '''Localize coarse fiducials based on fsaverage coregistration
        and fine tuned with iterative headshape fit.'''
        coreg = mne.coreg.Coregistration(self.raw_rest.info, 
                                         f'sub-{self.subject}', 
                                         subjects_dir=self.subjects_dir, 
                                         fiducials='estimated')
        coreg.fit_fiducials(verbose=True)
        coreg.omit_head_shape_points(distance=15. / 1000)  # distance is in meters
        # use default parameters for the icp fit
        coreg.fit_icp(verbose=True)
        return coreg.trans
            
    # make the parcellation and subparcellation
    @log
    def do_make_aparc_sub(self):
        write_aparc_sub(subjid=f'sub-{self.subject}', 
                        subjects_dir=self.subjects_dir)
        
# =============================================================================
#       Source Localization and spectral parameterization
# =============================================================================
         
    @log
    def do_beamformer(self):    # Function to do the beamformer    
        
        if self.do_dics:
            # read in all the necessary files
            dat_csd = mne.time_frequency.read_csd(self.fnames.rest_csd)
            forward = mne.read_forward_solution(self.fnames.rest_fwd)
            epochs = mne.read_epochs(self.fnames.rest_epo)
            fname_dics = self.fnames.dics #Pre-assign output name
        else:
            dat_cov = mne.read_cov(self.fnames.rest_cov)
            forward = mne.read_forward_solution(self.fnames.rest_fwd)
            epochs = mne.read_epochs(self.fnames.rest_epo)
            fname_lcmv = self.fnames.lcmv #Pre-assign output name
        
        if(_check_maxfilter(self.raw_rest)):
            epo_rank = mne.compute_rank(epochs, rank='info')
        else:
            epo_rank = mne.compute_rank(epochs)          # compute rank of rest dataset
        
        # If emptyroom present - use in beamformer

        if self.meg_er_raw != None:

            noise_rank = mne.compute_rank(self.raw_eroom)
            
            if 'mag' in epo_rank:
                if epo_rank['mag'] < noise_rank['mag']:
                    noise_rank['mag']=epo_rank['mag']
            if 'grad' in epo_rank:
                if epo_rank['grad'] < noise_rank['grad']:
                    noise_rank['grad']=epo_rank['grad']
                
            if self.do_dics:
                noise_csd = mne.time_frequency.read_csd(self.fnames.eroom_csd)
                #filters=mne.beamformer.make_dics(epochs.info, forward, dat_csd, reg=0.05, pick_ori='max-power',
                #    noise_csd=noise_csd, inversion='matrix', weight_norm='unit-noise-gain', rank=noise_rank)
                filters=mne.beamformer.make_dics(epochs.info, forward, dat_csd, reg=0.05, pick_ori='max-power',
                    noise_csd=noise_csd, inversion='single', weight_norm=None, depth=1, rank=noise_rank)
            else:
                noise_cov = mne.read_cov(self.fnames.eroom_cov)
                filters = make_lcmv(epochs.info, forward, dat_cov, reg=0.05, noise_cov=noise_cov,  pick_ori='max-power',
                    weight_norm='unit-noise-gain', rank=noise_rank)
        elif self.meg_er_raw == None and self.vendor[0] == '306m':
            noise_cov = mne.make_ad_hoc_cov(epochs.info)
            filters = make_lcmv(epochs.info, forward, dat_cov, reg=0.05, noise_cov=noise_cov,
                        pick_ori='max-power', weight_norm='unit-noise-gain', rank=epo_rank)
        else:
            #Build beamformer without emptyroom noise
            if self.do_dics:
                filters=mne.beamformer.make_dics(epochs.info, forward, dat_csd, reg=0.05,pick_ori='max-power',
                    inversion='matrix', weight_norm='unit-noise-gain', rank=epo_rank)
            
            else:
                # make an ad hoc covariance with 
                noise_cov = mne.make_ad_hoc_cov(epochs.info,std=5.0e-13)
                filters = make_lcmv(epochs.info, forward, dat_cov, reg=0.05, noise_cov=noise_cov,
                            pick_ori='max-power', weight_norm='unit-noise-gain', rank=epo_rank)
        
        if self.do_dics:
            filters.save(fname_dics, overwrite=True)
            psds, freqs = apply_dics_csd(dat_csd, filters) 
            self.psds = psds
            self.freqs = freqs
        else:
            filters.save(fname_lcmv, overwrite=True)
            stcs = apply_lcmv_epochs(epochs, filters, return_generator=True)
            self.stcs=stcs
        
    @log    
    def do_label_psds(self):    # Function to extract the psd for each label. Takes a long time. 
        labels_lh=mne.read_labels_from_annot(f'sub-{self.subject}',
                                             parc='aparc_sub',
                                            subjects_dir=self.subjects_dir,
                                            hemi='lh') 
        labels_rh=mne.read_labels_from_annot(f'sub-{self.subject}',
                                             parc='aparc_sub',
                                             subjects_dir=self.subjects_dir,
                                             hemi='rh') 
        labels=labels_lh + labels_rh 
        self.labels = labels
        
        if self.do_dics:
            label_ts = mne.source_estimate.extract_label_time_course(self.psds, 
                                                                 labels, 
                                                                 self.rest_fwd['src'],
                                                                 mode='mean')
        else:
            
            label_ts = mod_label_extract.mod_extract_label_time_course(self.stcs,
                                                         labels,
                                                         self.rest_fwd['src'],
                                                         mode='pca15_multitaper',
                                                         fmin=self.proc_vars['fmin'], 
                                                         fmax=self.proc_vars['fmax'])
                                                         
        #Convert list of numpy arrays to ndarray (Epoch/Label/Sample)
        self.label_ts = np.stack(label_ts)
    
    @log    
    def do_spectral_parameterization(self):
        '''
        Passes spectra to fooof alg for peak and 1/f analysis

        Returns
        -------
        None.

        '''
        if self.do_dics:
            freq_bins = np.array(self.freqs)    
        else:
            freq_bins = np.linspace(fmin,fmax,n_bins)
            
        #Initialize 
        labels = self.labels
        label_power = np.zeros([len(labels), len(freq_bins)])  
        alpha_peak = np.zeros(len(labels))
        offset = np.zeros(len(labels))
        exponent = np.zeros(len(labels))
        
        outfolder = self.deriv_path.directory / \
            self.deriv_path.copy().update(datatype=None, extension=None).basename
        if not os.path.exists(outfolder): os.mkdir(outfolder)
        
        #Create PSD for each label
        label_stack = self.label_ts
        for label_idx in range(len(self.labels)):
            if self.do_dics:
                current_psd = label_stack[label_idx, :]
            else:
                labels_trimmedmean = trim_mean(label_stack[:,label_idx,:], 0.1, axis=0)
                current_psd = labels_trimmedmean  
            label_power[label_idx,:] = current_psd
            
            #spectral_image_path = os.path.join(outfolder, 'Spectra_'+
            #                                    labels[label_idx].name + '.png')   
            spectral_image_path = None  ## supress output of spectra .png files for every region
            
            tmp_fmodel = calc_spec_peak(freq_bins, current_psd, 
                                out_image_path=spectral_image_path)
            try:
                # work around for when fooof identifies multiple alpha peaks - set to np.nan
                potential_alpha_idx = np.where((8.0 <= tmp_fmodel.peak_params[:,0] ) & \
                                        (tmp_fmodel.peak_params[:,0] <= 12.0 ) )[0]
                if len(potential_alpha_idx) != 1:
                    alpha_peak[label_idx] = np.nan         
                else:
                    alpha_peak[label_idx] = tmp_fmodel.peak_params[potential_alpha_idx[0]][0]
                    print('label_idx: %d, alpha_peak: %f' % (label_idx, alpha_peak[label_idx]))

            except:
                alpha_peak[label_idx] = np.nan  # case where no alpha peak is identified - set to np.nan
            
            offset[label_idx] = tmp_fmodel.aperiodic_params[0]
            exponent[label_idx] = tmp_fmodel.aperiodic_params[1]
            
        #Save the label spectrum to assemble the relative power
        freq_bin_names=[str(binval) for binval in freq_bins]
        label_spectra_dframe = pd.DataFrame(label_power, columns=[freq_bin_names])
        label_spectra_dframe.to_csv(self.fnames['spectra'] , index=False)

        # with open(os.path.join(info.outfolder, 'label_spectra.npy'), 'wb') as f:
        #     np.save(f, label_power)
        
        relative_power = label_power / label_power.sum(axis=1, keepdims=True)
    
        #Define bands
        bands = [[1,3], [3,6], [8,12], [13,35], [35,45]]
        band_idxs = get_freq_idx(bands, freq_bins)
    
        #initialize output
        band_means = np.zeros([len(labels), len(bands)]) 
        #Loop over all bands, select the indexes assocaited with the band and average    
        for mean_band, band_idx in enumerate(band_idxs):
            band_means[:, mean_band] = relative_power[:, band_idx].sum(axis=1)    
    
        bands_str = [str(i) for i in bands]
        label_names = [i.name for i in labels]
        
        output_dframe = pd.DataFrame(band_means, columns=bands_str, 
                                     index=label_names)
        output_dframe['AlphaPeak'] = alpha_peak
        output_dframe['AperiodicOffset'] = offset
        output_dframe['AperiodicExponent'] = exponent
        output_dframe.to_csv(self.fnames['power'], sep='\t')  

    @log    
    def do_mri_segstats(self):
        '''
        Self-explanatory does mri_segstats
        
        Returns
        -------
        # outputs eTIV, lh.orig.nofix holes, rh.orig.nofix holes, and average holes 
        '''
        os.environ['SUBJECTS_DIR'] = self.subjects_dir
        out = subprocess.getoutput(f'mri_segstats --seg {self.subjects_dir}/sub-{self.subject}/mri/aseg.mgz --subject sub-{self.subject} --etiv-only')
        pattern = r"atlas_icv \(eTIV\) = (\d+) mm\^3"
        tiv = re.search(pattern,out).group(1)
        out = subprocess.getoutput(f'mris_euler_number {self.subjects_dir}/sub-{self.subject}/surf/lh.orig.nofix')
        pattern = r"index = (\d+)"
        lh_tmp = re.search(pattern,out)
        if hasattr(lh_tmp, 'group'):
            lh_holes = lh_tmp.group(1)
        else:
            lh_holes = '0'
        out = subprocess.getoutput(f'mris_euler_number {self.subjects_dir}/sub-{self.subject}/surf/rh.orig.nofix')
        pattern = r"index = (\d+)"
        rh_tmp = re.search(pattern,out)
        if hasattr(rh_tmp, 'group'):
            rh_holes = rh_tmp.group(1)
        else:
            rh_holes = '0'
        logstring = 'eTIV: ' + str(tiv) + ' lh_holes: ' + str(lh_holes) + ' rh_holes: ' + str(rh_holes) + ' avg_holes: ' + str((int(lh_holes)+int(rh_holes))/2)
        print(logstring)
        logger.info(logstring)
        
    def cleanup(self):
        rogue_derivpath = self.deriv_path.update(extension=None)
        rogue_derivdir = op.join(rogue_derivpath.directory, rogue_derivpath.basename)
        if os.path.isdir(rogue_derivdir):
            if len(os.listdir(rogue_derivdir))==0: # make sure directory is empty
                os.rmdir(rogue_derivdir)
        rogue_bidspath = self.bids_path
        rogue_bidsdir = op.join(rogue_bidspath.directory, rogue_bidspath.basename)
        if os.path.isdir(rogue_bidsdir):
            if len(os.listdir(rogue_bidsdir))==0:
                os.rmdir(rogue_bidsdir)
        
# =============================================================================
#       Perform all functions on an instance of the process class
# =============================================================================
    
    def do_proc_allsteps(self):  # do all the steps for single subject command line processing
        self.load_data()
        self.vendor_prep(megin_ignore=self._megin_ignore)
        self.do_ica()
        self.do_classify_ica()
        self.do_preproc()
        self.do_clean_ica()
        self.do_proc_epochs()
        self.proc_mri(t1_override=self._t1_override)
        self.do_beamformer()
        self.do_make_aparc_sub()
        self.do_label_psds()
        self.do_spectral_parameterization()
        self.do_mri_segstats()
        self.cleanup()
        
# =============================================================================
#       Super secret hidden debugging functions. 
# =============================================================================
        
    # hidden debugging function to list all output files generated
    def list_outputs(self):
        exists = [i for i in self.fnames if op.exists(self.fnames[i])]
        missing = [i for i in self.fnames if not op.exists(self.fnames[i])]
        for i in exists:
            print(f'Present: {self.fnames[i]}')
        print('\n\n')
        for i in missing:
            print(f'Missing: {self.fnames[i]}')  
    
    # hidden debugging function to check alignment
    def check_alignment(self): # test function to look at alignment - not called
        self.trans = mne.read_trans(self.fnames['rest_trans'])
        self.load_data()
        mne.viz.plot_alignment(self.raw_rest.info,
                                    trans=self.trans,
                                    subject='sub-'+self.bids_path.subject,
                                    subjects_dir=self.subjects_dir,
                                    dig=True)
 
# =========================== End of class and method =========================        
        
# =============================================================================
#       Utility and helper functions
# =============================================================================
    
def subcommand(function_str):               # simple function to run something on the command line
    from subprocess import check_call
    check_call(function_str.split(' '))
        
def compile_fs_process_list(process):       # function to determine what freesurfer steps still must be run
    process_steps=[]
    fs_subject = 'sub-'+process.subject
    if not(process.anat_vars.fsdict['001mgz']):
        process_steps.append('recon-all -i {} -s {}'.format(process.fnames['anat'], fs_subject))
    if not(process.anat_vars.fsdict['brainmask']):
        process_steps.append('recon-all -autorecon1 -s {}'.format(fs_subject))
    if not(process.anat_vars.fsdict['lh_pial']):
        process_steps.append('recon-all -autorecon2 -careg -s {}'.format(fs_subject))
    if not(process.anat_vars.fsdict['lh_dkaparc']) and not(process.anat_vars.fsdict['lh_dkaparc_alt']):
        process_steps.append('recon-all -autorecon3 -s {}'.format(fs_subject))
    return process_steps   


def get_fs_filedict(subject, bids_root):    # make a dictionary of freesurfer filenames
    subjects_dir = op.join(bids_root, 'derivatives', 'freesurfer', 'subjects')
    
    fs_dict = {}
    # FS default files
    fs_dict['001mgz'] = f'sub-{subject}/mri/orig/001.mgz'
    fs_dict['brainmask'] = f'sub-{subject}/mri/brainmask.mgz'
    fs_dict['lh_pial'] = f'sub-{subject}/surf/lh.pial'
    fs_dict['rh_pial'] = f'sub-{subject}/surf/rh.pial'
    fs_dict['lh_dkaparc'] = f'sub-{subject}/label/lh.aparc.DKTatlas.annot'
    fs_dict['lh_dkaparc_alt'] = f'sub-{subject}/label/lh.aparc.annot'
    fs_dict['rh_dkaparc'] = f'sub-{subject}/label/rh.aparc.DKTatlas.annot'
    
    # FS - post process files
    fs_dict['in_skull'] = f'sub-{subject}/bem/inner_skull.surf'
    fs_dict['head'] = f'sub-{subject}/bem/sub-{subject}-head.fif'
    fs_dict['head_dense'] = f'sub-{subject}/bem/sub-{subject}-head-dense.fif'
    
    # Morph maps for sub-aparc
    fs_dict['morph'] = f'morph-maps/fsaverage-sub-{subject}-morph.fif'
    
    # Prepend subjects_dir
    for key, value in fs_dict.items():
        fs_dict[key] = op.join(subjects_dir,value)
        if not op.exists(fs_dict[key]):
            fs_dict[key]=False
    return fs_dict

def _check_maxfilter(raw):
    maxfilter_status = False  
    if (len(raw.info['proc_history']) > 0):
        if ("max_info" in raw.info['proc_history'][0]):
            if ("sss_cal" in raw.info['proc_history'][0]['max_info']):
                if (len(raw.info['proc_history'][0]['max_info']['sss_cal']) > 0):
                    maxfilter_status = True
    return maxfilter_status
                    
 
def check_datatype(filename):               # function to determine the file format of MEG data 
    '''Check datatype based on the vendor naming convention to choose best loader'''
    if os.path.splitext(filename)[-1] == '.ds':
        return 'ctf'
    elif os.path.splitext(filename)[-1] == '.fif':
        return 'fif'
    elif os.path.splitext(filename)[-1] == '.4d' or ',' in str(filename):
        return '4d'
    elif os.path.isdir(str(filename)):
        tmp_ = glob.glob(op.join(str(filename),'*,*'))
        if len(tmp_) > 1: 
            raise ValueError('Too many files with commas in the meg bids folder')
        if len(tmp_) == 1:
            filename = tmp_[0]
            return '4d'
        else:
            raise ValueError('Cannot determine if this is 4D data or other')
    elif os.path.splitext(filename)[-1] == '.sqd':
        return 'kit'
    elif os.path.splitext(filename)[-1] == 'con':
        return 'kit'
    else:
        raise ValueError('Could not detect datatype')
        
def return_dataloader(datatype):            # function to return a data loader based on file format
    '''Return the dataset loader for this dataset'''
    if datatype == 'ctf':
        return functools.partial(mne.io.read_raw_ctf, system_clock='ignore',
                                 clean_names=True)
    if datatype == 'fif':
        return functools.partial(mne.io.read_raw_fif, allow_maxshield=True)
    if datatype == '4d':
        return mne.io.read_raw_bti
    if datatype == 'kit':
        return mne.io.read_raw_kit

def load_data(filename):                    # simple function to load raw MEG data
    datatype = check_datatype(filename)
    dataloader = return_dataloader(datatype)
    if dataloader == mne.io.read_raw_bti:
        tmp_ = glob.glob(op.join(str(filename),'*,*'))
        hs_file = glob.glob(op.join(str(filename),'hs_file'))
        assert len(tmp_) == 1
        if len(hs_file) == 1:  
            raw = dataloader(filename / tmp_[0], preload=True, head_shape_fname=hs_file[0])
        else:  #This will be the case for emptyroom
            raw = dataloader(filename / tmp_[0], preload=True, head_shape_fname=None)
    else:
        raw = dataloader(filename, preload=True)
    return raw

def assess_bads(raw_fname, vendor, is_eroom=False): # assess MEG data for bad channels
    '''Code sampled from MNE python website
    https://mne.tools/dev/auto_tutorials/preprocessing/\
        plot_60_maxwell_filtering_sss.html'''
    from mne.preprocessing import find_bad_channels_maxwell
    # load data with load_data to ensure correct function is chosen
    raw = load_data(raw_fname)    
    if raw.times[-1] > 60.0:
        raw.crop(tmax=60.0)    
    raw.info['bads'] = []
    
    # catch any channels with no locations
    badlocnames=[]
    for chan in raw.info['chs']:
        if np.isnan(sum(chan['loc'])):
            badlocnames.append(chan['ch_name'])
    raw.info['bads']=badlocnames
    
    raw_check = raw.copy()
    
    if vendor == '306m' or vendor == '122m':
        if is_eroom==False:
            auto_noisy_chs, auto_flat_chs, auto_scores = find_bad_channels_maxwell(
                raw_check, cross_talk=None, calibration=None,
                return_scores=True, verbose=True)
        else:
            auto_noisy_chs, auto_flat_chs, auto_scores = find_bad_channels_maxwell(
                raw_check, cross_talk=None, calibration=None,
                return_scores=True, verbose=True, coord_frame="meg")
            
        # find_bad_channels_maxwell is actually pretty bad at finding flat channels - 
        # it uses a much too stringent threshold. So, we need some supplementary code
        # This is extra complicated for Elekta/MEGIN, because there are both mags and 
        # grads, which will be on a different scale
            
        mags = mne.pick_types(raw_check.info, meg='mag')
        grads = mne.pick_types(raw_check.info, meg='grad')
        # get the standard deviation for each channel, and the trimmed mean of the stds
        # have to do this separately for mags and grads
        stdraw_mags = np.std(raw_check._data[mags,:],axis=1)
        stdraw_grads = np.std(raw_check._data[grads,:],axis=1)    
        stdraw_trimmedmean_mags = sp.stats.trim_mean(stdraw_mags,0.1)
        stdraw_trimmedmean_grads = sp.stats.trim_mean(stdraw_grads,0.1)
        # we can't use the same threshold here, because grads have a much greater 
        # variance in the variances 
        flat_idx_mags = mags[np.where(stdraw_mags < stdraw_trimmedmean_mags/100)[0]]
        flat_idx_grads = grads[np.where(stdraw_grads < stdraw_trimmedmean_grads/1000)[0]]
        flats = []
        for flat in flat_idx_mags:
            flats.append(raw_check.info['ch_names'][int(flat)])
        for flat in flat_idx_grads:
            flats.append(raw_check.info['ch_names'][int(flat)])
        
    # ignore references and use 'meg' coordinate frame for CTF and KIT
    
    elif vendor == 'CTF_275':
        raw_check.apply_gradient_compensation(0)
        auto_noisy_chs, auto_flat_chs, auto_scores = find_bad_channels_maxwell(
            raw_check, cross_talk=None, calibration=None, coord_frame='meg',
            return_scores=True, verbose=True, ignore_ref=True)
        
        # again, finding flat/bad channels is not great, so we add another algorithm
        # since other vendors don't mix grads and mags, we only need to do this for
        # a single channel type
        
        megs = mne.pick_types(raw_check.info, meg=True,ref_meg=False)
        # get the standard deviation for each channel, and the trimmed mean of the stds
        stdraw_megs = np.std(raw_check._data[megs,:],axis=1)
        stdraw_trimmedmean_megs = sp.stats.trim_mean(stdraw_megs,0.1)
        flat_idx_megs = megs[np.where(stdraw_megs < stdraw_trimmedmean_megs/100)[0]]
        flats = []
        for flat in flat_idx_megs:
            flats.append(raw_check.info['ch_names'][flat]) 
    
    else: 
        auto_noisy_chs, auto_flat_chs, auto_scores = find_bad_channels_maxwell(
            raw_check, cross_talk=None, calibration=None, coord_frame='meg',
            return_scores=True, verbose=True, ignore_ref=True)
    
        # again, finding flat/bad channels is not great, so we add another algorithm
        # since other vendors don't mix grads and mags, we only need to do this for
        # a single channel type
    
        megs = mne.pick_types(raw_check.info, meg=True,ref_meg=False)
        # get the standard deviation for each channel, and the trimmed mean of the stds
        stdraw_megs = np.std(raw_check._data[megs,:],axis=1)
        stdraw_trimmedmean_megs = sp.stats.trim_mean(stdraw_megs,0.1)
        # note the that the threshold here is different for non-CTF or MEGIN systems
        # we empirically observed that flat channels in KIT scanners sometimes had
        # significant noise resulting in a failure to detect the flat channel
        flat_idx_megs = megs[np.where(stdraw_megs < stdraw_trimmedmean_megs/50)[0]]
        flats = []
        for flat in flat_idx_megs:
            flats.append(raw_check.info['ch_names'][flat]) 
    
    auto_flat_chs = auto_flat_chs + flats
    auto_flat_chs = list(set(auto_flat_chs))
    print(auto_noisy_chs, auto_flat_chs)
            
    return auto_noisy_chs, auto_flat_chs            

def write_aparc_sub(subjid=None, subjects_dir=None):    # write the parcel annotation, fetch fsaverage if needed
    '''Check for fsaverage and aparc_sub and download
    Morph fsaverage aparc_sub labels to single subject data
    
    https://mne.tools/stable/auto_examples/visualization/plot_parcellation.html
    '''
    if not op.exists(op.join(subjects_dir, 'fsaverage')):
        mne.datasets.fetch_fsaverage(verbose='ERROR') #True requires TQDM
    if not op.exists(op.join(subjects_dir, 'fsaverage', 'label', 
                             'lh.aparc_sub.annot')):
        mne.datasets.fetch_aparc_sub_parcellation(subjects_dir=subjects_dir,
                                          verbose='ERROR')
    
    sub_labels=mne.read_labels_from_annot('fsaverage',parc='aparc_sub', 
                                   subjects_dir=subjects_dir)        
    subject_labels=mne.morph_labels(sub_labels, subject_to=subjid, 
                                 subjects_dir=subjects_dir)
    mne.write_labels_to_annot(subject_labels, subject=subjid, 
                              parc='aparc_sub', subjects_dir=subjects_dir, 
                              overwrite=True)
    
def find_cal_files(args=None, bids_path=None):
    '''
    The SSS files (ct_sparse.fif and sss_cal.dat) can be provided in three different
    ways in order of precedence:
        1) Commandline option in args
        2) BIDS subject/sess/meg folder with the acq tag
            sub-01_acq-crosstalk_meg.fif 
            sub-01_acq-calibration_meg.dat 
        3) BIDS root folder

    Parameters
    ----------
    args : argparse args object, optional
        Args from the commandline input. The default is None.
    bids_path : mne_bids.BIDSPath , optional
        Subject BIDSPath. The default is None.

    Returns
    -------
    ct_sparse_path : str
    sss_cal_path : str

    '''
    # 1) Commandline options
    if hasattr(args,'ct_sparse') & hasattr(args, 'sss_cal'):
        if (args.ct_sparse != None) & (args.sss_cal != None):
            assert args.ct_sparse[-3:] == 'fif'
            assert args.sss_cal[-3:] == 'dat'
            return args.ct_sparse, args.sss_cal
    
    # 2) BIDS compliant parsing
    crosstalk_bids_path = bids_path.copy().update(acquisition='crosstalk')
    cal_bids_path = bids_path.copy().update(acquisition='calibration', extension='.dat')
    if op.exists(crosstalk_bids_path.fpath) & op.exists(cal_bids_path.fpath):
        return crosstalk_bids_path.fpath, cal_bids_path.fpath
    
    # 3) Files located at the bids_root
    bids_root=bids_path.root
    ct_sparse_path = op.join(bids_root, 'ct_sparse.fif')
    sss_cal_path = op.join(bids_root, 'sss_cal.dat')
    if (op.exists(ct_sparse_path)) & (op.exists(sss_cal_path)):
        return ct_sparse_path, sss_cal_path
    
    # 4) Could not find files
    raise(ValueError('''Could not find the ct_sparse and sss_cal files, please
                     provide these as a commandline alt options'''))
        
        
    

# hidden testing functions
def load_test_data(**kwargs):
    os.environ['n_jobs']='6'
    proc = process(subject='ON02747',
                        bids_root=op.expanduser('~/ds004215'),
                        session='01',
                        run='01',
                        emptyroom_tagname='noise', 
                        mains=60,
                        t1_override='~/ds004215/sub-ON02747/ses-01/anat/sub-ON02747_ses-01_acq-MPRAGE_T1w.nii.gz',
                        **kwargs)
    # proc.load_data()
    return proc

# hidden testing functions
def load_test_data_noer(**kwargs):
    os.environ['n_jobs']='6'
    proc = process(subject='ON02747',
                        bids_root=op.expanduser('~/ds004215'),
                        session='01',
                        emptyroom_tagname=None, 
                        mains=60,
                        t1_override='~/ds004215/sub-ON02747/ses-01/anat/sub-ON02747_ses-01_acq-MPRAGE_T1w.nii.gz',
                        **kwargs)
    # proc.load_data()
    return proc

# print(__name__)
# # For Testing run this cell
# if __name__!='__main__':
# proc = load_test_data()

# simple function to get frequency indices
def get_freq_idx(bands, freq_bins):
    ''' Get the frequency indexes'''
    output=[]
    for band in bands:
        tmp = np.nonzero((band[0] < freq_bins) & (freq_bins < band[1]))[0]   
        output.append(tmp)
    return output

# =============================================================================
#       Master processing functions 
# =============================================================================

def process_subject(subject, args):
    # logger = get_subj_logger(subject, args.session, args.rest_tag, args.run, log_dir)
    # logger.info('Initializing structure')
    proc = process(subject=subject, 
            bids_root=args.bids_root, 
            deriv_root=None,
            subjects_dir=None,
            rest_tagname=args.rest_tag,
            emptyroom_tagname=args.emptyroom_tag, 
            session=args.session, 
            mains=float(args.mains),
            run=args.run,
            t1_override=None,
            megin_ignore=args.megin_ignore,
            fs_ave_fids=args.fs_ave_fids,
            do_dics=args.do_dics)
    proc.do_proc_allsteps()
    
def process_subject_up_to_icaqa(subject, args):
    logger = get_subj_logger(subject, args.session, args.rest_tag, args.run, log_dir)
    logger.info('Initializing structure')
    proc = process(subject=subject, 
            bids_root=args.bids_root, 
            deriv_root=None,
            subjects_dir=None,
            rest_tagname=args.rest_tag,
            emptyroom_tagname=args.emptyroom_tag, 
            session=args.session, 
            mains=float(args.mains),
            run=args.run,
            t1_override=None,
            megin_ignore=args.megin_ignore,
            fs_ave_fids=args.fs_ave_fids,
            do_dics=args.do_dics
            )
    proc.load_data()
    proc.vendor_prep(megin_ignore=proc._megin_ignore)
    proc.do_ica()
    proc.prep_ica_qa()    
    
def process_subject_after_icaqa(subject, args):
    logger = get_subj_logger(subject, args.session,args.rest_tag, args.run, log_dir)
    logger.info('Initializing structure')
    proc = process(subject=subject, 
            bids_root=args.bids_root, 
            deriv_root=None,
            subjects_dir=None,
            rest_tagname=args.rest_tag,
            emptyroom_tagname=args.emptyroom_tag, 
            session=args.session, 
            mains=float(args.mains),
            run=args.run,
            t1_override=None,
            megin_ignore=args.megin_ignore,
            fs_ave_fids=args.fs_ave_fids,
            do_dics=args.do_dics
            )
    proc.load_data()
    proc.vendor_prep(megin_ignore=proc._megin_ignore)
    proc.set_ica_comps_manual()
    proc.do_preproc()
    proc.do_clean_ica()
    proc.do_proc_epochs()
    proc.proc_mri(t1_override=proc._t1_override)
    proc.do_beamformer()
    proc.do_make_aparc_sub()
    proc.do_label_psds()
    proc.do_spectral_parameterization()
    proc.do_mri_segstats()
    proc.cleanup()
        
def parse_manual_ica_qa(self):
    logfile_path = self.bids_root + '/derivatives/ENIGMA_MEG_QA/ica_QA_logfile.txt'
    self_subjrun = 'sub-' + self.subject + '_ses-' + self.session + '_task-rest_run-' + self.run
    with open(logfile_path) as f:
        logcontents = f.readlines()
    dictionary = build_status_dict(logcontents)

    newdict = {}
    for key, value in dictionary.items():
        subjrun = key.split('_icacomp-')[0]
        
        # while the code here was originally set up to process the file and create a dictionary for all subjects, this outer conditional staement 
        # has it only check for bad components and add to the dictionary for the current subject being processed. 
        if subjrun == self_subjrun:     

            if newdict == {}:  # if this is the first key and the new dict is empty
                if dictionary[key].strip('\n') == 'BAD':  # if component is bad
                    dropcomp = key.split('_icacomp-')[1].split('.png')[0]
                    newdict = {subjrun: [dropcomp]}
                else:
                    newdict = {subjrun: []}  # if compoenent is good
                   
            elif subjrun in newdict.keys():   # If the key is already in the new dict
                if dictionary[key].strip('\n') == 'BAD':  # If component is bad (if comp is good, do nothing)
                    dropcomp = key.split('_icacomp-')[1].split('.png')[0]
                    newdict[subjrun].append(dropcomp)
    
            else: # if the key isn't in the new dictionary
                if dictionary[key].strip('\n') == 'BAD':  # if compoenent is bad
                    dropcomp = key.split('_icacomp-')[1].split('.png')[0]
                    newdict[subjrun] = [dropcomp]
                else:
                    newdict[subjrun] = [] # if compoenent is good
    return newdict

#%%  Argparse
def return_args():
    import argparse  
    parser = argparse.ArgumentParser()
    standardargs = parser.add_argument_group('Standard Inputs')
    standardargs.add_argument('-bids_root',
                        help='''Top level directory of the bids data'''
                        )

    standardargs.add_argument('-subject',
                        help='''BIDS ID of subject to process''',
                        default=None
                        )
    standardargs.add_argument('-subjects_dir',
                        help='''Freesurfer subjects directory, only specify if not \
                        bids_root/derivatives/freesurfer/subjects'''
                        )
    standardargs.add_argument('-fs_subject',
                        help='''Freefurfer subject ID if different from BIDS ID'''
                        )
    standardargs.add_argument('-run',
                        help='''Run number.  \nNOTE: 01 is different from 1''',
                        default='1'
                        )
    standardargs.add_argument('-session',
                        default=None,
                        )
    standardargs.add_argument('-mains',
                        help='Electric mains frequency  (50 or 60)',
                        default=60.0,
                        type=float
                        )
    standardargs.add_argument('-rest_tag',
                        help='Override in case task name is other than rest\
                            for example - resteyesopen',
                        default='rest'
                        )
    standardargs.add_argument('-emptyroom_tag',
                        help='Override in case emptryoom is other than \
                            emptyroom.  In case of no emptyroom, set as None on cmdline',
                        default='emptyroom'
                        )
    standardargs.add_argument('-n_jobs',
                        help='''number of jobs to run concurrently for 
                        multithreaded operations''',
                        default=1
                        )
    csvargs = parser.add_argument_group('Inputs from a CSV')
    csvargs.add_argument('-proc_fromcsv',
                        help='''Loop over all subjects in the bids_root
                        and process. Requires CSV file with processing manifest''',
                        default=None
                        )
    
    altargs = parser.add_argument_group('Alternative Inputs')
    altargs.add_argument('-fs_ave_fids',
                        help='''If no fiducials have been localized to the mri
                        manually, this provides a coarse fit from the ave brain
                        which is fine tuned with the headshape.  This is less
                        acurate than a manually assessed fid placement''',
                        action='store_true',
                        default=False
                        )
    altargs.add_argument('-do_dics',
                        help='''If flag is present, do a DICS beamformer. Otherwise, do lcmv. ''',
                        action='store_true',
                        default=0
                        )
    altargs.add_argument('-ct_sparse', 
                         help='''(Elekta/MEGIN datasets) This is the ct_sparse.fif
                         file associated with the MEG system.  This is an override if
                         the file cannot be found in the sub/ses/meg BIDS directory or at the 
                         top level of the bids tree.''', 
                         default=None
                         )
    altargs.add_argument('-sss_cal',
                         help='''(Elekta/MEGIN datasets) This is the sss_cal.dat
                         file associated with the MEG system.  This is an override if
                         the file cannot be found in the sub/ses/meg BIDS directory or at the 
                         top level of the bids tree.''', 
                         default=None
                         )
    altargs.add_argument('-emptyroom_run',
                        help='Override in case run designation differs for \
                             the emptyroom and rest datasets',
                        default=None)
    qaargs = parser.add_argument_group('QA Inputs')
    qaargs.add_argument('-ica_manual_qa_prep',
                        help='''if flag is present, stop after ICA for manual QA''',
                        action='store_true',
                        default=0
                        )
    qaargs.add_argument('-process_manual_ica_qa',
                        help='''If flag is present, pick up analysis after performing manual ICA QA''',
                        action='store_true',
                        default=0
                        )
    parser.add_argument('-remove_old',
                        help='''If flag is present, remove any files from a prior run (excepting freesurfer data). ''',
                        action='store_true',
                        default=0
                        )
    parser.add_argument('-megin_ignore',
                        help='''Flag can be set to ignore megin processing, i.e. motcorr''',
                        action='store_true',
                        default=None)
                                   
    args = parser.parse_args()
    # print help if no arguments
    if len(sys.argv) == 1:
        parser.print_help()
        parser.exit(1) 
    
    if not op.exists(args.bids_root):    # throw an error if the BIDS root directory doesn't exist
        parser.print_help()
        raise ValueError('Please specify a correct -bids_root')     
    return args
    

def main():
    args = return_args()
    
    n_jobs = args.n_jobs  
    os.environ['n_jobs'] = str(n_jobs)
        
    # set some defaults
    if args.run:
        if args.run.lower()=='none': args.run=None
    if args.session:
        if args.session.lower()=='none': args.session=None
    if args.emptyroom_tag:
        if args.emptyroom_tag.lower()=='none': args.emptyroom_tag=None
        
    if not args.bids_root:
        bids_root = op.join(os.getcwd(), 'bids_out')
        args.bids_root = bids_root
    else:
        bids_root=args.bids_root
        
    
    # To make file handling easier, even if there is another subjects directory, we'll create one in 
    # the BIDS derivatives/ folder and set up symbolic links there. 
    
    if not os.path.isdir(os.path.join(bids_root,'derivatives')):
        os.makedirs(os.path.join(bids_root,'derivatives'))
    if not os.path.isdir(os.path.join(bids_root,'derivatives/freesurfer')):
        os.makedirs(os.path.join(bids_root,'derivatives/freesurfer'))
    if not os.path.isdir(os.path.join(bids_root,'derivatives/freesurfer/subjects')):
        os.makedirs(os.path.join(bids_root,'derivatives/freesurfer/subjects'))
        
    # We have to find out if there is an fsaverage in the freesurfer directory, and if there is, if 
    # it is a sym link. If it's a link, it will break later when we try to get the bem directory
    
    if os.path.isdir(os.path.join(bids_root,'derivatives/freesurfer/subjects/fsaverage')):
        if os.path.islink(os.path.join(bids_root,'derivatives/freesurfer/subjects/fsaverage')):
            raise ValueError('$SUBJECTS_DIR/fsaverage cannot be a symlink; remove and rerun process_MEG to fetch data')
            
    # check and make sure all fsaverage files are present and download if not. 
    mne.datasets.fetch_fsaverage(op.join(bids_root,'derivatives/freesurfer/subjects/'))
    
    global logger
    log_dir = f'{bids_root}/derivatives/ENIGMA_MEG/logs'
    if not os.path.isdir(os.path.join(bids_root,'derivatives/ENIGMA_MEG')):
        os.makedirs(os.path.join(bids_root,'derivatives/ENIGMA_MEG'))
    if not os.path.isdir(os.path.join(bids_root,'derivatives/ENIGMA_MEG/logs')):
        os.makedirs(os.path.join(bids_root,'derivatives/ENIGMA_MEG/logs'))
    
    # megin calibration files
    if (args.sss_cal != None) & (args.ct_sparse != None):
        global megin_cal_files
        megin_cal_files=munch.Munch(dict(sss_cal=args.sss_cal,
                                         ct_sparse=args.ct_sparse))
    
    # single subject vs. multiple subject processing.     
    
    if args.subject:    
        
        args.subject=args.subject.replace('sub-','') # strip the sub- off for uniformity
        print(args.subject)
        
        if args.emptyroom_run == None:
            args.emptyroom_run = args.run
        
        if args.remove_old:
            print('Removing files from prior runs')
            logfilename = args.subject + '_ses-' + str(args.session) + '_log.txt'
            subprocess.call(['rm', os.path.join(log_dir, logfilename)])
            subject_enigmadir = 'sub-' + args.subject
            enigmadir = os.path.join(bids_root,'derivatives/ENIGMA_MEG')
            subprocess.call(['rm','-r', os.path.join(enigmadir, subject_enigmadir)])
            
        if args.proc_fromcsv != None:
            raise ValueError("You can't specify both a subject id and a csv file, sorry")
            
        print('processing a single subject %s' % args.subject)      
            
        # now, parse the inputs for freesurfer directory and subject (if not in derivatives)
        # so freesurfer processing doesn't have to be repeated if it exists elsewhere
    
        if args.subjects_dir:
            
            dir_entered = os.path.abspath(args.subjects_dir) # get absolute paths to compare
            default_dir = os.path.abspath(os.path.join(bids_root, 'derivatives/freesurfer/subjects'))
            
            if dir_entered == default_dir: # don't use -subjects_dir if using the default subjects_dir                
                print('Specified FS subjects dir same as default')
                
            elif args.fs_subject:
                if args.fs_subject == ('sub-' + args.subject):
                    raise ValueError('Specified FS subject ID is same as subject ID, please remove -fs_subject and try again')
                else:
                    # make a symbolic link from the existing freesurfer directory to the subjects directory 
                    # in the derivatives folder in the bids tree. Make sure we match the bids name
                    print('linking freesurfer subject %s to bids subject %s in derivatives folder',(args.fs_subject, args.subject))
                    subprocess.call(['ln','-s',os.path.join(dir_entered, args.fs_subject),
                                 os.path.join(default_dir, 'sub-'+args.subject)])
                    
            else: # case where a different subjects dir was entered, but not a separate fs ID
                # make sure subjects directory exists with same ID as BIDS ID
                # make a symbolic link for the subject in the derivatives/freesurfer/subjects directory
                if os.path.isdir(os.path.join(dir_entered, 'sub-'+args.subject)):
                        subprocess.call(['ln','-s',os.path.join(dir_entered, 'sub-'+args.subject),
                            os.path.join(default_dir, 'sub-'+args.subject)])
                else: 
                    raise ValueError('No folder for subject in specified subjects_dir, please try again')
            
            # now that we've set up the symbolic links, we can now use the default subjects directory
            args.subjects_dir = default_dir
      
        logger.info(f'processing subject {args.subject} session {args.session}')
        
        if args.ica_manual_qa_prep:
            
            ica_qa_dir = f'{bids_root}/derivatives/ENIGMA_MEG_QA'
            if not os.path.isdir(ica_qa_dir):
                os.makedirs(ica_qa_dir)
            if not os.path.isdir(os.path.join(ica_qa_dir,'sub-'+args.subject)):
                os.makedirs(os.path.join(ica_qa_dir,'sub-'+args.subject))
            if not os.path.isdir(os.path.join(ica_qa_dir,'sub-'+args.subject+'/ses-'+args.session)):
                os.makedirs(os.path.join(ica_qa_dir,'sub-'+args.subject+'/ses-'+args.session))
            process_subject_up_to_icaqa(args.subject, args)

        elif args.process_manual_ica_qa:
            process_subject_after_icaqa(args.subject, args)

        else:
            process_subject(args.subject, args)  # process the single specified subject
  
    ## batch processing from a .csv file
               
    elif args.proc_fromcsv:
        
        # if the user specified a subjects_dir, check to see if it is the same as the default dir
        if args.subjects_dir:
                
            dir_entered = os.path.abspath(args.subjects_dir) # get absolute paths to compare
            default_dir = os.path.abspath(os.path.join(bids_root, 'derivatives/freesurfer/subjects'))
            
            if dir_entered == default_dir: # if the directories are the same, replace with absolute path
                print('Specified FS subjects dir same as default')
                args.subjects_dir = default_dir
                need_link = False
            else:
                args.subjects_dir = dir_entered
                need_link = True
        
        print('processing subject list from %s' % args.proc_fromcsv)
        
        dframe = pd.read_csv(args.proc_fromcsv, dtype={'sub':str, 'run':str, 'ses':str})
        dframe = dframe.astype(object).replace(np.nan,None)
        
        for idx, row in dframe.iterrows():  # iterate over each row in the .csv file
            
            print(row)
            
            subject=row['sub']
            subject = subject.replace('sub-','')
            session=str(row['ses'])
            run=str(row['run'])

            logger.info(f'processing subject {subject} session {session}')
                        
            if args.remove_old:
                print('Removing files from prior runs')
                logfilename = subject + '_ses-' + str(session) + '_log.txt'
                subprocess.call(['rm', os.path.join(log_dir, logfilename)])
                subject_enigmadir = 'sub-' + subject
                enigmadir = os.path.join(bids_root,'derivatives/ENIGMA_MEG')
                subprocess.call(['rm','-r', os.path.join(enigmadir, subject_enigmadir)])

            if row['mripath'] == None:
                logger.info('No MRI, cannot process any further')
                print("Can't process subject %s, no MRI found" % args.subject)
            
            else:
                rest_ent = mne_bids.get_entities_from_fname(row.path)
                er_ent = mne_bids.get_entities_from_fname(row.eroom)
                process_subj = process(subject = subject,
                                       bids_root = bids_root,
                                       deriv_root = None,
                                       subjects_dir = args.subjects_dir,
                                       rest_tagname = rest_ent['task'],
                                       emptyroom_tagname = er_ent['task'],
                                       emptyroom_run = er_ent['run'],
                                       session = row['ses'],
                                       mains = float(args.mains),
                                       run = row['run'],
                                       t1_override = row['mripath'],
                                       fs_ave_fids = False,
                                       check_paths = True,
                                       do_dics = args.do_dics, 
                                       megin_ignore = args.megin_ignore
                                       )
                
                process_subj.load_data()
                
                if (args.ica_manual_qa_prep == 1):
                    process_subj.vendor_prep()
                    process_subj.do_ica()
                    process_subj.prep_ica_qa()
                elif(args.process_manual_ica_qa == 1):
                    process_subj.vendor_prep()
                    process_subj.set_ica_comps_manual()
                    process_subj.do_preproc()
                    process_subj.do_clean_ica()
                    process_subj.do_proc_epochs()
                    process_subj.proc_mri()
                    process_subj.do_beamformer()
                    process_subj.do_make_aparc_sub()
                    process_subj.do_label_psds()
                    process_subj.do_spectral_parameterization()
                    process_subj.do_mri_segstats()
                    
                else:    
                    process_subj.do_proc_allsteps()
 
if __name__=='__main__':
    main()
