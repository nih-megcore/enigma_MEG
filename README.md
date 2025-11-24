[![DOI](https://zenodo.org/badge/294786606.svg)](https://doi.org/10.5281/zenodo.14616848)
# Installation

Setup Conda Environment:

```
conda create --override-channels --channel=conda-forge --name=enigma_meg 'mne>=1.6' 'python<3.12'
conda activate enigma_meg
pip install enigma-meg
megnet_init #to download ICA classification weights 
```
# Description

The programs in this package perform the full processing pipeline for the ENIGMA MEG Working Group. This suite requires
that your data be organized in BIDS format; if you need a tool for that you can use enigma_anonymization_lite. The core tool 
is process_meg.py, which performs all processing steps for the anatomical MRI and the associated MEG. You can either process
a single subject, or you can loop over all subjects in batch mode. In order to do batch processing, you must first run 
parse_bids.py to produce a .csv file manifest of all available MEG scans (or use some other method to generate the .csv file). 
There are two methods of artifact correction supported. The first is ica with manual identification of ica components. This is
likely the most accurate, if you have a very small dataset to process and you have lots of time. The second method is to use
ica with MEGnet automated classification of artifact components. MEGnet was retrained on data from the CTF, Elekta/MEGIN, 4D, 
and KIT data. The model classifies components with >98% accuracy, so this is also an excellent option. 

Once all the processing is complete, you can generate QA images using prep_QA.py. Like process_meg.py, prep_QA.py will operate 
either on a single subject or on all subjects listed in a the .csv file produced by parse_bids.py. Once the .png files are 
created, you can use Run_enigma_QA_GUI.py to interactively label your subject images as good or bad. 

# Main Processing Pipeline
```
usage: process_meg.py [-h] [-bids_root BIDS_ROOT] [-subject SUBJECT] [-subjects_dir SUBJECTS_DIR] [-fs_subject FS_SUBJECT] 
		      [-run RUN] [-session SESSION] [-mains MAINS] [-rest_tag REST_TAG] [-emptyroom_tag EMPTYROOM_TAG] 
		      [-fs_ave_fids] [-proc_fromcsv PROC_FROMCSV] [-n_jobs NJOBS]
		      
This runs all anatomical MRI and MEG processing to produce ENIGMA MEG working gruop results.

optional arguments: 
  -h, --help            show this help message and exit 
  -bids_root BIDS_ROOT 	the root directory for the BIDS tree
  -subject		BIDS ID of the subject to process
  -subjects_dir SUBJECTS_DIR
                        Freesurfer subjects_dir can be assigned at the commandline
                        if not in bids_root/derivatives/freesurfer/subjects
  -fs_subject FSID	Define subject's freesurfer ID if different from BIDS subject ID
  -run RUN		Run identifier for MEG scan in BIDS dir
  -session SESSION	Session identifier for MEG scan in BIDS dir
  -mains MAINS		Powerline frequency, defaults to 60
  -rest_tag REST_TAG	Stem for finding resting state MEG scans in the BIDS directory, defaults to 'rest'
  -emptyroom_tag EMPTYROOM_TAG
  			Stem for finding emptyroom datasets in the BIDS directory, defaults to 'emptyroom'
  -fs_ave_fids		optional coarse registration, please don't use this option unless you have to
  -proc_fromcsv	CSVFILE	Option to loop over rows in a processing manifest .csv file, such as that created
			with parse_bids.py
  -n_jobs NJOBS		Number of workers to use for multithreaded processing
  -ica_manual_qa_prep  	If this flag is present, only those steps up to the ica calculation will be performed,
			at which point you can manually QA the ICA components
  -process_manual_ica_qa
			If this flag is present, resume the analysis after manual identification of ica
			artifactual components
  -do_dics		Perform a DICS analysis instead of LCMV. Not recommended
  -remove_old		Flag to remove all output files and logs from a prior run of process_meg.py
```
Output:
```
BIDS_ROOT/derivatives/ENIGMA_MEG/logs/sub-SUBJECT-SESSION_log.txt
BIDS_ROOT/derivatives/ENIGMA_MEG/sub-SUBJECT/ses-SESSION/meg/sub-SUBJECT_ses-SESSION_bem.fif
BIDS_ROOT/derivatives/ENIGMA_MEG/sub-SUBJECT/ses-SESSION/meg/sub-SUBJECT_ses-SESSION_fooof_results_run-01/Band_rel_power.csv
BIDS_ROOT/derivatives/ENIGMA_MEG/sub-SUBJECT/ses-SESSION/meg/sub-SUBJECT_ses-SESSION_fooof_results_run-01/label_spectra.csv
BIDS_ROOT/derivatives/ENIGMA_MEG/sub-SUBJECT/ses-SESSION/meg/sub-SUBJECT_ses-SESSION_run-RUN_lcmv.h5
BIDS_ROOT/derivatives/ENIGMA_MEG/sub-SUBJECT/ses-SESSION/meg/sub-SUBJECT_ses-SESSION_src.fif
BIDS_ROOT/derivatives/ENIGMA_MEG/sub-SUBJECT/ses-SESSION/meg/sub-SUBJECT_ses-SESSION_task-EMPTYROOM_TAG_run-RUN_cov.fif
BIDS_ROOT/derivatives/ENIGMA_MEG/sub-SUBJECT/ses-SESSION/meg/sub-SUBJECT_ses-SESSION_task-EMPTYROOM_TAG_run-RUN_epo.fif
BIDS_ROOT/derivatives/ENIGMA_MEG/sub-SUBJECT/ses-SESSION/meg/sub-SUBJECT_ses-SESSION_task-EMPTYROOM_TAG_run-RUN_proc-filt_meg.fif
BIDS_ROOT/derivatives/ENIGMA_MEG/sub-SUBJECT/ses-SESSION/meg/sub-SUBJECT_ses-SESSION_task-REST_TAG_run-RUN_cov.fif
BIDS_ROOT/derivatives/ENIGMA_MEG/sub-SUBJECT/ses-SESSION/meg/sub-SUBJECT_ses-SESSION_task-REST_TAG_run-RUN_epo.fif
BIDS_ROOT/derivatives/ENIGMA_MEG/sub-SUBJECT/ses-SESSION/meg/sub-SUBJECT_ses-SESSION_task-REST_TAG_run-RUN_fwd.fif
BIDS_ROOT/derivatives/ENIGMA_MEG/sub-SUBJECT/ses-SESSION/meg/sub-SUBJECT_ses-SESSION_task-REST_TAG_run-RUN_trans.fif
BIDS_ROOT/derivatives/ENIGMA_MEG/sub-SUBJECT/ses-SESSION/meg/sub-SUBJECT_ses-SESSION_task-REST_TAG_run-RUN_proc-filt_meg.fif
BIDS_ROOT/derivatives/ENIGMA_MEG/sub-SUBJECT/ses-SESSION/meg/sub-SUBJECT_ses-SESSION_task-REST_TAG_run-RUN_ica/
BIDS_ROOT/derivatives/ENIGMA_MEG_QA/sub-SUBJECT/ses-SESSION/sub-SUBJECT_ses-SESSION_tasl=REST_TAG_run-RUN_cleaned.png

```
Example:
```
process_meg.py -subjects_dir /data/freesurfer/subjects -bids_root bids_out -subject sub-Subj01 -fs_subject Sub01 -run 01
	-session 1 -mains 50 -rest_tag rest -emptyroom_tag noise -n_jobs 4
```
This example will run a single subject. This subject has already been processed by freesurfer, and the subjects directory is not
in the BIDS tree. In addition, the freesurfer subject ID and BIDS subject ID are different. The rest MEG images are labeled 
'task-rest' and the emptyroom datasets are labeled 'task-noise' in the BIDS directory. This data was collected in Europe, so the
mains frequency is 50Hz. Process the data collected in session '1' and run '01'. Use 4 workers for any multithreaded functions.
```
process_meg.py -mains 60 -n_jobs 8 -proc_fromcsv ParsedBIDS_dataframe.csv
```
In this example, the data were collected in the US where the mains frequency is 60Hz. Use 8 jobs for multithreaded operations. 
Instead of processing a single subject, process all the subjects in the 'ParsedBIDS_dataframe.csv' file, which was created by 
parse_bids.py
```
process_meg.py -subject sub-Subj07 -run 01 -session 1 -mains 60 -ica_manual_qa_prep -remove_old
```
This example will run sub-Subj07, session 1, run 01, up through the calculation of the ICA, at which point it will stop. Old
files from a previous run of this subject will be erased first
```
process_meg.py -subject sub-Subj07 -run 01 -session 1 -mains 60 -process_manual_ica_qa
```
This example picks up with processing the same subject after QA of ica components has been performed using Run_enigma_QA_GUI.py 
(detailed below)

# Parsing the BIDS tree for batch processing
```
usage: parse_bids.py [-h] [-bids_root BIDS_ROOT] [-rest_tag REST_TAG] [-emptyroom_tag EMPTYROOM_TAG]

This python script parses a BIDS directory into a CSV file with one line per MEG to process

options:
  -h, --help            show this help message and exit
  -bids_root BIDS_ROOT  The name of the BIDS directory to be parsed
  -rest_tag REST_TAG    The filename stem to find rest datasets
  -emptyroom_tag EMPTYROOM_TAG
                        The filename stem to find emptyroom datasets
```
Output:
```
ParsedBIDS_dataframe.csv
```
Example:
```
parse_bids.py -bids_root bids_out -rest_tag resteyesopen -emptyroom_tag emptyroom
```
In this example, parse the BIDS tree located in 'bids_out'. The rest MEG images are labeled 'task-resteyesopen' and the 
emptyroom datasets are labeled 'task-emptyroom'. Importantly, this parser will look for MRI files that are potentially
stored in a separate session from the MEG data, and will attempt to match. A visual inspection of the .csv file is 
recommended before using the .csv file for batch processing of process_meg.py

# Extract QA images for assessment 
```
usage: enigma_prep_QA.py [-h] [-bids_root BIDS_ROOT] [-subjects_dir SUBJECTS_DIR] [-subjid SUBJID] [-session SESSION] [-run RUN] [-proc_from_csv PROC_FROM_CSV]

This python script will compile a series of QA images for assessment of the enigma_MEG pipeline

options:
  -h, --help            show this help message and exit
  -bids_root BIDS_ROOT  BIDS root directory
  -subjects_dir SUBJECTS_DIR
                        Freesurfer subjects_dir can be assigned at the commandline if not already exported
  -subjid SUBJID        Define the BIDS subject id to process
  -fs_subjid FSID	Freesurfer subject id if different from subjid
  -session SESSION      Session number
  -run RUN              Run number, note that 01 is different from 1
  -proc_from_csv PROC_FROM_CSV
                        Loop over all subjects in a .csv file (ideally produced by parse_bids.py
```
Output:
```
BIDS_ROOT/derivatives/ENIGMA_MEG_QA/sub-SUBJID/ses-SESSION/sub-SUBJID_ses-SESSION_run-RUN_beamformer.png
BIDS_ROOT/derivatives/ENIGMA_MEG_QA/sub-SUBJID/ses-SESSION/sub-SUBJID_ses-SESSION_run-RUN_bem.png
BIDS_ROOT/derivatives/ENIGMA_MEG_QA/sub-SUBJID/ses-SESSION/sub-SUBJID_ses-SESSION_run-RUN_coreg.png
BIDS_ROOT/derivatives/ENIGMA_MEG_QA/sub-SUBJID/ses-SESSION/sub-SUBJID_ses-SESSION_run-RUN_spectra.png
BIDS_ROOT/derivatives/ENIGMA_MEG_QA/sub-SUBJID/ses-SESSION/sub-SUBJID_ses-SESSION_run-RUN_src.png
BIDS_ROOT/derivatives/ENIGMA_MEG_QA/sub-SUBJID/ses-SESSION/sub-SUBJID_ses-SESSION_run-RUN_surf.png
```
Example:

```
enigma_prep_QA.py -bids_root BIDS -subjid sub-Subj1 -fs_subjid Subj1 -subjects_dir /home/freesurfer/subjects -session 1 -run 01
```
In this example, the bids directory is called BIDS, and we want to prepare the QA images for subject sub-Sub1. The freesurfer 
processing was run separately in the directory /home/freesurfer/subjects, where the subject id is Sub1. Process the QA images
for session 1 and run01.

# Run the QA GUI 
```
usage: Run_enigma_QA_GUI.py [-h] [-bids_root BIDS_ROOT] [-QAtype QATYPE] [-rows ROWS] [-columns COLUMNS] [-imgsize IMGSIZE]

options:
  -h, --help            show this help message and exit
  -bids_root BIDS_ROOT  Location of bids directory, default=bids_out
  -QAtype QATYPE        QA type to run. Options are:
  			'coreg' - produces images showing MEG/MRI alignment
			'bem' - shows the bem surface overlaid on the anatomical MRI
			'surf' - shows the freesurfer source reconstruction from multiple angles, along with the parcellation
			'src' - shows the source space 
			'spectra' - shows the average power spectral density in all sensors (overlaid)
			'beamformer' - shows the source localized (per parcel) alpha power overlaid on the brain surface
			'cleaned' - shows an overlay of raw data before and after ica cleaning
			'ica' - shows the ica time series and topomaps for components to allow manual idenfication of artifacts
	
  -rows ROWS            number of rows in QA browser, default value dependent on QAtype
  -columns COLUMNS      number of columns in QA browser, default value dependent on QAtype
  -imgsize IMGSIZE      make images smaller or larger, default value dependent on QAtype
  ```
  Output:
  ```
  BIDS_ROOT/derivatives/ENIGMA_MEG_QA/QATYPE_QA_logfile.txt
  ```
  Example:
  ```
  Run_enigma_QA_GUI.py -bids_root BIDS_OUT -QAtype coreg -rows 3 -columns 6 -imgsize 500
  ```
  In this example, the BIDS root directory is BIDS_OUT, and you want to QA the MRI/MEG coregistrations. You have a very large 
  widescreen monitor, so you can QA a 3x6 array of images at once. 
  ```
  Run_enigma_QA_GUI.py -bids_root BIDS_OUT -QAtype ica
  ```
  In this example, all of the calculated ica components for all subjects will be loaded into the QA viewer. This option exists
  for users that do not wish to use the MEGnet automated classifier and instead prefer to identify artifactual components 
  manually. Once you have identified the bad components you can return to run process_meg.py with -process_manual_ica_qa
