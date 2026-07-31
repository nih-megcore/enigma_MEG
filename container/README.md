# Enigma MEG singularity
## Download container
`wget https://megcore.nih.gov/MEG/enigma_meg_v1.1.0.sif`

## Running container
Run the following substituting in your bids root folders in the variables below<br><br>
`singularity run -B ${FREESURFER_HOME}/license.txt:/opt/freesurfer-7.4.1/license.txt -B ${BIDS_ROOT}:${BIDS_ROOT} enigma_meg_v1.1.0.sif` <br><br>
Results in the container environment<br>
NOTE: Your home folder and your BIDS_ROOT are writeable from the container env <br>
```
USER@COMPUTER:~/src/enigma_MEG/container$ singularity run enigma_meg_v1.1.0.sif 
bash: /fast/freesurfer/SetUpFreeSurfer.sh: No such file or directory
bash: /opt/MNE_C/MNE-2.7.0-3106-Linux-x86_64/bin/mne_setup_sh: No such file or directory
Apptainer> 
Apptainer> 
```
Setup internal freesurfer env:
```
Apptainer> /opt/freesurfer-7.4.1/SetUpFreeSurfer.sh
```

Run the process_meg.py commands from the main page:
```
Apptainer> process_meg.py -h 
usage: process_meg.py [-h] [-bids_root BIDS_ROOT] [-subject SUBJECT] [-subjects_dir SUBJECTS_DIR]
                      [-fs_subject FS_SUBJECT] [-run RUN] [-session SESSION] [-mains MAINS]
                      [-rest_tag REST_TAG] [-emptyroom_tag EMPTYROOM_TAG] [-n_jobs N_JOBS]
                      [-proc_fromcsv PROC_FROMCSV] [-fs_ave_fids] [-do_dics] [-ct_sparse CT_SPARSE]
                      [-sss_cal SSS_CAL] [-emptyroom_run EMPTYROOM_RUN] [-ica_manual_qa_prep]
                      [-process_manual_ica_qa] [-remove_old] [-megin_ignore]

options:
  -h, --help            show this help message and exit
```

## Make singularity definition file (Not necessary for users)
Neurodocker version 1.0.1
```
neurodocker generate singularity \
    --copy ./enigma_meg.yml ./condastartup.sh / \
    --pkg-manager apt \
    --install git \
    --base-image neurodebian:bullseye\
    --freesurfer version=7.4.1 \
    --miniconda version=latest env_name="enigma_meg" yaml_file="enigma_meg.yml" \
    --entrypoint='/condastartup.sh' \
    --user nonroot > enigma_meg.def 
```

## Build Singularity container
`sudo singularity build enigma_meg.sif enigma_meg.def`


