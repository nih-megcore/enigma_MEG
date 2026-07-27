
#>>>> https://stackoverflow.com/questions/53382383/makefile-cant-use-conda-activate
# Need to specify bash in order for conda activate to work.
SHELL=/bin/bash
# Note that the extra activate is needed to ensure that the activate floats env to the front of PATH
CONDA_ACTIVATE=source $$(conda info --base)/etc/profile.d/conda.sh ; conda activate ; conda activate 
# <<<<

data_repo = multi_vendor_test
data_repo2 = enigma_test_data
tmp_dir = $(pwd)
test_datadir = /data/NIGHTLY_TESTDATA

install_data:
	mkdir -p $(test_datadir)
	($(CONDA_ACTIVATE) datalad ; cd $(test_datadir); datalad install $(MEG_DATA_SERVER):$(data_repo); cd $(data_repo);  datalad get ./*  ) 
	cd $(tmp_dir)  #Revert out of datalad download location
	($(CONDA_ACTIVATE) datalad ; cd $(test_datadir); datalad install $(MEG_DATA_SERVER):$(data_repo2); cd $(data_repo2);  datalad get ./*  ) 


install_test:
	#conda install --channel=conda-forge --name=base mamba -y
	conda env remove -n enigma_meg_test -y
	conda create --override-channels --channel=conda-forge --name=enigma_meg_test "python<3.12" "mne=1.5" "numba<0.60" pip -y
	($(CONDA_ACTIVATE) enigma_meg_test ; pip install -e .[testing]; pip install pytest pytest-reportlog )
	git submodule init
	git pull --recurse-submodules

install_headless_test:
	#conda install --channel=conda-forge --name=base mamba -y
	conda env remove -n enigma_meg_test -y
	mamba create --override-channels --channel=conda-forge --name=enigma_meg_test "mne=1.5" "python<3.12" "numba<0.60" pip "vtk>=9.2=*osmesa*" "mesalib=21.2.5" -y
	($(CONDA_ACTIVATE) enigma_meg_test ; pip install -e .[testing]; pip install pytest pytest-reportlog )
	git submodule init
	git pull --recurse-submodules

install_system_requirements:
	dnf install Xvfb -y
	dnf install git git-annex -y

test:
	($(CONDA_ACTIVATE) enigma_meg_test ; cd enigma_MEG; pytest -vv --report-log=/tmp/enigma_MEG_test_logfile.txt )  

test_headless:
	($(CONDA_ACTIVATE) enigma_meg_test ; cd enigma_MEG; xvfb-run -a pytest -vv --report-log=/tmp/enigma_MEG_test_logfile.txt )


test_iterate_fs:
	($(CONDA_ACTIVATE) enigma_meg_test ; cd enigma_MEG; pytest -vv --report-log=./test_logfile.txt )  #xvfb-run -a pytest -s )

