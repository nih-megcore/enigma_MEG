from functools import partial
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from mne_bids import BIDSPath

from enigmeg import process_meg


@pytest.mark.parametrize(
    ("filename", "expected"),
    [
        ("sub-01_task-rest_meg.ds", "ctf"),
        ("sub-01_task-rest_meg.fif", "fif"),
        ("sub-01_task-rest_meg.4d", "4d"),
        ("c,rfDC", "4d"),
        ("sub-01_task-rest_meg.sqd", "kit"),
    ],
)
def test_check_datatype_detects_supported_file_names(filename, expected):
    assert process_meg.check_datatype(filename) == expected


def test_check_datatype_detects_bti_directory(tmp_path):
    bti_dir = tmp_path / "sub-01_task-rest_meg"
    bti_dir.mkdir()
    (bti_dir / "c,rfDC").touch()

    assert process_meg.check_datatype(bti_dir) == "4d"


def test_check_datatype_rejects_ambiguous_bti_directory(tmp_path):
    bti_dir = tmp_path / "sub-01_task-rest_meg"
    bti_dir.mkdir()
    (bti_dir / "c,rfDC").touch()
    (bti_dir / "e,rfhp").touch()

    with pytest.raises(ValueError, match="Too many files"):
        process_meg.check_datatype(bti_dir)


def test_check_datatype_rejects_unknown_file_name():
    with pytest.raises(ValueError, match="Could not detect datatype"):
        process_meg.check_datatype("sub-01_task-rest_meg.txt")


def test_return_dataloader_dispatches_supported_datatypes():
    ctf_loader = process_meg.return_dataloader("ctf")
    assert isinstance(ctf_loader, partial)
    assert ctf_loader.func is process_meg.mne.io.read_raw_ctf
    assert ctf_loader.keywords == {"system_clock": "ignore", "clean_names": True}

    fif_loader = process_meg.return_dataloader("fif")
    assert isinstance(fif_loader, partial)
    assert fif_loader.func is process_meg.mne.io.read_raw_fif
    assert fif_loader.keywords == {"allow_maxshield": True}

    assert process_meg.return_dataloader("4d") is process_meg.mne.io.read_raw_bti
    assert process_meg.return_dataloader("kit") is process_meg.mne.io.read_raw_kit
    assert process_meg.return_dataloader("unknown") is None


def test_load_data_calls_non_bti_loader(monkeypatch):
    calls = []

    def fake_loader(filename, preload):
        calls.append((filename, preload))
        return "raw"

    monkeypatch.setattr(process_meg, "check_datatype", lambda filename: "fif")
    monkeypatch.setattr(process_meg, "return_dataloader", lambda datatype: fake_loader)

    assert process_meg.load_data("sample.fif") == "raw"
    assert calls == [("sample.fif", True)]


def test_load_data_passes_bti_pdf_and_head_shape(monkeypatch, tmp_path):
    bti_dir = tmp_path / "sub-01_task-rest_meg"
    bti_dir.mkdir()
    pdf = bti_dir / "c,rfDC"
    head_shape = bti_dir / "hs_file"
    pdf.touch()
    head_shape.touch()
    calls = []

    def fake_read_raw_bti(filename, preload, head_shape_fname):
        calls.append((Path(filename), preload, Path(head_shape_fname)))
        return "raw-bti"

    monkeypatch.setattr(process_meg.mne.io, "read_raw_bti", fake_read_raw_bti)
    monkeypatch.setattr(process_meg, "check_datatype", lambda filename: "4d")
    monkeypatch.setattr(
        process_meg,
        "return_dataloader",
        lambda datatype: process_meg.mne.io.read_raw_bti,
    )

    assert process_meg.load_data(bti_dir) == "raw-bti"
    assert calls == [(pdf, True, head_shape)]


def test_check_maxfilter_detects_sss_calibration_history():
    raw = SimpleNamespace(
        info={"proc_history": [{"max_info": {"sss_cal": {"cal_chans": [1]}}}]}
    )

    assert process_meg._check_maxfilter(raw) is True


@pytest.mark.parametrize(
    "proc_history",
    [
        [],
        [{}],
        [{"max_info": {}}],
        [{"max_info": {"sss_cal": {}}}],
    ],
)
def test_check_maxfilter_returns_false_without_sss_calibration(proc_history):
    raw = SimpleNamespace(info={"proc_history": proc_history})

    assert process_meg._check_maxfilter(raw) is False


def test_compile_fs_process_list_returns_missing_recon_all_steps():
    proc = SimpleNamespace(
        subject="01",
        fnames={"anat": "/bids/sub-01/anat/sub-01_T1w.nii.gz"},
        anat_vars=SimpleNamespace(
            fsdict={
                "001mgz": False,
                "brainmask": False,
                "lh_pial": "/subjects/sub-01/surf/lh.pial",
                "lh_dkaparc": False,
                "lh_dkaparc_alt": False,
            }
        ),
    )

    assert process_meg.compile_fs_process_list(proc) == [
        "recon-all -i /bids/sub-01/anat/sub-01_T1w.nii.gz -s sub-01",
        "recon-all -autorecon1 -s sub-01",
        "recon-all -autorecon3 -s sub-01",
    ]


def test_get_fs_filedict_marks_present_files_and_missing_files(tmp_path):
    bids_root = tmp_path / "bids"
    subjects_dir = bids_root / "derivatives" / "freesurfer" / "subjects"
    existing = [
        subjects_dir / "sub-01" / "mri" / "orig" / "001.mgz",
        subjects_dir / "sub-01" / "surf" / "lh.pial",
        subjects_dir / "morph-maps" / "fsaverage-sub-01-morph.fif",
    ]
    for filename in existing:
        filename.parent.mkdir(parents=True, exist_ok=True)
        filename.touch()

    fsdict = process_meg.get_fs_filedict("01", bids_root)

    assert fsdict["001mgz"] == str(existing[0])
    assert fsdict["lh_pial"] == str(existing[1])
    assert fsdict["morph"] == str(existing[2])
    assert fsdict["brainmask"] is False


def test_find_cal_files_prefers_command_line_args(tmp_path):
    args = SimpleNamespace(
        ct_sparse=str(tmp_path / "ct_sparse.fif"),
        sss_cal=str(tmp_path / "sss_cal.dat"),
    )

    assert process_meg.find_cal_files(args=args, bids_path=None) == (
        args.ct_sparse,
        args.sss_cal,
    )


def test_find_cal_files_uses_bids_sidecars(tmp_path):
    bids_path = BIDSPath(
        root=tmp_path,
        subject="01",
        session="01",
        datatype="meg",
        suffix="meg",
        extension=".fif",
    )
    crosstalk = bids_path.copy().update(acquisition="crosstalk").fpath
    calibration = bids_path.copy().update(
        acquisition="calibration", extension=".dat"
    ).fpath
    crosstalk.parent.mkdir(parents=True)
    crosstalk.touch()
    calibration.touch()

    assert process_meg.find_cal_files(args=None, bids_path=bids_path) == (
        crosstalk,
        calibration,
    )


def test_get_comps_returns_final_manual_qa_components(tmp_path):
    logfile = tmp_path / "process.log"
    logfile.write_text(
        "2024-01-01 :: INFO :: Final ICA components after manual QA: [0, 2]\n"
        "2024-01-01 :: INFO :: other message\n",
        encoding="utf-8",
    )

    assert process_meg.get_comps(logfile) == [0, 2]


def test_get_freq_idx_uses_strict_band_boundaries():
    freq_bins = np.array([1, 2, 3, 4, 8, 10, 12, 13])
    bands = [[1, 3], [8, 12]]

    idxs = process_meg.get_freq_idx(bands, freq_bins)

    assert [idx.tolist() for idx in idxs] == [[1], [5]]
