"""Regression tests for the ENIGMA-MEG pipeline using real scanner data."""

import ast
import glob
import os
import os.path as op
from pathlib import Path
import random
from types import SimpleNamespace
import warnings

import mne
import numpy as np
import pandas as pd
import pytest

from enigmeg.process_meg import process


RANDOM_SEED = 0
MAX_TEST_DURATION = 180.0

# A single worker avoids run-to-run differences in reductions performed by MNE
# and its numerical dependencies. MEGnet receives RANDOM_SEED through process.
os.environ["n_jobs"] = "1"
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")

pytestmark = [pytest.mark.meg, pytest.mark.slow]

test_root = os.environ.get("ENIGMA_TEST_DIR")
if test_root is None:
    pytest.skip(
        "ENIGMA_TEST_DIR is required for the real-data multivendor tests",
        allow_module_level=True,
    )
enigma_test_dir = op.join(test_root, "enigma_test_data")


elekta_kwargs = {
    "subject": "CC110101",
    "bids_root": op.join(enigma_test_dir, "CAMCAN"),
    "run": "01",
    "session": "01",
    "mains": 50,
    "rest_tagname": "rest",
    "emptyroom_tagname": "emptyroom",
    "random_seed": RANDOM_SEED,
}

ctf_kwargs = {
    "subject": "A2021",
    "bids_root": op.join(enigma_test_dir, "MOUS"),
    "run": None,
    "session": None,
    "mains": 50,
    "rest_tagname": "rest",
    "emptyroom_tagname": None,
    "random_seed": RANDOM_SEED,
}

kit_kwargs = {
    "subject": "0001",
    "bids_root": op.join(enigma_test_dir, "YOKOGOWA"),
    "run": "1",
    "session": "1",
    "mains": 50,
    "rest_tagname": "eyesclosed",
    "emptyroom_tagname": None,
    "random_seed": RANDOM_SEED,
}

# The CAMCAN and MOUS regression outputs were generated from the 180-second
# crop. The YOKOGOWA golden outputs use the complete recording.
VENDOR_CASES = [
    (
        elekta_kwargs,
        "CAMCAN_crop",
        MAX_TEST_DURATION,
        [11, 15],
        [11, 15],
    ),
    (ctf_kwargs, "MOUS_crop", MAX_TEST_DURATION, [12], [12]),
    (
        kit_kwargs,
        "YOKOGOWA",
        None,
        [5, 14],
        [0, 5, 14],
    ),
]


def _single_file(topdir, basename):
    """Return one exact regression artifact with an informative failure."""
    matches = sorted(
        glob.glob(op.join(str(topdir), "**", basename), recursive=True)
    )
    assert len(matches) == 1, (
        f"Expected exactly one {basename!r} below {topdir}, found {matches}"
    )
    assert op.exists(matches[0]), f"DataLad content is unavailable for {matches[0]}"
    return matches[0]


def _golden_file(vendor, generated_file, *, allow_uncropped=False):
    basename = Path(generated_file).name
    matches = sorted(
        glob.glob(op.join(str(vendor.golden_root), "**", basename), recursive=True)
    )
    assert len(matches) == 1, (
        f"Expected exactly one {basename!r} below {vendor.golden_root}, "
        f"found {matches}"
    )
    if op.exists(matches[0]):
        return matches[0]

    if allow_uncropped and vendor.uncropped_golden_root != vendor.golden_root:
        fallback = sorted(
            glob.glob(
                op.join(str(vendor.uncropped_golden_root), "**", basename),
                recursive=True,
            )
        )
        assert len(fallback) == 1, (
            f"Expected exactly one fallback {basename!r} below "
            f"{vendor.uncropped_golden_root}, found {fallback}"
        )
        if op.exists(fallback[0]):
            return fallback[0]

    pytest.skip(
        f"DataLad content is unavailable for golden artifact {matches[0]}; "
        "materialize the regression dataset with `datalad get`"
    )


def _assert_relative_l2(actual, expected, *, max_error):
    """Compare numerical artifacts relative to the total expected magnitude."""
    actual = np.asarray(actual)
    expected = np.asarray(expected)
    assert actual.shape == expected.shape
    expected_norm = np.linalg.norm(expected)
    assert expected_norm > 0
    relative_error = np.linalg.norm(actual - expected) / expected_norm
    assert relative_error <= max_error, (
        f"Relative L2 error {relative_error:.6g} exceeds {max_error:.6g}"
    )


def _crop_for_regression(raw, max_duration):
    if max_duration is not None:
        raw.crop(tmax=min(max_duration, raw.times[-1]))


def _run_pipeline(kwargs, max_duration, regression_ica_components):
    """Run the same deterministic processing stages for each scanner vendor."""
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    proc = process(**kwargs)
    proc.load_data()
    _crop_for_regression(proc.raw_rest, max_duration)
    if proc.raw_eroom is not None:
        _crop_for_regression(proc.raw_eroom, max_duration)

    with warnings.catch_warnings():
        # This recording contains intervals without enough acceptable cHPI
        # fits. MNE emits one warning per 10 ms interval, which can overwhelm
        # pytest output without adding regression information.
        warnings.filterwarnings(
            "ignore",
            message=r".*good HPI fits, cannot determine the transformation.*",
            category=RuntimeWarning,
        )
        proc.vendor_prep(megin_ignore=proc._megin_ignore)
    proc.do_ica()
    proc.do_classify_ica()
    proc.megnet_ica_comps = list(proc.ica_comps_toremove)
    # Downstream numerical golden files must use a fixed component set. This
    # keeps those tests independent of classifier/model revisions, which are
    # regressed separately in test_megnet_classification.
    proc.ica_comps_toremove = list(regression_ica_components)
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
    return proc


@pytest.fixture(scope="module", params=VENDOR_CASES, ids=["MEGIN", "CTF", "KIT"])
def processed_vendor(request):
    """Process each real dataset once and share its outputs across assertions."""
    (
        kwargs,
        golden_dataset,
        max_duration,
        expected_megnet_components,
        regression_ica_components,
    ) = request.param
    bids_root = Path(kwargs["bids_root"])
    golden_root = (
        Path(enigma_test_dir)
        / "all_derivatives"
        / golden_dataset
        / "ENIGMA_MEG"
    )
    assert bids_root.is_dir(), f"Real test data are unavailable: {bids_root}"
    assert golden_root.is_dir(), f"Golden outputs are unavailable: {golden_root}"

    if os.environ.get("ENIGMA_MEG_REUSE_DERIVATIVES") == "1":
        proc = process(**kwargs)
        proc._set_ica_fnames()
    else:
        proc = _run_pipeline(kwargs, max_duration, regression_ica_components)
    return SimpleNamespace(
        kwargs=kwargs,
        proc=proc,
        max_duration=max_duration,
        expected_megnet_components=expected_megnet_components,
        regression_ica_components=regression_ica_components,
        bids_root=bids_root,
        deriv_root=bids_root / "derivatives",
        enigma_root=bids_root / "derivatives" / "ENIGMA_MEG",
        golden_root=golden_root,
        uncropped_golden_root=(
            Path(enigma_test_dir)
            / "all_derivatives"
            / bids_root.name
            / "ENIGMA_MEG"
        ),
    )


def _latest_logged_ica_components(processed_vendor):
    subject = processed_vendor.kwargs["subject"]
    logfile = _single_file(
        processed_vendor.enigma_root / "logs",
        f"{subject}_*_log.txt",
    )
    with open(logfile, encoding="utf-8") as log_stream:
        matching_lines = [
            line for line in log_stream if "Components to reject:" in line
        ]
    assert matching_lines, f"No MEGnet classification was logged in {logfile}"
    return ast.literal_eval(
        matching_lines[-1].split("Components to reject:", maxsplit=1)[1].strip()
    )


def test_pipeline_outputs_exist(processed_vendor):
    proc = processed_vendor.proc
    expected_outputs = [
        "rest_filt",
        "rest_epo",
        "rest_cov",
        "rest_fwd",
        "rest_trans",
        "bem",
        "src",
        "lcmv",
        "spectra",
        "power",
        "parc",
    ]
    if processed_vendor.kwargs["emptyroom_tagname"] is not None:
        expected_outputs.extend(["eroom_filt", "eroom_epo", "eroom_cov"])

    missing = [
        proc.fnames[name]
        for name in expected_outputs
        if not op.exists(proc.fnames[name])
    ]
    assert not missing, f"Pipeline did not create expected outputs: {missing}"
    assert proc.random_seed == RANDOM_SEED


def test_mcorr_outputs(processed_vendor):
    if processed_vendor.kwargs["subject"] != elekta_kwargs["subject"]:
        pytest.skip("Movement compensation is specific to the MEGIN test case")

    target = (
        f"sub-{processed_vendor.kwargs['subject']}_ses-01_meg_run-01_headpos.npy"
    )
    headpos = np.load(_single_file(processed_vendor.enigma_root, target))
    headpos_gt = np.load(_single_file(processed_vendor.golden_root, target))
    np.testing.assert_allclose(headpos, headpos_gt, atol=1e-3, rtol=1e-7)


def test_src(processed_vendor):
    src = mne.read_source_spaces(processed_vendor.proc.fnames.src)
    src_gt = mne.read_source_spaces(
        _golden_file(
            processed_vendor,
            processed_vendor.proc.fnames.src,
            allow_uncropped=True,
        )
    )

    assert len(src) == len(src_gt) == 2
    for hemisphere, hemisphere_gt in zip(src, src_gt, strict=True):
        np.testing.assert_allclose(hemisphere["rr"], hemisphere_gt["rr"])
        np.testing.assert_allclose(hemisphere["nn"], hemisphere_gt["nn"])
        np.testing.assert_array_equal(hemisphere["vertno"], hemisphere_gt["vertno"])


def test_bem(processed_vendor):
    bem = mne.read_bem_solution(processed_vendor.proc.fnames.bem)
    bem_gt = mne.read_bem_solution(
        _golden_file(processed_vendor, processed_vendor.proc.fnames.bem)
    )

    assert type(bem) is type(bem_gt)
    _assert_relative_l2(bem["solution"], bem_gt["solution"], max_error=1e-9)


def test_fwd(processed_vendor):
    fwd = mne.read_forward_solution(processed_vendor.proc.fnames.rest_fwd)
    fwd_gt = mne.read_forward_solution(
        _golden_file(
            processed_vendor,
            processed_vendor.proc.fnames.rest_fwd,
        )
    )
    assert fwd["sol"]["row_names"] == fwd_gt["sol"]["row_names"]
    _assert_relative_l2(
        fwd["sol"]["data"], fwd_gt["sol"]["data"], max_error=1e-6
    )


def test_ica_solution(processed_vendor):
    """Regress the seeded ICA decomposition before downstream cleaning."""
    ica = mne.preprocessing.read_ica(processed_vendor.proc.fnames.ica)
    ica_gt = mne.preprocessing.read_ica(
        _golden_file(processed_vendor, processed_vendor.proc.fnames.ica)
    )

    assert ica.ch_names == ica_gt.ch_names
    assert ica.n_samples_ == ica_gt.n_samples_
    assert ica.method == ica_gt.method
    _assert_relative_l2(
        ica.unmixing_matrix_,
        ica_gt.unmixing_matrix_,
        max_error=0.02,
    )


def test_megnet_classification(processed_vendor):
    if hasattr(processed_vendor.proc, "megnet_ica_comps"):
        actual = processed_vendor.proc.megnet_ica_comps
    else:
        actual = _latest_logged_ica_components(processed_vendor)
    assert actual == processed_vendor.expected_megnet_components


def test_aparcsub(processed_vendor):
    """Verify that both generated hemispheres contain usable unique labels."""
    subject = f"sub-{processed_vendor.kwargs['subject']}"
    subjects_dir = processed_vendor.deriv_root / "freesurfer" / "subjects"

    for hemi in ("lh", "rh"):
        labels = mne.read_labels_from_annot(
            subject,
            parc="aparc_sub",
            subjects_dir=subjects_dir,
            hemi=hemi,
        )
        names = [label.name for label in labels]
        assert labels, f"No {hemi} labels were generated"
        assert len(names) == len(set(names))
        assert all(label.hemi == hemi for label in labels)
        assert all(label.vertices.size > 0 for label in labels)


def test_transform(processed_vendor):
    trans = mne.read_trans(processed_vendor.proc.fnames.rest_trans)
    trans_gt = mne.read_trans(
        _golden_file(processed_vendor, processed_vendor.proc.fnames.rest_trans)
    )
    np.testing.assert_allclose(trans["trans"], trans_gt["trans"])
    assert trans["to"] == trans_gt["to"]
    assert trans["from"] == trans_gt["from"]


def test_covariance(processed_vendor):
    covariance = mne.read_cov(processed_vendor.proc.fnames.rest_cov)
    covariance_gt = mne.read_cov(
        _golden_file(processed_vendor, processed_vendor.proc.fnames.rest_cov)
    )
    assert covariance.ch_names == covariance_gt.ch_names
    _assert_relative_l2(
        covariance["data"], covariance_gt["data"], max_error=0.01
    )


def test_beamformer(processed_vendor):
    beamformer = mne.beamformer.read_beamformer(processed_vendor.proc.fnames.lcmv)
    beamformer_gt = mne.beamformer.read_beamformer(
        _golden_file(processed_vendor, processed_vendor.proc.fnames.lcmv)
    )
    _assert_relative_l2(
        beamformer["weights"], beamformer_gt["weights"], max_error=0.02
    )


def test_logfile_records_successful_run(processed_vendor):
    subject = processed_vendor.kwargs["subject"]
    logfile = _single_file(
        processed_vendor.enigma_root / "logs",
        f"{subject}_*_log.txt",
    )
    with open(logfile, encoding="utf-8") as log_stream:
        lines = log_stream.readlines()

    completions = [
        index
        for index, line in enumerate(lines)
        if "do_mri_segstats :: COMPLETED" in line
    ]
    assert completions, f"No completed pipeline run was logged in {logfile}"
    latest_completion = completions[-1]
    starts = [
        index
        for index, line in enumerate(lines)
        if (
            "Initializing subject level enigma log" in line
            and index < latest_completion
        )
    ]
    assert starts, f"No pipeline initialization was logged in {logfile}"
    latest_run = lines[starts[-1] : latest_completion + 1]
    assert not any(" :: ERROR :: " in line for line in latest_run)
    assert any("do_spectral_parameterization :: COMPLETED" in line for line in latest_run)
    assert any("do_mri_segstats :: COMPLETED" in line for line in latest_run)


def test_spectra_outputs(processed_vendor):
    spectra = pd.read_csv(processed_vendor.proc.fnames.spectra)
    spectra_gt = pd.read_csv(
        _golden_file(processed_vendor, processed_vendor.proc.fnames.spectra)
    )
    pd.testing.assert_frame_equal(
        spectra,
        spectra_gt,
        check_exact=False,
        atol=1e-4,
        rtol=1e-7,
    )


def test_fooof_outputs(processed_vendor):
    relative_power = pd.read_csv(
        processed_vendor.proc.fnames.power, sep="\t", index_col=0
    )
    relative_power_gt = pd.read_csv(
        _golden_file(processed_vendor, processed_vendor.proc.fnames.power),
        sep="\t",
        index_col=0,
    )
    pd.testing.assert_index_equal(relative_power.index, relative_power_gt.index)
    pd.testing.assert_index_equal(
        relative_power.columns, relative_power_gt.columns
    )
    np.testing.assert_array_equal(
        relative_power.isna(), relative_power_gt.isna()
    )

    # Integrated band power is numerically stable. FOOOF's fitted parameters
    # can move slightly across SciPy releases, so compare them in their natural
    # units instead of applying one permissive tolerance to every column.
    band_columns = ["[1, 3]", "[3, 6]", "[8, 12]", "[13, 35]", "[35, 45]"]
    np.testing.assert_allclose(
        relative_power[band_columns],
        relative_power_gt[band_columns],
        atol=1e-6,
        rtol=1e-7,
    )
    np.testing.assert_allclose(
        relative_power["AlphaPeak"],
        relative_power_gt["AlphaPeak"],
        atol=0.1,
        rtol=1e-7,
        equal_nan=True,
    )
    np.testing.assert_allclose(
        relative_power[["AperiodicOffset", "AperiodicExponent"]],
        relative_power_gt[["AperiodicOffset", "AperiodicExponent"]],
        atol=0.002,
        rtol=1e-7,
        equal_nan=True,
    )
