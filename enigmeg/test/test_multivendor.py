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
from scipy.optimize import linear_sum_assignment

from enigmeg.process_meg import process


RANDOM_SEED = 0
MAX_TEST_DURATION = 180.0
ICA_SUBSPACE_MAX_ERROR = 0.1
ICA_MEDIAN_COMPONENT_CORRELATION = 0.8
ICA_SELECTED_COMPONENT_CORRELATION = 0.9
COVARIANCE_MAX_ERROR = 0.05
BEAMFORMER_MEDIAN_COSINE = 0.9
BEAMFORMER_MIN_COSINE = 0.5
BEAMFORMER_MIN_MATCH_FRACTION = 0.9
SPECTRA_MAX_ERROR = 0.2
SPECTRA_MIN_ROW_CORRELATION = 0.95
ALPHA_PEAK_MAX_MISSING_DISAGREEMENT = 0.1
ALPHA_PEAK_95TH_PERCENTILE_ERROR = 0.3
PARAMETERIZATION_MAX_ERROR = 0.05

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


def _row_correlations(actual, expected):
    """Return Pearson correlations between corresponding matrix rows."""
    actual = np.asarray(actual)
    expected = np.asarray(expected)
    assert actual.shape == expected.shape
    actual = actual - actual.mean(axis=1, keepdims=True)
    expected = expected - expected.mean(axis=1, keepdims=True)
    denominator = np.linalg.norm(actual, axis=1) * np.linalg.norm(
        expected,
        axis=1,
    )
    assert np.all(denominator > 0)
    return np.sum(actual * expected, axis=1) / denominator


def _load_ica_pair(proc, golden_root):
    """Load the generated and golden ICA solutions for one vendor."""
    actual = mne.preprocessing.read_ica(proc.fnames.ica)
    golden_path = _single_file(golden_root, Path(proc.fnames.ica).name)
    golden = mne.preprocessing.read_ica(golden_path)
    return actual, golden


def _match_ica_components(actual, golden):
    """Match ICA topographies without assuming stable signs or ordering."""
    assert actual.ch_names == golden.ch_names
    actual_maps = actual.get_components()
    golden_maps = golden.get_components()
    assert actual_maps.shape == golden_maps.shape

    actual_norms = np.linalg.norm(actual_maps, axis=0)
    golden_norms = np.linalg.norm(golden_maps, axis=0)
    assert np.all(actual_norms > 0)
    assert np.all(golden_norms > 0)

    correlations = np.abs(
        (actual_maps / actual_norms).T @ (golden_maps / golden_norms)
    )
    actual_indices, golden_indices = linear_sum_assignment(-correlations)
    actual_to_golden = dict(
        zip(actual_indices.tolist(), golden_indices.tolist(), strict=True)
    )
    golden_to_actual = {
        golden_index: actual_index
        for actual_index, golden_index in actual_to_golden.items()
    }
    matched_correlations = {
        actual_index: correlations[actual_index, golden_index]
        for actual_index, golden_index in actual_to_golden.items()
    }
    return SimpleNamespace(
        actual_maps=actual_maps,
        golden_maps=golden_maps,
        actual_to_golden=actual_to_golden,
        golden_to_actual=golden_to_actual,
        matched_correlations=matched_correlations,
    )


def _current_components_for_golden(
    proc,
    golden_root,
    golden_components,
):
    """Translate stable golden component identities to the current ICA order."""
    actual, golden = _load_ica_pair(proc, golden_root)
    match = _match_ica_components(actual, golden)
    current_components = []
    for golden_index in golden_components:
        actual_index = match.golden_to_actual[golden_index]
        correlation = match.matched_correlations[actual_index]
        assert correlation >= ICA_SELECTED_COMPONENT_CORRELATION, (
            f"Current ICA component {actual_index} only correlates "
            f"{correlation:.3f} with golden component {golden_index}"
        )
        current_components.append(actual_index)
    return current_components


def _find_bad_channels_maxwell_compat(raw, *args, **kwargs):
    """Handle legacy KIT FIF reflections during device-frame assessment.

    Some converted KIT files contain a reflected ``dev_head_t``. Newer MNE
    versions reject that matrix while converting it to a quaternion even when
    bad channels are assessed entirely in device coordinates. The transform is
    irrelevant in that coordinate frame, so use identity on this temporary
    assessment copy only.
    """
    if kwargs.get("coord_frame") == "meg":
        dev_head_t = raw.info.get("dev_head_t")
        if (
            dev_head_t is not None
            and np.linalg.det(dev_head_t["trans"][:3, :3]) < 0
        ):
            raw = raw.copy()
            raw.info["dev_head_t"] = mne.transforms.Transform("meg", "head")
    return _ORIGINAL_FIND_BAD_CHANNELS_MAXWELL(raw, *args, **kwargs)


_ORIGINAL_FIND_BAD_CHANNELS_MAXWELL = (
    mne.preprocessing.find_bad_channels_maxwell
)


def _crop_for_regression(raw, max_duration):
    if max_duration is not None:
        raw.crop(tmax=min(max_duration, raw.times[-1]))


def _run_pipeline(
    kwargs,
    golden_root,
    max_duration,
    regression_ica_components,
):
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
        with pytest.MonkeyPatch.context() as monkeypatch:
            monkeypatch.setattr(
                mne.preprocessing,
                "find_bad_channels_maxwell",
                _find_bad_channels_maxwell_compat,
            )
            proc.vendor_prep(megin_ignore=proc._megin_ignore)
    proc.do_ica()
    proc.do_classify_ica()
    proc.megnet_ica_comps = list(proc.ica_comps_toremove)
    # Keep downstream tests independent of model revisions while allowing MNE
    # and its numerical dependencies to reorder or flip ICA components.
    proc.ica_comps_toremove = _current_components_for_golden(
        proc,
        golden_root,
        regression_ica_components,
    )
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
        proc = _run_pipeline(
            kwargs,
            golden_root,
            max_duration,
            regression_ica_components,
        )
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
    """Regress ICA while allowing sign, order, and solver-version changes."""
    ica, ica_gt = _load_ica_pair(
        processed_vendor.proc,
        processed_vendor.golden_root,
    )

    assert ica.n_samples_ == ica_gt.n_samples_
    assert ica.method == ica_gt.method
    match = _match_ica_components(ica, ica_gt)

    # ICA can rotate within its retained PCA subspace as numerical libraries
    # evolve. Compare the physical sensor subspaces rather than raw solver
    # matrices, whose rows also have arbitrary signs and ordering.
    actual_basis, _ = np.linalg.qr(match.actual_maps, mode="reduced")
    golden_basis, _ = np.linalg.qr(match.golden_maps, mode="reduced")
    actual_projection = actual_basis @ actual_basis.T
    golden_projection = golden_basis @ golden_basis.T
    subspace_error = (
        np.linalg.norm(actual_projection - golden_projection)
        / np.linalg.norm(golden_projection)
    )
    assert subspace_error <= ICA_SUBSPACE_MAX_ERROR, (
        f"ICA sensor-subspace error {subspace_error:.6g} exceeds "
        f"{ICA_SUBSPACE_MAX_ERROR:.6g}"
    )

    median_correlation = np.median(
        list(match.matched_correlations.values())
    )
    assert median_correlation >= ICA_MEDIAN_COMPONENT_CORRELATION, (
        f"Median matched ICA topography correlation {median_correlation:.3f} "
        f"is below {ICA_MEDIAN_COMPONENT_CORRELATION:.3f}"
    )


def test_megnet_classification(processed_vendor):
    if hasattr(processed_vendor.proc, "megnet_ica_comps"):
        actual = processed_vendor.proc.megnet_ica_comps
    else:
        actual = _latest_logged_ica_components(processed_vendor)

    ica, ica_gt = _load_ica_pair(
        processed_vendor.proc,
        processed_vendor.golden_root,
    )
    match = _match_ica_components(ica, ica_gt)
    matched_golden_components = []
    for actual_index in actual:
        correlation = match.matched_correlations[actual_index]
        assert correlation >= ICA_SELECTED_COMPONENT_CORRELATION, (
            f"Classified component {actual_index} has no reliable golden "
            f"match: correlation={correlation:.3f}"
        )
        matched_golden_components.append(
            match.actual_to_golden[actual_index]
        )
    assert sorted(matched_golden_components) == sorted(
        processed_vendor.expected_megnet_components
    )


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
        covariance["data"],
        covariance_gt["data"],
        max_error=COVARIANCE_MAX_ERROR,
    )


def test_beamformer(processed_vendor):
    beamformer = mne.beamformer.read_beamformer(processed_vendor.proc.fnames.lcmv)
    beamformer_gt = mne.beamformer.read_beamformer(
        _golden_file(processed_vendor, processed_vendor.proc.fnames.lcmv)
    )
    for key in (
        "ch_names",
        "inversion",
        "is_free_ori",
        "kind",
        "n_sources",
        "pick_ori",
        "subject",
        "weight_norm",
    ):
        assert beamformer[key] == beamformer_gt[key]
    for vertices, vertices_gt in zip(
        beamformer["vertices"],
        beamformer_gt["vertices"],
        strict=True,
    ):
        np.testing.assert_array_equal(vertices, vertices_gt)

    # Max-power source orientations have arbitrary signs and can rotate when
    # covariance eigensolvers change. Compare each filter's sensor-space
    # direction up to sign instead of applying L2 to the raw weight matrix.
    weights = beamformer["weights"]
    weights_gt = beamformer_gt["weights"]
    assert weights.shape == weights_gt.shape
    weight_norms = np.linalg.norm(weights, axis=1)
    weight_norms_gt = np.linalg.norm(weights_gt, axis=1)
    assert np.all(weight_norms > 0)
    assert np.all(weight_norms_gt > 0)
    absolute_cosines = np.abs(
        np.sum(weights * weights_gt, axis=1)
        / (weight_norms * weight_norms_gt)
    )
    median_cosine = np.median(absolute_cosines)
    assert median_cosine >= BEAMFORMER_MEDIAN_COSINE, (
        f"Median beamformer direction cosine {median_cosine:.3f} is below "
        f"{BEAMFORMER_MEDIAN_COSINE:.3f}"
    )
    match_fraction = np.mean(
        absolute_cosines >= BEAMFORMER_MIN_COSINE
    )
    assert match_fraction >= BEAMFORMER_MIN_MATCH_FRACTION, (
        f"Only {match_fraction:.1%} of beamformer directions correlate at "
        f"least {BEAMFORMER_MIN_COSINE:.2f}"
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
    pd.testing.assert_index_equal(spectra.index, spectra_gt.index)
    pd.testing.assert_index_equal(spectra.columns, spectra_gt.columns)
    assert np.isfinite(spectra.to_numpy()).all()
    assert np.isfinite(spectra_gt.to_numpy()).all()
    _assert_relative_l2(
        spectra,
        spectra_gt,
        max_error=SPECTRA_MAX_ERROR,
    )
    row_correlations = _row_correlations(spectra, spectra_gt)
    minimum_correlation = row_correlations.min()
    assert minimum_correlation >= SPECTRA_MIN_ROW_CORRELATION, (
        f"Minimum label-spectrum correlation {minimum_correlation:.3f} is "
        f"below {SPECTRA_MIN_ROW_CORRELATION:.3f}"
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

    # Integrated band power remains close across beamformer and SciPy changes,
    # but it is not bitwise stable.
    band_columns = ["[1, 3]", "[3, 6]", "[8, 12]", "[13, 35]", "[35, 45]"]
    assert not relative_power[band_columns].isna().to_numpy().any()
    assert not relative_power_gt[band_columns].isna().to_numpy().any()
    _assert_relative_l2(
        relative_power[band_columns],
        relative_power_gt[band_columns],
        max_error=PARAMETERIZATION_MAX_ERROR,
    )

    # Alpha-peak fits close to FOOOF's detection threshold can appear or
    # disappear across optimizer versions. Bound both that disagreement and
    # the error where both versions detect a peak.
    alpha_peak = relative_power["AlphaPeak"]
    alpha_peak_gt = relative_power_gt["AlphaPeak"]
    missing_disagreement = np.mean(
        alpha_peak.isna() != alpha_peak_gt.isna()
    )
    assert (
        missing_disagreement <= ALPHA_PEAK_MAX_MISSING_DISAGREEMENT
    ), (
        f"AlphaPeak presence differs for {missing_disagreement:.1%} of labels"
    )
    shared_peaks = alpha_peak.notna() & alpha_peak_gt.notna()
    assert shared_peaks.any()
    alpha_peak_errors = np.abs(
        alpha_peak[shared_peaks] - alpha_peak_gt[shared_peaks]
    )
    error_95 = np.quantile(alpha_peak_errors, 0.95)
    assert error_95 <= ALPHA_PEAK_95TH_PERCENTILE_ERROR, (
        f"AlphaPeak 95th-percentile error {error_95:.3f} Hz exceeds "
        f"{ALPHA_PEAK_95TH_PERCENTILE_ERROR:.3f} Hz"
    )

    aperiodic_columns = ["AperiodicOffset", "AperiodicExponent"]
    assert not relative_power[aperiodic_columns].isna().to_numpy().any()
    assert not relative_power_gt[aperiodic_columns].isna().to_numpy().any()
    _assert_relative_l2(
        relative_power[["AperiodicOffset", "AperiodicExponent"]],
        relative_power_gt[["AperiodicOffset", "AperiodicExponent"]],
        max_error=PARAMETERIZATION_MAX_ERROR,
    )
