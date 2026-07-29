"""Portable regression tests for :mod:`enigmeg.process_meg`.

The tests use small, synthesized MNE objects so they do not depend on a
particular scanner, BIDS dataset, or FreeSurfer installation.
"""

from importlib import import_module
from types import SimpleNamespace

import mne
from mne_bids import BIDSPath
import munch
import numpy as np
import pandas as pd
import pytest
from scipy.signal import welch

from enigmeg import process_meg


RANDOM_SEED = 20240729


def _make_synthetic_raw(
    *,
    seed=RANDOM_SEED,
    sfreq=600.0,
    duration=12.0,
    line_frequency=60.0,
):
    """Create reproducible magnetometer data with known spectral content."""
    rng = np.random.default_rng(seed)
    times = np.arange(round(sfreq * duration)) / sfreq
    noise = 0.08e-12 * rng.standard_normal((3, times.size))
    data = np.vstack(
        [
            1.20e-12 * np.sin(2 * np.pi * 10.0 * times),
            0.90e-12 * np.sin(2 * np.pi * 20.0 * times + 0.2),
            0.60e-12 * np.sin(2 * np.pi * 35.0 * times + 0.4),
        ]
    )
    data += 0.25e-12 * np.sin(2 * np.pi * line_frequency * times)
    data += noise

    info = mne.create_info(
        ["MEG001", "MEG002", "MEG003"],
        sfreq=sfreq,
        ch_types=["mag", "mag", "mag"],
    )
    return mne.io.RawArray(data, info, verbose=False)


def _derivative_path(tmp_path, *, task="rest"):
    path = BIDSPath(
        root=tmp_path,
        subject="01",
        session="01",
        task=task,
        run="01",
        datatype="meg",
        suffix="meg",
        extension=".fif",
        check=False,
    )
    path.directory.mkdir(parents=True, exist_ok=True)
    return path


def _bare_process(raw, *, sfreq=300.0, epoch_len=2.0):
    """Build only the process state needed by numerical unit tests."""
    proc = process_meg.process.__new__(process_meg.process)
    proc.raw_rest = raw
    proc.raw_eroom = None
    proc.proc_vars = munch.Munch(
        sfreq=sfreq,
        epoch_len=epoch_len,
        mains=60.0,
        fmin=process_meg.fmin,
        fmax=process_meg.fmax,
    )
    proc._n_jobs = 1
    proc.do_dics = False
    return proc


def _power_at(data, sfreq, frequency):
    freqs, power = welch(data, fs=sfreq, nperseg=int(2 * sfreq))
    return power[np.argmin(np.abs(freqs - frequency))]


def test_synthetic_raw_is_deterministic():
    first = _make_synthetic_raw(seed=RANDOM_SEED).get_data()
    second = _make_synthetic_raw(seed=RANDOM_SEED).get_data()
    different = _make_synthetic_raw(seed=RANDOM_SEED + 1).get_data()

    np.testing.assert_array_equal(first, second)
    assert not np.array_equal(first, different)


def test_load_data_round_trip_with_synthetic_fif(tmp_path):
    raw = _make_synthetic_raw(duration=2.0)
    raw_path = tmp_path / "synthetic_raw.fif"
    raw.save(raw_path, overwrite=True, verbose=False)

    loaded = process_meg.load_data(raw_path)

    assert loaded.preload
    assert process_meg.check_datatype(raw_path) == "fif"
    np.testing.assert_allclose(loaded.get_data(), raw.get_data(), rtol=1e-7)


def test_preproc_filters_and_saves_synthetic_data(tmp_path):
    raw = _make_synthetic_raw()
    proc = _bare_process(raw)
    deriv_path = _derivative_path(tmp_path)

    proc._preproc(raw_inst=raw, deriv_path=deriv_path)

    output_path = deriv_path.copy().update(processing="filt", extension=".fif")
    assert output_path.fpath.exists()
    assert raw.info["sfreq"] == pytest.approx(300.0)

    channel = raw.get_data(picks=["MEG001"])[0]
    alpha_power = _power_at(channel, raw.info["sfreq"], 10.0)
    line_power = _power_at(channel, raw.info["sfreq"], 60.0)
    assert line_power < alpha_power * 1e-4


def test_proc_epochs_writes_deterministic_covariance(tmp_path):
    raw = _make_synthetic_raw(sfreq=300.0)
    proc = _bare_process(raw)
    deriv_path = _derivative_path(tmp_path)

    proc._proc_epochs(raw_inst=raw, deriv_path=deriv_path)

    epochs_path = deriv_path.copy().update(suffix="epo", extension=".fif")
    covariance_path = deriv_path.copy().update(suffix="cov", extension=".fif")
    epochs = mne.read_epochs(epochs_path, preload=True, verbose=False)
    covariance = mne.read_cov(covariance_path, verbose=False)

    assert len(epochs) == 6
    assert covariance.data.shape == (3, 3)
    np.testing.assert_allclose(covariance.data, covariance.data.T, atol=1e-35)
    np.testing.assert_allclose(
        np.diag(covariance.data),
        [7.58e-25, 4.42e-25, 2.17e-25],
        rtol=0.08,
    )


def test_spectral_parameterization_recovers_synthetic_alpha_peak(tmp_path):
    rng = np.random.default_rng(RANDOM_SEED)
    frequencies = np.linspace(process_meg.fmin, process_meg.fmax, process_meg.n_bins)
    labels = [SimpleNamespace(name="alpha-label"), SimpleNamespace(name="beta-label")]

    def spectrum_with_peak(peak_frequency):
        aperiodic = frequencies**-1.5
        peak = 0.16 * np.exp(-0.5 * ((frequencies - peak_frequency) / 0.7) ** 2)
        return aperiodic + peak

    base_spectra = np.vstack([spectrum_with_peak(10.0), spectrum_with_peak(20.0)])
    label_ts = np.stack(
        [
            base_spectra * (1 + rng.normal(0, 0.002, base_spectra.shape))
            for _ in range(5)
        ]
    )

    proc = process_meg.process.__new__(process_meg.process)
    proc.do_dics = False
    proc.labels = labels
    proc.label_ts = label_ts
    proc.deriv_path = _derivative_path(tmp_path)
    proc.fnames = munch.Munch(
        spectra=str(tmp_path / "spectra.csv"),
        power=str(tmp_path / "relative_power.tsv"),
    )

    proc.do_spectral_parameterization()

    spectra = pd.read_csv(proc.fnames.spectra)
    relative_power = pd.read_csv(proc.fnames.power, sep="\t", index_col=0)
    expected_spectra = np.mean(label_ts, axis=0)

    np.testing.assert_allclose(spectra.to_numpy(), expected_spectra, rtol=5e-3)
    assert relative_power.loc["alpha-label", "AlphaPeak"] == pytest.approx(
        10.0, abs=0.3
    )
    assert relative_power.loc["alpha-label", "[8, 12]"] > relative_power.loc[
        "alpha-label", "[13, 35]"
    ]
    assert relative_power.loc["beta-label", "[13, 35]"] > relative_power.loc[
        "beta-label", "[8, 12]"
    ]


def test_do_ica_passes_configured_random_seed(monkeypatch, tmp_path):
    captured = {}

    def fake_ica(raw, **kwargs):
        captured.update(kwargs)

    ica_module = import_module("MEGnet.prep_inputs.ICA")
    monkeypatch.setattr(ica_module, "main", fake_ica)

    proc = _bare_process(_make_synthetic_raw(duration=2.0))
    proc.random_seed = 8675309
    proc.bad_channels = []
    proc.meg_rest_raw = SimpleNamespace(basename="synthetic")
    proc.deriv_path = _derivative_path(tmp_path)
    proc.fnames = munch.Munch()

    proc.do_ica()

    assert captured["seedval"] == 8675309
    assert proc.fnames.ica.name == "synthetic_ica_8675309-ica.fif"


@pytest.mark.parametrize(
    ("bands", "expected"),
    [
        ([(1, 3)], [[2]]),
        ([(3, 6), (8, 12)], [[4, 5], [9, 10, 11]]),
    ],
)
def test_get_freq_idx_uses_open_band_boundaries(bands, expected):
    frequency_bins = np.arange(0, 13)

    actual = process_meg.get_freq_idx(bands, frequency_bins)

    assert [indices.tolist() for indices in actual] == expected
