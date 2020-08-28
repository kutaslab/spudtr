import pytest
import numpy as np
import pandas as pd
import patsy
from spudtr import DATA_DIR, mneutils, get_demo_df


def test_streams2mne_digmont():

    eeg_streams = [
        "lle",
        "lhz",
        "MiPf",
        "LLPf",
        "RLPf",
        "LMPf",
        "RMPf",
        "LDFr",
        "RDFr",
        "LLFr",
        "RLFr",
        "LMFr",
        "RMFr",
        "LMCe",
        "RMCe",
        "MiCe",
        "MiPa",
        "LDCe",
        "RDCe",
        "LDPa",
        "RDPa",
        "LMOc",
        "RMOc",
        "LLTe",
        "RLTe",
        "LLOc",
        "RLOc",
        "MiOc",
        "rle",
        "rhz",
    ]
    montage = mneutils.streams2mne_digmont(eeg_streams)
    assert montage.ch_names == eeg_streams

    eeg_streams = [
        "lle",
        "lhz",
        "MiPf",
        "LLPf",
        "RLPf",
        "aa",
    ]

    with pytest.raises(ValueError) as excinfo:
        mneutils.streams2mne_digmont(eeg_streams)
    assert 'eeg_streams not found in cap' in str(excinfo.value)


def test_spudtr2mne():

    epochs_df = get_demo_df('sub000p3.ms100.epochs.feather')
    epoch_id = "epoch_id"
    time = "time_ms"
    _categories = "stim"
    sfreq = 500
    eeg_streams = ["MiPf", "MiCe", "MiPa", "MiOc"]

    epochs = mneutils.spudtr2mne(epochs_df, eeg_streams, time, epoch_id, sfreq)
    assert epochs.ch_names == eeg_streams


def test_spudtr2mne_epochs():

    epochs_df = get_demo_df('sub000p3.ms100.epochs.feather')
    epoch_id = "epoch_id"
    time = "time_ms"
    time_stamp = 0
    _categories = "stim"
    sfreq = 500
    eeg_streams = ["MiPf", "MiCe", "MiPa", "MiOc"]

    mne_event_id, mne_events = mneutils._categories2eventid(
        epochs_df, _categories, epoch_id, time, time_stamp
    )

    epochs = mneutils.spudtr2mne_epochs(
        epochs_df, eeg_streams, time, epoch_id, sfreq, mne_events, mne_event_id
    )

    assert epochs.event_id == {
        "stim[cal]": 1,
        "stim[standard]": 2,
        "stim[target]": 3,
    }
    assert epochs.ch_names == eeg_streams


def test__categories2eventid():

    epochs_df = get_demo_df('sub000p3.ms100.epochs.feather')
    epoch_id = "epoch_id"
    time = "time_ms"
    time_stamp = 0
    _categories = "stim"
    # sfreq = 500
    # eeg_streams = ['MiPf', 'MiCe', 'MiPa', 'MiOc']

    mne_event_id, mne_events = mneutils._categories2eventid(
        epochs_df, _categories, epoch_id, time, time_stamp
    )

    event_id_1_head = np.array(
        [[0, 0, 3], [1, 0, 3], [2, 0, 3], [3, 0, 3], [4, 0, 3]]
    )

    assert mne_event_id == {
        "stim[cal]": 1,
        "stim[standard]": 2,
        "stim[target]": 3,
    }
    assert np.array_equal(event_id_1_head, mne_events[0:5])

    # gold standard
    event_id_2_head = np.array(
        [[27, 0, 2], [28, 0, 2], [29, 0, 2], [30, 0, 2], [31, 0, 2]]
    )
    assert np.array_equal(
        event_id_2_head,
        mne_events[[v for v in map(lambda x: x[2] == 2, mne_events)]][0:5],
    )

    time_stamp = 9999
    with pytest.raises(ValueError) as excinfo:
        mneutils._categories2eventid(
            epochs_df, _categories, epoch_id, time, time_stamp
        )
    assert 'time_stamp' in str(excinfo.value)
