#!/usr/bin/env python3

import pandas as pd
import mne
import tkinter
import tkinter.filedialog
import yaml
from tkinter import font as tkFont
from spudtr import mneutils, DATA_DIR

root = tkinter.Tk()
helv16 = tkFont.Font(family="Helvetica", size=16, weight=tkFont.BOLD)


def OpenFile():
    global f_eeg
    f_eeg = tkinter.filedialog.askopenfilename(
        parent=root,
        initialdir=DATA_DIR,
        title="Choose file",
        filetypes=[("feather files", "*.feather"), ("all files", "*.*")],
    )
    print(f_eeg)


config_file = DATA_DIR / "default.yml"


def OpenYaml_config():
    global config_file
    config_file = tkinter.filedialog.askopenfilename(
        parent=root,
        initialdir=DATA_DIR,
        title="Choose yaml file",
        filetypes=[("yaml files", "*.yml"), ("all files", "*.*")],
    )
    print(config_file)


b1 = tkinter.Button(
    root,
    text="Select a eeg file",
    height=3,
    width=20,
    font=helv16,
    command=OpenFile,
)
b1.pack(fill="x")
b2 = tkinter.Button(
    root,
    text="Select a config yaml file",
    height=3,
    width=20,
    font=helv16,
    command=OpenYaml_config,
)
b2.pack(fill="x")


def quit(self):
    self.root.destroy()


button = tkinter.Button(
    root,
    text="Create epochs plot",
    height=3,
    width=20,
    font=helv16,
    command=root.destroy,
)
button.pack(fill="x")

root.wm_title("Spudtr epochs plot")
root.geometry("320x230")
root.mainloop()

with open(config_file, "r") as stream:
    config_data = yaml.safe_load(stream)

eeg_streams = config_data["eeg_streams"]
time = config_data["time"]
epoch_id = config_data["epoch_id"]
sfreq = config_data["sfreq"]
categories = config_data["categories"]
time_stamp = config_data["time_stamp"]
key = config_data["key"]

if f_eeg[-7:].lower() == "feather":
    epochs_df = pd.read_feather(f_eeg)
else:
    epochs_df = pd.read_hdf(f_eeg, key=key)

montage = mneutils.streams2mne_digmont(eeg_streams)
# montage.plot(kind='topomap', show_names=True);
mne_event_id, mne_events = mneutils._categories2eventid(
    epochs_df, categories, epoch_id, time, time_stamp
)
epochs = mneutils.spudtr2mne_epochs(
    epochs_df, eeg_streams, time, epoch_id, sfreq, mne_events, mne_event_id
)
epochs.plot(
    picks="eeg",
    scalings="auto",
    show=True,
    block=True,
    n_channels=10,
    n_epochs=10,
)
