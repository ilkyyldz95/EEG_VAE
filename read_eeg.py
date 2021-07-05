from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import os
import time
from mne.io import RawArray, read_raw_edf
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt
import pickle
from scipy.signal import butter, lfilter, freqz

modality = "rodent_eeg"
fs = 2000.0
lowcut = 0.5
highcut = 50.0

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


# Read all EEG data
input_dir = "../{}".format(modality)
EEG_files = [join(input_dir, f) for f in listdir(input_dir) if isfile(join(input_dir, f)) and (".edf" in f)]
print("{} EEG files found for modality {}".format(len(EEG_files), modality))

all_eegs = []
all_files = []
for file_name in EEG_files:
    if "br_raw" in file_name:
        tmp = read_raw_edf(file_name)
    else:
        tmp = read_raw_edf(file_name, preload=True)
    # Convert the read data to pandas DataFrame data format
    tmp = tmp.to_data_frame()
    # convert to numpy's unique data format, each signal may have a different length!
    # channels x time points
    current_eeg = tmp.values.T
    all_eegs.append(current_eeg)
    all_files.append(file_name)

# Save raw EEG data
with open('{}_all_raw_eegs.pickle'.format(modality), 'wb') as handle:
    pickle.dump(all_eegs, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('{}_all_eeg_files.pickle'.format(modality), 'wb') as handle:
    pickle.dump(all_files, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Bandpass filter all EEG data
prep_eegs = []
for signal in all_eegs:
    y = butter_bandpass_filter(signal, lowcut, highcut, fs)
    prep_eegs.append(y)
    print("{} signals filtered".format(len(prep_eegs)))

# Save preprocessed EEG data
with open('{}_preprocessed_eegs.pickle'.format(modality), 'wb') as handle:
    pickle.dump(prep_eegs, handle, protocol=pickle.HIGHEST_PROTOCOL)

