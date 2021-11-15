from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import os
from mne.io import RawArray, read_raw_edf
from os import listdir
from os.path import isfile, join
from shutil import copyfile
import matplotlib.pyplot as plt
import pickle
from scipy.signal import butter, lfilter, freqz
import xlrd, datetime
from glob import glob

fs = 250.0
lowcut = 0.5
highcut = 50.0
root_dir = "/ifs/loni/faculty/dduncan/epibios_collab/iyildiz/tuheeg/"
edf_dir = root_dir + "edf/train/01_tcp_ar/"
label_dir = root_dir + "_DOCS/ref_train.txt"

def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def prepare_one_signal(file_name, event_windows, eegs, files,
                       lowcut, highcut, fs):
    try:
        tmp = read_raw_edf(file_name, preload=True)
        print("Reading", file_name)
        current_fs = tmp.info['sfreq']
        # Convert the read data to pandas DataFrame data format
        tmp = tmp.to_data_frame()
        # convert to numpy's unique data format, each signal may have a different length!
        # signal: channels x time points
        signal = tmp.values.T
        print("Number of channels: ", len(signal))
        for (event_start_s, event_duration_s) in event_windows:
            print("Slicing window", event_start_s, event_start_s + event_duration_s)
            # slice to the desired window
            sliced_signal = signal[:, int(current_fs * event_start_s):
                            int(current_fs * (event_start_s + event_duration_s))]
            # downsample if necessary
            if current_fs > 1.5 * fs:
                downsampling_factor = int(current_fs / fs)
                print("downsample with factor", downsampling_factor)
                sliced_signal = sliced_signal[:, ::downsampling_factor]
            # Filter in 0.5-50 Hz
            current_eeg = butter_bandpass_filter(sliced_signal, lowcut, highcut, fs)
            eegs.append(current_eeg)
            file_name = file_name.split("/")[-1].split(".edf")[0]
            files.append(file_name)
    except (ValueError, NotImplementedError):
        print("cannot be loaded with", file_name)
    return eegs, files

shortest_duration = 100000
all_cases = [y for x in os.walk(edf_dir) for y in glob(os.path.join(x[0], '*.edf'))]
print("There are {} cases".format(len(all_cases)))
train_signal_dict = dict()
seizure_signal_dict = dict()
with open(label_dir) as f:
    lines = [line.rstrip() for line in f]
for line in lines:
    edf_file_found = False
    current_file, start_time, end_time, label, _ = line.split(" ")
    for case in all_cases:
        if current_file in case:
            current_file = case
            print("Reading {} file name {}".format(label, current_file))
            edf_file_found = True
            break
    if not edf_file_found:
        print("CANNOT read {} file name {}".format(label, current_file))
        continue
    start_time, end_time = float(start_time), float(end_time)
    if label == "bckg":
        # each key is a file name, each value is a list of event start and event duration tuples
        if current_file in train_signal_dict.keys():
            train_signal_dict[current_file].append((start_time, end_time - start_time))
        else:
            train_signal_dict[current_file] = [(start_time, end_time - start_time)]
        print(train_signal_dict[current_file])
    elif label == "seiz":
        duration = end_time - start_time
        if duration < shortest_duration:
            shortest_duration = duration
        if current_file in seizure_signal_dict.keys():
            seizure_signal_dict[current_file].append((start_time, end_time - start_time))
        else:
            seizure_signal_dict[current_file] = [(start_time, end_time - start_time)]
        print(seizure_signal_dict[current_file])

print("=> Training signals reviewed", np.sum([len(val_list)
                                       for val_list in train_signal_dict.values()]))
print("=> Seizure signals reviewed", np.sum([len(val_list)
                                       for val_list in seizure_signal_dict.values()]))
class_ratio = np.sum([len(val_list)
                                       for val_list in seizure_signal_dict.values()]) / \
                np.sum([len(val_list)
                                       for val_list in train_signal_dict.values()])
print("=> Class ratio", class_ratio)
print("=> Shortest seizure duration in seconds", shortest_duration)
"""
=> Training signals reviewed 4406
=> Seizure signals reviewed 1229
=> Shortest seizure duration in seconds 1.8515999999999906
"""

train_eegs = []
train_files = []
seizure_eegs = []
seizure_files = []

max_counter = 190
print("*** Processing train signals")
start_counter = 0
counter = 0
# Load previously saved checkpoint
if start_counter > 0:
    with open('tuh_train_eegs_{}.pickle'.format(start_counter), 'rb') as handle:
        train_eegs = pickle.load(handle)
    with open('tuh_train_files_{}.pickle'.format(start_counter), 'rb') as handle:
        train_files = pickle.load(handle)
for current_file, train_windows in train_signal_dict.items():
    counter += 1
    if counter <= start_counter or counter >= max_counter:
        continue
    file_name = os.path.join(label_dir, current_file.split("_")[0], current_file)
    train_eegs, train_files = prepare_one_signal(file_name,
                    train_windows, train_eegs, train_files, lowcut, highcut, fs)
    # Save checkpoints
    if counter % 10 == 9:
        print("%% Counter,", counter)
        with open('tuh_train_eegs_{}.pickle'.format(counter), 'wb') as handle:
            pickle.dump(train_eegs, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open('tuh_train_files_{}.pickle'.format(counter), 'wb') as handle:
            pickle.dump(train_files, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('tuh_train_eegs.pickle', 'wb') as handle:
    pickle.dump(train_eegs, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('tuh_train_files.pickle', 'wb') as handle:
    pickle.dump(train_files, handle, protocol=pickle.HIGHEST_PROTOCOL)

print("*** Processing seizure signals")
start_counter = 0
counter = 0
# Load previously saved checkpoint
if start_counter > 0:
    with open('tuh_seizure_eegs_{}.pickle'.format(start_counter), 'rb') as handle:
        seizure_eegs = pickle.load(handle)
    with open('tuh_seizure_files_{}.pickle'.format(start_counter), 'rb') as handle:
        seizure_files = pickle.load(handle)
for current_file, seizure_windows in seizure_signal_dict.items():
    counter += 1
    if counter <= start_counter or counter >= max_counter:
        continue
    file_name = os.path.join(label_dir, current_file.split("_")[0], current_file)
    seizure_eegs, seizure_files = prepare_one_signal(file_name,
                    seizure_windows, seizure_eegs, seizure_files, lowcut, highcut, fs)
    # Save checkpoints
    if counter % 10 == 9:
        print("%% Counter,", counter)
        with open('tuh_seizure_eegs_{}.pickle'.format(counter), 'wb') as handle:
            pickle.dump(seizure_eegs, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open('tuh_seizure_files_{}.pickle'.format(counter), 'wb') as handle:
            pickle.dump(seizure_files, handle, protocol=pickle.HIGHEST_PROTOCOL)
# preserve class ratio
if len(seizure_files) / len(train_files) > class_ratio:
    seizure_eegs = seizure_eegs[:int(len(train_eegs) * class_ratio)]
    seizure_files = seizure_files[:int(len(train_files) * class_ratio)]

with open('tuh_seizure_eegs.pickle', 'wb') as handle:
    pickle.dump(seizure_eegs, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('tuh_seizure_files.pickle', 'wb') as handle:
    pickle.dump(seizure_files, handle, protocol=pickle.HIGHEST_PROTOCOL)

print("=> Training signals saved", len(train_eegs))
print("=> Seizure signals saved", len(seizure_eegs))