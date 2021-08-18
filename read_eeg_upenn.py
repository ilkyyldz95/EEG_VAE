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
import scipy.io

fs = 500.0
lowcut = 0.5
highcut = 50.0
label_dir = "/ifs/loni/faculty/dduncan/epibios_collab/iyildiz/clipsupenn/"
max_signal_length_s = 3600

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

all_patients = [x[0].split("/")[-1] for x in os.walk(label_dir)][1:]
print("There are {} patients".format(len(all_patients)))
training_signal_keys = []
seizure_signal_keys = []
for case in all_patients:
    current_dir = os.path.join(label_dir, case)
    training_signal_keys.extend([file for file in os.listdir(current_dir) if "_interictal_" in file])
    seizure_signal_keys.extend([file for file in os.listdir(current_dir) if "_ictal_" in file])
print("There are {} training signals".format(len(training_signal_keys)))
print("There are {} seizure signals".format(len(seizure_signal_keys)))

train_eegs = []
train_files = []
seizure_eegs = []
seizure_files = []

# Read eegs as numpy arrays and bandpass filter
print("*** Processing training signals")
start_counter = 0
counter = 0
# Load previously saved checkpoint
if start_counter > 0:
    with open('upenn_train_eegs_{}.pickle'.format(start_counter), 'rb') as handle:
        train_eegs = pickle.load(handle)
    with open('upenn_train_files_{}.pickle'.format(start_counter), 'rb') as handle:
        train_files = pickle.load(handle)
for current_file in training_signal_keys:
    file_name = os.path.join(label_dir, current_file.split("_")[0] + "_" + current_file.split("_")[1],
                             current_file)
    counter += 1
    if counter <= start_counter:
        continue
    try:
        print("Reading", file_name)
        mat = scipy.io.loadmat(file_name)
        # convert to numpy's unique data format, each signal may have a different length!
        # signal: channels x time points
        signal = mat["data"]
        current_fs = mat["freq"]
        print("Number of channels: ", len(signal))
        # downsample if necessary
        if current_fs > 1.5 * fs:
            downsampling_factor = int(current_fs / fs)
            print("downsample with factor", downsampling_factor)
            signal = signal[:, ::downsampling_factor]
        # Filter in 0.5-50 Hz
        current_eeg = butter_bandpass_filter(signal, lowcut, highcut, fs)
        train_eegs.append(current_eeg)
        train_files.append(file_name)
    except (ValueError, NotImplementedError):
        print("cannot be loaded with", file_name)
    # Save checkpoints
    #if counter % 10 == 9:
    #    print("%% Counter,", counter)
    #    with open('upenn_train_eegs_{}.pickle'.format(counter), 'wb') as handle:
    #        pickle.dump(train_eegs, handle, protocol=pickle.HIGHEST_PROTOCOL)
    #    with open('upenn_train_files_{}.pickle'.format(counter), 'wb') as handle:
    #        pickle.dump(train_files, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('upenn_train_eegs.pickle', 'wb') as handle:
    pickle.dump(train_eegs, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('upenn_train_files.pickle', 'wb') as handle:
    pickle.dump(train_files, handle, protocol=pickle.HIGHEST_PROTOCOL)

print("*** Processing seizure signals")
start_counter = 0
counter = 0
# Load previously saved checkpoint
if start_counter > 0:
    with open('upenn_seizure_eegs_{}.pickle'.format(start_counter), 'rb') as handle:
        seizure_eegs = pickle.load(handle)
    with open('upenn_seizure_files_{}.pickle'.format(start_counter), 'rb') as handle:
        seizure_files = pickle.load(handle)
for current_file in seizure_signal_keys:
    file_name = os.path.join(label_dir, current_file.split("_")[0] + "_" + current_file.split("_")[1],
                             current_file)
    counter += 1
    if counter <= start_counter:
        continue
    try:
        print("Reading", file_name)
        mat = scipy.io.loadmat(file_name)
        # convert to numpy's unique data format, each signal may have a different length!
        # signal: channels x time points
        signal = mat["data"]
        current_fs = mat["freq"]
        print("Number of channels: ", len(signal))
        #event_start_s = mat["latency"]
        # downsample if necessary
        if current_fs > 1.5 * fs:
            downsampling_factor = int(current_fs / fs)
            print("downsample with factor", downsampling_factor)
            signal = signal[:, ::downsampling_factor]
        # Filter in 0.5-50 Hz
        current_eeg = butter_bandpass_filter(signal, lowcut, highcut, fs)
        seizure_eegs.append(current_eeg)
        seizure_files.append(file_name)
        #for ch in range(len(current_eeg)):
        #    plt.figure()
        #    _, ax = plt.subplots()
        #    signal_plotted = current_eeg[ch]
        #    plt.plot(signal_plotted, c="b", linewidth=0.5)
        #    ax.set(ylabel=r'$\mu V$')
        #    plt.savefig('./example_events_upenn/example_{}_{}.pdf'.
        #                format(file_name.split("/")[-1].split(".")[0], ch), bbox_inches='tight')
        #    plt.close()
    except (ValueError, NotImplementedError):
        print("cannot be loaded with", file_name)
    # Save checkpoints
    #if counter % 10 == 9:
    #    print("%% Counter,", counter)
    #    with open('upenn_seizure_eegs_{}.pickle'.format(counter), 'wb') as handle:
    #        pickle.dump(seizure_eegs, handle, protocol=pickle.HIGHEST_PROTOCOL)
    #    with open('upenn_seizure_files_{}.pickle'.format(counter), 'wb') as handle:
    #        pickle.dump(seizure_files, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('upenn_seizure_eegs.pickle', 'wb') as handle:
    pickle.dump(seizure_eegs, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('upenn_seizure_files.pickle', 'wb') as handle:
    pickle.dump(seizure_files, handle, protocol=pickle.HIGHEST_PROTOCOL)

print("=> Training signals saved", len(train_eegs))
print("=> Seizure signals saved", len(seizure_eegs))