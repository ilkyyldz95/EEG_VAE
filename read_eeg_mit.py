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

fs = 256.0
lowcut = 0.5
highcut = 50.0
label_dir = "/ifs/loni/faculty/dduncan/epibios_collab/iyildiz/chbmit/1.0.0/"
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
            files.append(file_name)
            for ch in range(len(current_eeg)):
                plt.figure()
                _, ax = plt.subplots()
                signal_plotted = current_eeg[ch]
                plt.plot(signal_plotted[:500], c="b", linewidth=0.5)
                ax.set(ylabel=r'$\mu V$')
                plt.savefig('./example_events_mit/example_{}_{}.pdf'.
                            format(file_name.split("/")[-1].split(".")[0], ch), bbox_inches='tight')
                plt.close()
    except (ValueError, NotImplementedError):
        print("cannot be loaded with", file_name)
    return eegs, files

shortest_duration = 100000
all_cases = [x[0].split("/")[-1] for x in os.walk(label_dir)][1:]
print("There are {} cases".format(len(all_cases)))
training_signal_keys = []
seizure_signal_dict = dict()
for case in all_cases:
    current_dir = os.path.join(label_dir, case)
    # Read summary
    with open(os.path.join(current_dir, case + "-summary.txt")) as f:  # this is actually ordered in time
        lines = [line.rstrip() for line in f]
    new_seizure_found = False
    for line in lines:
        line = line.lower()
        if "file name" in line:
            current_file = line.split(" ")[-1]
            print(current_file)
        # check each signal as seizure or not
        if "number of seizures in file" in line and "0" not in line:
            new_seizure_found = True
        elif "number of seizures in file" in line and "0" in line:
            new_seizure_found = False
            training_signal_keys.append(current_file)
            print("No seizure found")
        if new_seizure_found and "start time" in line and "seizure" in line:
            print(line)
            start_time = float(line.split(":")[-1].split(" ")[1])
        elif new_seizure_found and "end time" in line and "seizure" in line:
            print(line)
            end_time = float(line.split(":")[-1].split(" ")[1])
            duration = end_time - start_time
            if duration < shortest_duration:
                shortest_duration = duration
            # each key is a file name, each value is a list of event start and event duration tuples
            # check if this file already has seizure
            if current_file in seizure_signal_dict.keys():
                seizure_signal_dict[current_file].append((start_time, end_time - start_time))
            else:
                seizure_signal_dict[current_file] = [(start_time, end_time - start_time)]
            print(seizure_signal_dict[current_file])

print("=> Training signals reviewed", len(training_signal_keys))
print("=> Seizure signals reviewed", np.sum([len(val_list)
                                       for val_list in seizure_signal_dict.values()]))
print("=> Shortest seizure duration in seconds", shortest_duration)

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
    with open('mit_train_eegs_{}.pickle'.format(start_counter), 'rb') as handle:
        train_eegs = pickle.load(handle)
    with open('mit_train_files_{}.pickle'.format(start_counter), 'rb') as handle:
        train_files = pickle.load(handle)
for current_file in training_signal_keys:
    file_name = os.path.join(label_dir, current_file.split("_")[0], current_file)
    counter += 1
    if counter <= start_counter:
        continue
    try:
        tmp = read_raw_edf(file_name, preload=True)
        print("Reading", file_name)
        current_fs = tmp.info['sfreq']
        # Convert the read data to pandas DataFrame data format
        tmp = tmp.to_data_frame()
        # convert to numpy's unique data format, each signal may have a different length!
        # signal: channels x time points
        signal = tmp.values.T
        # cut first 60 mins due to memory issues
        signal = signal[:, :min(int(max_signal_length_s * current_fs), signal.shape[-1])]
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
    if counter % 10 == 9:
        print("%% Counter,", counter)
        with open('mit_train_eegs_{}.pickle'.format(counter), 'wb') as handle:
            pickle.dump(train_eegs, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open('mit_train_files_{}.pickle'.format(counter), 'wb') as handle:
            pickle.dump(train_files, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('mit_train_eegs.pickle', 'wb') as handle:
    pickle.dump(train_eegs, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('mit_train_files.pickle', 'wb') as handle:
    pickle.dump(train_files, handle, protocol=pickle.HIGHEST_PROTOCOL)

print("*** Processing seizure signals")
start_counter = 0
counter = 0
# Load previously saved checkpoint
if start_counter > 0:
    with open('mit_seizure_eegs_{}.pickle'.format(start_counter), 'rb') as handle:
        seizure_eegs = pickle.load(handle)
    with open('mit_seizure_files_{}.pickle'.format(start_counter), 'rb') as handle:
        seizure_files = pickle.load(handle)
for current_file, seizure_windows in seizure_signal_dict.items():
    counter += 1
    if counter <= start_counter:
        continue
    file_name = os.path.join(label_dir, current_file.split("_")[0], current_file)
    seizure_eegs, seizure_files = prepare_one_signal(file_name,
                    seizure_windows, seizure_eegs, seizure_files, lowcut, highcut, fs)
    # Save checkpoints
    if counter % 10 == 9:
        print("%% Counter,", counter)
        with open('mit_seizure_eegs_{}.pickle'.format(counter), 'wb') as handle:
            pickle.dump(seizure_eegs, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open('mit_seizure_files_{}.pickle'.format(counter), 'wb') as handle:
            pickle.dump(seizure_files, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('mit_seizure_eegs.pickle', 'wb') as handle:
    pickle.dump(seizure_eegs, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('mit_seizure_files.pickle', 'wb') as handle:
    pickle.dump(seizure_files, handle, protocol=pickle.HIGHEST_PROTOCOL)

print("=> Training signals saved", len(train_eegs))
print("=> Seizure signals saved", len(seizure_eegs))