from mne.io import read_raw_edf
from scipy.signal import butter, lfilter
import csv
import glob
import numpy as np
import os
import pickle

root_dir = "./edf/train"
seizures = ['fnsz', 'gnsz', 'spsz', 'tcsz', 'cpsz', 'tnsz', 'absz']
lowcut = 0.5
highcut = 50.0
fs = 250.0

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

def prepare_one_signal(file_name, event_windows, eegs, files, lowcut, highcut, fs):
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
            sliced_signal = signal[:, int(current_fs * event_start_s):int(current_fs * (event_start_s + event_duration_s))]
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

train_signal_dict = dict()
seizure_signal_dict = dict()

for edf_file in glob.iglob(os.path.join(root_dir, '**/01_tcp_ar/*.edf'), recursive=True):
    csv_file = f'{edf_file[:-4]}.csv'
    with open(csv_file, 'r') as file:
        csv_reader = csv.DictReader(filter(lambda row: row[0]!='#', file))
        csv_list_dict = list(csv_reader)

        # Retrieving all seizures timestamps for each channel
        seizures_timestamps = []
        max_stop = 0.0
        for line in csv_list_dict:
            start_time = float(line['start_time'])
            stop_time = float(line['stop_time'])
            if stop_time > max_stop:
                max_stop = stop_time
            if line['label'] in seizures:
                seizures_timestamps.append([start_time, stop_time])
        #print("max_stop:", max_stop)

        # Merging overlapping seizures timestamps
        seizures_intervals = []
        if seizures_timestamps != []:
            for start, stop in sorted(seizures_timestamps):
                if seizures_intervals and seizures_intervals[-1][1] >= start:
                    seizures_intervals[-1][1] = max(seizures_intervals[-1][1], stop)
                else:
                    seizures_intervals.append([start, stop])
        #print("seizures_intervals:", seizures_intervals)

        # Determining timestamps for background activity
        min_start = 0.0
        for interval in seizures_intervals:
            if min_start < interval[0]:
                # bckg
                if edf_file in train_signal_dict.keys():
                    train_signal_dict[edf_file].append((min_start, interval[0] - min_start))
                else:
                    train_signal_dict[edf_file] = [(min_start, interval[0] - min_start)]
            # seiz
            if edf_file in seizure_signal_dict.keys():
                seizure_signal_dict[edf_file].append((interval[0], interval[1] - interval[0]))
            else:
                seizure_signal_dict[edf_file] = [(interval[0], interval[1] - interval[0])]
            min_start = interval[1]
        if min_start != max_stop:
            # bckg
            if edf_file in train_signal_dict.keys():
                train_signal_dict[edf_file].append((min_start, max_stop - min_start))
            else:
                train_signal_dict[edf_file] = [(min_start, max_stop - min_start)]

print("=> Training signals reviewed", np.sum([len(val_list) for val_list in train_signal_dict.values()]))
print("=> Seizure signals reviewed", np.sum([len(val_list) for val_list in seizure_signal_dict.values()]))

class_ratio = np.sum([len(val_list) for val_list in seizure_signal_dict.values()]) / np.sum([len(val_list) for val_list in train_signal_dict.values()])
print("=> Class ratio", class_ratio)

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
    train_eegs, train_files = prepare_one_signal(current_file, train_windows, train_eegs, train_files, lowcut, highcut, fs)
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
    seizure_eegs, seizure_files = prepare_one_signal(current_file, seizure_windows, seizure_eegs, seizure_files, lowcut, highcut, fs)
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
