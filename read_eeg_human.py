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

fs = 250.0
lowcut = 0.5
highcut = 50.0
signal_dir = "/ifs/loni/faculty/dduncan/epibios_collab/iyildiz/human_eeg/"
label_dir = "/ifs/loni/faculty/dduncan/epibios_collab/iyildiz/human_eeg_completed_reviews/"
max_signal_length_s = 3600
# For visualization
plot_duration_seconds = 300
plot_duration_threshold = 200

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

def prepare_one_signal(patient, file_name, event_windows, eegs, files,
                       lowcut, highcut, fs, event_name):
    try:
        tmp = read_raw_edf(file_name, preload=True)
        print(patient, "has signal", file_name)
        current_fs = tmp.info['sfreq']
        # Convert the read data to pandas DataFrame data format
        tmp = tmp.to_data_frame()
        # convert to numpy's unique data format, each signal may have a different length!
        # signal: channels x time points
        signal = tmp.values.T
        print("Number of channels: ", len(signal))
        for (event_start_s, event_duration_s) in event_windows:
            if event_duration_s < plot_duration_threshold:
                print("Visualizing window with before and after")
                plot_time_points = int(plot_duration_seconds * current_fs)
                T = np.arange(1, plot_time_points + 1) / (current_fs)
                plot_offset_s = (plot_duration_seconds - event_duration_s) / 2
                plot_start_s = event_start_s - plot_offset_s
                plot_end_s = event_start_s + event_duration_s + plot_offset_s
                for ch in range(len(signal)):
                    plt.figure()
                    _, ax = plt.subplots()
                    signal_plotted = signal[ch, int(current_fs * plot_start_s):
                                int(current_fs * plot_end_s)]
                    plt.plot(T, signal_plotted, c="b", linewidth=0.5)
                    plt.axvline(plot_offset_s, np.min(signal_plotted), np.max(signal_plotted)
                                , c="r", linewidth=2)
                    plt.axvline(plot_duration_seconds - plot_offset_s,
                                np.min(signal_plotted), np.max(signal_plotted)
                                , c="r", linewidth=2)
                    plt.title("Patient {} from site {}".format(patient.split("_")[2],
                                                               patient.split("_")[1]))
                    ax.set(ylabel=r'$\mu V$', xlabel='Time (s)')
                    plt.savefig('./example_events/example_{}_{}_{}.pdf'.
                                format(event_name, patient, ch), bbox_inches='tight')
                    plt.close()

            print("Slicing window", event_start_s, event_start_s + event_duration_s)
            # slice to the desired window
            sliced_signal = signal[:, int(current_fs * event_start_s):
                            int(current_fs * (event_start_s + event_duration_s))]
            # downsample if necessary
            if current_fs > 1.5 * fs:
                downsampling_factor = int(current_fs / fs)
                print(patient, "signal downsample with factor", downsampling_factor)
                sliced_signal = sliced_signal[:, ::downsampling_factor]
            # Filter in 0.5-50 Hz
            current_eeg = butter_bandpass_filter(sliced_signal, lowcut, highcut, fs)
            eegs.append(current_eeg)
            files.append(file_name)
    except (ValueError, NotImplementedError):
        print(patient, "signal cannot be loaded with", file_name)
    return eegs, files

# Move all reviewed raw .edf files to the desired directory
if not os.path.isdir(signal_dir):
    for path, subdirs, files in os.walk("/ifs/loni/faculty/dduncan/persyst_temporary_storage/TO BE REVIEWED - Already Processed"):
        for name in files:
            if ".edf" in name:
                copyfile(os.path.join(path, name), signal_dir + os.path.join(path, name).replace("/","_"))

all_label_file_names = os.listdir(label_dir)
training_signal_keys = []
test_normal_signal_dict = dict()
test_signal_keys = []
seizure_signal_dict = dict()
pd_signal_dict = dict()
rda_signal_dict = dict()

# Create label dictionaries
# Value Format: (start time in s, duration in s)
for label_file_name in all_label_file_names:
    label_start_idx = 7
    print("*** Reading", label_file_name)
    patient_name = label_file_name[label_file_name.find("3_"):label_file_name.find("3_")+9]
    print("** Patient", patient_name)
    xl_workbook = xlrd.open_workbook(os.path.join(label_dir, label_file_name))
    xl_sheet = xl_workbook.sheet_by_name('EpiBioS4Rx EEG Reviewer Form')
    # Patients with no events become training data: no review / one line review with no event
    if xl_sheet.nrows <= label_start_idx:
        print("No label found for", patient_name)
        training_signal_keys.append("(" + patient_name + ")_EEG_")
        continue
    elif xl_sheet.nrows == label_start_idx + 1:
        event_date = xl_sheet.cell(label_start_idx, 0).value
        event_start = xl_sheet.cell(label_start_idx, 1).value
        if (not isinstance(event_date, float)) or (not isinstance(event_start, float)):
            print("No label found for", patient_name)
            training_signal_keys.append("(" + patient_name + ")_EEG_")
            continue
    for label_idx in np.arange(label_start_idx, xl_sheet.nrows):
        # read date and time as they are
        event_date = xl_sheet.cell(label_idx, 0).value
        event_start = xl_sheet.cell(label_idx, 1).value
        if (not isinstance(event_date, float)) or (not isinstance(event_start, float)):
            print("No label found for row", label_idx + 1)
            continue
        # date: year/month/date -> year-month-day
        event_date = str(xlrd.xldate.xldate_as_datetime(event_date, xl_workbook.datemode)).split(" ")[0]
        # Patient & date key format: (3_17_0089)_EEG_2018-12-21
        key = "(" + patient_name + ")_EEG_" + event_date
        # time: 24 hour:minute:second -> seconds
        event_start = str(xlrd.xldate.xldate_as_datetime(event_start, xl_workbook.datemode)).split(" ")[1]
        event_start_s = float(event_start.split(":")[0]) * 3600 + float(event_start.split(":")[1]) * 60 \
                        + float(event_start.split(":")[2])
        # take the chunk until the first event of each patient & date key as normal
        if key not in test_signal_keys:
            if event_start_s > max_signal_length_s:  # at least 60 min before ictal activity
                test_normal_signal_dict[key] = [(0, event_start_s - max_signal_length_s)]
                print("* Test normal", key, test_normal_signal_dict[key])
            test_signal_keys.append(key)
        # Read event labels
        seizure_label = xl_sheet.cell(label_idx, 3).value
        pd_label = xl_sheet.cell(label_idx, 12).value
        rda_label = xl_sheet.cell(label_idx, 24).value
        if isinstance(seizure_label, float) and float(seizure_label) == 1:
            event_duration = float(xl_sheet.cell(label_idx, 10).value)
            if key in seizure_signal_dict.keys():
                seizure_signal_dict[key].append((event_start_s, event_duration))
            else:
                seizure_signal_dict[key] = [(event_start_s, event_duration)]
            print("* Seizure found for", key, seizure_signal_dict[key])
        elif isinstance(pd_label, float) and float(pd_label) == 1:
            if not isinstance(xl_sheet.cell(label_idx, 20).value, float):
                event_duration = float(xl_sheet.cell(label_idx, 21).value) * 60  # convert to s
            else:
                event_duration = float(xl_sheet.cell(label_idx, 20).value)
            if key in pd_signal_dict.keys():
                pd_signal_dict[key].append((event_start_s, event_duration))
            else:
                pd_signal_dict[key] = [(event_start_s, event_duration)]
            print("* PD found for", key, pd_signal_dict[key])
        elif isinstance(rda_label, float) and float(rda_label) == 1:
            if not isinstance(xl_sheet.cell(label_idx, 32).value, float):
                event_duration = float(xl_sheet.cell(label_idx, 33).value) * 60  # convert to s
            else:
                event_duration = float(xl_sheet.cell(label_idx, 32).value)
            if key in rda_signal_dict.keys():
                rda_signal_dict[key].append((event_start_s, event_duration))
            else:
                rda_signal_dict[key] = [(event_start_s, event_duration)]
            print("* RDA found for", key, rda_signal_dict[key])
        else:
            print("No label found for row", label_idx + 1)
print("=> Training signals reviewed", len(training_signal_keys))
print("=> Test normal signals reviewed", len(test_normal_signal_dict.keys()))
print("=> Seizure signals reviewed", np.sum([len(val_list)
                                       for val_list in seizure_signal_dict.values()]))
print("=> PD signals reviewed", np.sum([len(val_list)
                                       for val_list in pd_signal_dict.values()]))
print("=> RDA signals reviewed", np.sum([len(val_list)
                                       for val_list in rda_signal_dict.values()]))

# Read all EEG data into numpy arrays
EEG_files = [join(signal_dir, f) for f in listdir(signal_dir)
             if isfile(join(signal_dir, f)) and (".edf" in f or ".EDF" in f)]

train_eegs = []
train_files = []
test_normal_eegs = []
test_normal_files = []
seizure_eegs = []
seizure_files = []
pd_eegs = []
pd_files = []
rda_eegs = []
rda_files = []

# Read eegs as numpy arrays and bandpass filter
print("*** Processing training signals")
start_counter = 0
counter = 0
# Load previously saved checkpoint
if start_counter > 0:
    with open('human_train_eegs_{}.pickle'.format(start_counter), 'rb') as handle:
        train_eegs = pickle.load(handle)
    with open('human_train_files_{}.pickle'.format(start_counter), 'rb') as handle:
        train_files = pickle.load(handle)
for train_patient in training_signal_keys:
    for file_name in EEG_files:
        if train_patient in file_name:
            counter += 1
            if counter <= start_counter:
                continue
            try:
                tmp = read_raw_edf(file_name, preload=True)
                print(train_patient, "has signal", file_name)
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
                    print(train_patient, "signal downsample with factor", downsampling_factor)
                    signal = signal[:, ::downsampling_factor]
                # Filter in 0.5-50 Hz
                current_eeg = butter_bandpass_filter(signal, lowcut, highcut, fs)
                train_eegs.append(current_eeg)
                train_files.append(file_name)
            except (ValueError, NotImplementedError):
                print(train_patient, "signal cannot be loaded with", file_name)
            # Save checkpoints
            if counter % 10 == 9:
                print("%% Counter,", counter)
                with open('human_train_eegs_{}.pickle'.format(counter), 'wb') as handle:
                    pickle.dump(train_eegs, handle, protocol=pickle.HIGHEST_PROTOCOL)
                with open('human_train_files_{}.pickle'.format(counter), 'wb') as handle:
                    pickle.dump(train_files, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('human_train_eegs.pickle', 'wb') as handle:
    pickle.dump(train_eegs, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('human_train_files.pickle', 'wb') as handle:
    pickle.dump(train_files, handle, protocol=pickle.HIGHEST_PROTOCOL)

print("*** Processing seizure signals")
start_counter = 0
counter = 0
# Load previously saved checkpoint
if start_counter > 0:
    with open('human_seizure_eegs_{}.pickle'.format(start_counter), 'rb') as handle:
        seizure_eegs = pickle.load(handle)
    with open('human_seizure_files_{}.pickle'.format(start_counter), 'rb') as handle:
        seizure_files = pickle.load(handle)
for seizure_patient, seizure_windows in seizure_signal_dict.items():
    for file_name in EEG_files:
        if seizure_patient in file_name:
            counter += 1
            if counter <= start_counter:
                continue
            seizure_eegs, seizure_files = prepare_one_signal(seizure_patient, file_name,
                            seizure_windows, seizure_eegs, seizure_files,
                            lowcut, highcut, fs, event_name="seizure")
            # Save checkpoints
            if counter % 10 == 9:
                print("%% Counter,", counter)
                with open('human_seizure_eegs_{}.pickle'.format(counter), 'wb') as handle:
                    pickle.dump(seizure_eegs, handle, protocol=pickle.HIGHEST_PROTOCOL)
                with open('human_seizure_files_{}.pickle'.format(counter), 'wb') as handle:
                    pickle.dump(seizure_files, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('human_seizure_eegs.pickle', 'wb') as handle:
    pickle.dump(seizure_eegs, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('human_seizure_files.pickle', 'wb') as handle:
    pickle.dump(seizure_files, handle, protocol=pickle.HIGHEST_PROTOCOL)

print("*** Processing test normal signals")
start_counter = 0
counter = 0
# Load previously saved checkpoint
if start_counter > 0:
    with open('human_test_normal_eegs_{}.pickle'.format(start_counter), 'rb') as handle:
        test_normal_eegs = pickle.load(handle)
    with open('human_test_normal_files_{}.pickle'.format(start_counter), 'rb') as handle:
        test_normal_files = pickle.load(handle)
for test_normal_patient, test_normal_windows in test_normal_signal_dict.items():
    for file_name in EEG_files:
        if test_normal_patient in file_name:
            counter += 1
            if counter <= start_counter:
                continue
            test_normal_eegs, test_normal_files = prepare_one_signal(test_normal_patient, file_name,
                            test_normal_windows, test_normal_eegs, test_normal_files,
                            lowcut, highcut, fs)
            # Save checkpoints
            if counter % 10 == 9:
                print("%% Counter,", counter)
                with open('human_test_normal_eegs_{}.pickle'.format(counter), 'wb') as handle:
                    pickle.dump(test_normal_eegs, handle, protocol=pickle.HIGHEST_PROTOCOL)
                with open('human_test_normal_files_{}.pickle'.format(counter), 'wb') as handle:
                    pickle.dump(test_normal_files, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('human_test_normal_eegs.pickle', 'wb') as handle:
    pickle.dump(test_normal_eegs, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('human_test_normal_files.pickle', 'wb') as handle:
    pickle.dump(test_normal_files, handle, protocol=pickle.HIGHEST_PROTOCOL)

print("*** Processing pd signals")
start_counter = 0
counter = 0
# Load previously saved checkpoint
if start_counter > 0:
    with open('human_pd_eegs_{}.pickle'.format(start_counter), 'rb') as handle:
        pd_eegs = pickle.load(handle)
    with open('human_pd_files_{}.pickle'.format(start_counter), 'rb') as handle:
        pd_files = pickle.load(handle)
for pd_patient, pd_windows in pd_signal_dict.items():
    for file_name in EEG_files:
        if pd_patient in file_name:
            counter += 1
            if counter <= start_counter:
                continue
            pd_eegs, pd_files = prepare_one_signal(pd_patient, file_name,
                            pd_windows, pd_eegs, pd_files, lowcut, highcut, fs, event_name="pd")
            # Save checkpoints
            if counter % 10 == 9:
                print("%% Counter,", counter)
                with open('human_pd_eegs_{}.pickle'.format(counter), 'wb') as handle:
                    pickle.dump(pd_eegs, handle, protocol=pickle.HIGHEST_PROTOCOL)
                with open('human_pd_files_{}.pickle'.format(counter), 'wb') as handle:
                    pickle.dump(pd_files, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('human_pd_eegs.pickle', 'wb') as handle:
    pickle.dump(pd_eegs, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('human_pd_files.pickle', 'wb') as handle:
    pickle.dump(pd_files, handle, protocol=pickle.HIGHEST_PROTOCOL)

print("*** Processing rda signals")
start_counter = 0
counter = 0
# Load previously saved checkpoint
if start_counter > 0:
    with open('human_rda_eegs_{}.pickle'.format(start_counter), 'rb') as handle:
        rda_eegs = pickle.load(handle)
    with open('human_rda_files_{}.pickle'.format(start_counter), 'rb') as handle:
        rda_files = pickle.load(handle)
for rda_patient, rda_windows in rda_signal_dict.items():
    for file_name in EEG_files:
        if rda_patient in file_name:
            counter += 1
            if counter <= start_counter:
                continue
            rda_eegs, rda_files = prepare_one_signal(rda_patient, file_name,
                            rda_windows, rda_eegs, rda_files, lowcut, highcut, fs, event_name="rda")
            # Save checkpoints
            if counter % 10 == 9:
                print("%% Counter,", counter)
                with open('human_rda_eegs_{}.pickle'.format(counter), 'wb') as handle:
                    pickle.dump(rda_eegs, handle, protocol=pickle.HIGHEST_PROTOCOL)
                with open('human_rda_files_{}.pickle'.format(counter), 'wb') as handle:
                    pickle.dump(rda_files, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('human_rda_eegs.pickle', 'wb') as handle:
    pickle.dump(rda_eegs, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('human_rda_files.pickle', 'wb') as handle:
    pickle.dump(rda_files, handle, protocol=pickle.HIGHEST_PROTOCOL)

print("=> Training signals saved", len(train_eegs))
print("=> Test normal signals saved", len(test_normal_eegs))  # 20
print("=> Seizure signals saved", len(seizure_eegs))
print("=> PD signals saved", len(pd_eegs))    # 24
print("=> RDA signals saved", len(rda_eegs))    # 24
