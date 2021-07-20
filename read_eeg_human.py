from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import os
import time
from mne.io import RawArray, read_raw_edf
from os import listdir
from os.path import isfile, join
from shutil import copyfile
import matplotlib.pyplot as plt
import pickle
from scipy.signal import butter, lfilter, freqz
import xlrd, datetime

modality = "human"
fs = 2000.0
lowcut = 0.5
highcut = 50.0

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

signal_dir = "/ifs/loni/faculty/dduncan/epibios_collab/iyildiz/human_eeg/"
label_dir = "/ifs/loni/faculty/dduncan/epibios_collab/iyildiz/human_eeg_completed_reviews/"

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
        training_signal_keys.append(patient_name)
        continue
    elif xl_sheet.nrows == label_start_idx + 1:
        event_date = xl_sheet.cell(label_start_idx, 0).value
        event_start = xl_sheet.cell(label_start_idx, 1).value
        if (not isinstance(event_date, float)) or (not isinstance(event_start, float)):
            print("No label found for", patient_name)
            training_signal_keys.append(patient_name)
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
            test_normal_signal_dict[key] = (0, event_start_s)
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
print("Training signals count", len(training_signal_keys))
print("Test normal signals count", len(test_normal_signal_dict.keys()))
print("Seizure signals count", np.sum([len(val_list)
                                       for val_list in seizure_signal_dict.values()]))
print("PD signals count", np.sum([len(val_list)
                                       for val_list in pd_signal_dict.values()]))
print("RDA signals count", np.sum([len(val_list)
                                       for val_list in rda_signal_dict.values()]))


"""
# Read all EEG data
input_dir = "../{}".format(modality)
EEG_files = [join(input_dir, f) for f in listdir(input_dir) if isfile(join(input_dir, f)) and
             (".edf" in f or ".EDF" in f)]
print("{} EEG files found for modality {}".format(len(EEG_files), modality))

all_eegs = []
all_files = []
for file_name in EEG_files:
    tmp = read_raw_edf(file_name, preload=True)
    # Convert the read data to pandas DataFrame data format
    tmp = tmp.to_data_frame()
    # convert to numpy's unique data format, each signal may have a different length!
    # channels x time points
    current_eeg = tmp.values.T
    all_eegs.append(current_eeg)
    all_files.append(file_name)
    print("Number of channels: ", len(current_eeg))

# Save raw EEG data
with open('{}_all_raw_eegs.pickle'.format(modality), 'wb') as handle:
    pickle.dump(all_eegs, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('{}_all_eeg_files.pickle'.format(modality), 'wb') as handle:
    pickle.dump(all_files, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Bandpass filter all EEG data
prep_eegs = []
for signal in all_eegs:
    y = butter_bandpass_filter(signal, lowcut, highcut, fs)
    # check filtering success
    #n = len(y)  # length of the signal
    #k = np.arange(n)
    #T = n / fs
    #frq = k / T  # two sides frequency range
    #frq = frq[:len(frq) // 2]  # one side frequency range
    #Y = np.fft.fft(y) / n  # dft and normalization
    #Y = np.abs(Y[:n // 2])
    #print("Output Frequencies:", frq)
    #print("Output Amplitudes:", Y)
    prep_eegs.append(y)
    print("{} signals filtered".format(len(prep_eegs)))

# Save preprocessed EEG data
with open('{}_preprocessed_eegs.pickle'.format(modality), 'wb') as handle:
    pickle.dump(prep_eegs, handle, protocol=pickle.HIGHEST_PROTOCOL)
"""
