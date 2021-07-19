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

signal_dir = "/ifs/loni/faculty/dduncan/epibios_collab/iyildiz/human_eeg/"
label_dir = "/ifs/loni/faculty/dduncan/epibios_collab/iyildiz/human_eeg_completed_reviews/"

# Move all reviewed raw .edf files to the desired directory
if not os.path.isdir(signal_dir):
    for path, subdirs, files in os.walk("/ifs/loni/faculty/dduncan/persyst_temporary_storage/TO BE REVIEWED - Already Processed"):
        for name in files:
            if ".edf" in name:
                copyfile(os.path.join(path, name), signal_dir + os.path.join(path, name).replace("/","_"))

# Find all unique patient ids
all_label_file_names = os.listdir(label_dir)
training_signal_names = []
ictal_signal_dict = dict()

# Create label dictionary
# Key format: (3_17_0089)_EEG_2018-12-21
# Value Format: (label seizure/pd/rda, start time in s, duration in s)
for label_file_name in all_label_file_names:
    label_start_idx = 7
    print("Reading", label_file_name)
    patient_name = label_file_name[label_file_name.find("3_"):label_file_name.find("3_")+9]
    print("Patient", patient_name)
    xl_workbook = xlrd.open_workbook(os.path.join(label_dir, label_file_name))
    xl_sheet = xl_workbook.sheet_by_name('EpiBioS4Rx EEG Reviewer Form')
    if xl_sheet.nrows <= label_start_idx:
        print("No label found for", label_file_name)
        training_signal_names.append(label_file_name)
        continue
    for label_idx in np.arange(label_start_idx, xl_sheet.nrows):
        event_date = xl_sheet.cell(label_idx, 0).value
        event_start = xl_sheet.cell(label_idx, 1).value
        if not isinstance(event_date, float) or not isinstance(event_start, float):
            print("No label found for", label_file_name)
            training_signal_names.append(label_file_name)
            continue
        # read date and time as they are
        # date: year/month/date -> year-month-day
        event_date = str(xlrd.xldate.xldate_as_datetime(event_date, xl_workbook.datemode)).split(" ")[0]
        key = "(" + patient_name + ")_EEG_" + event_date
        # time: 24 hour:minute:second -> seconds
        event_start = str(xlrd.xldate.xldate_as_datetime(event_start, xl_workbook.datemode)).split(" ")[1]
        event_start_s = float(event_start.split(":")[0]) * 3600 + float(event_start.split(":")[1]) * 60 \
                        + float(event_start.split(":")[2])
        # Read labels
        seizure_label = xl_sheet.cell(label_idx, 3).value
        pd_label = xl_sheet.cell(label_idx, 12).value
        rda_label = xl_sheet.cell(label_idx, 24).value
        if isinstance(seizure_label, float) and float(seizure_label) == 1:
            event_label = "seizure"
            event_duration = float(xl_sheet.cell(label_idx, 10).value)
            if key in ictal_signal_dict.keys():
                ictal_signal_dict[key].append((event_label, event_start_s, event_duration))
            else:
                ictal_signal_dict[key] = [(event_label, event_start_s, event_duration)]
            print(key, ictal_signal_dict[key])
        elif isinstance(pd_label, float) and float(pd_label) == 1:
            event_label = "pd"
            if not isinstance(xl_sheet.cell(label_idx, 20).value, float):
                event_duration = float(xl_sheet.cell(label_idx, 21).value) * 60  # convert to s
            else:
                event_duration = float(xl_sheet.cell(label_idx, 20).value)
            if key in ictal_signal_dict.keys():
                ictal_signal_dict[key].append((event_label, event_start_s, event_duration))
            else:
                ictal_signal_dict[key] = [(event_label, event_start_s, event_duration)]
            print(key, ictal_signal_dict[key])
        elif isinstance(rda_label, float) and float(rda_label) == 1:
            event_label = "rda"
            if not isinstance(xl_sheet.cell(label_idx, 32).value, float):
                event_duration = float(xl_sheet.cell(label_idx, 33).value) * 60  # convert to s
            else:
                event_duration = float(xl_sheet.cell(label_idx, 32).value)
            if key in ictal_signal_dict.keys():
                ictal_signal_dict[key].append((event_label, event_start_s, event_duration))
            else:
                ictal_signal_dict[key] = [(event_label, event_start_s, event_duration)]
            print(key, ictal_signal_dict[key])
        else:
            print("No label found for", label_file_name, event_date, event_start)

"""
modality = "rodent_eeg"  # staba and monash
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
