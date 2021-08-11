import numpy as np
import os
from os import listdir
from os.path import isfile, join
from shutil import copyfile
import matplotlib.pyplot as plt
import pickle

fs = 250.0
duration_seconds = 60
short_duration_seconds = 2.4
modality = 'human'
max_time_points = int(duration_seconds * fs)
number_of_signals = 5
T = np.arange(1, max_time_points + 1) / (fs)

# Example events
with open('{}_seizure_eegs.pickle'.format(modality), 'rb') as handle:
    test_seizure_eegs = pickle.load(handle)  # each is channels x time
with open('{}_seizure_files.pickle'.format(modality), 'rb') as handle:
    test_seizure_files = pickle.load(handle)
permutation = np.random.permutation(range(len(test_seizure_eegs)))
signal_count = 0
for signal_idx in permutation:
    signal = test_seizure_eegs[signal_idx]
    patient_name = test_seizure_files[signal_idx].split("(")[1].split(")")[0]
    if signal.shape[-1] >= max_time_points:
        for ch in range(len(test_seizure_eegs[signal_idx])):
            plt.figure()
            _, ax = plt.subplots()
            plt.plot(T, signal[ch, :max_time_points], c="b", linewidth=0.5)
            plt.title("Patient {} from site {}".format(patient_name.split("_")[2], patient_name.split("_")[1]))
            ax.set(ylabel=r'$\mu V$', xlabel='Time (s)')
            plt.savefig('./example_events/example_seizure_{}_{}.pdf'.format(signal_idx, ch),
                        bbox_inches='tight')
            plt.close()
        short_time_points = int(short_duration_seconds * fs)
        T_short = np.arange(1, short_time_points + 1) / (fs)
        for ch in range(len(test_seizure_eegs[signal_idx])):
            plt.figure()
            _, ax = plt.subplots()
            plt.plot(T_short, signal[ch, :short_time_points], c="b", linewidth=0.5)
            plt.title("Patient {} from site {}".format(patient_name.split("_")[2], patient_name.split("_")[1]))
            ax.set(ylabel=r'$\mu V$', xlabel='Time (s)')
            plt.savefig('./example_events/example_seizure_window_{}_{}.pdf'.format(signal_idx, ch),
                        bbox_inches='tight')
            plt.close()
        signal_count += 1
        if signal_count > number_of_signals:
            break

with open('{}_pd_eegs.pickle'.format(modality), 'rb') as handle:
    test_pd_eegs = pickle.load(handle)  # each is channels x time
with open('{}_pd_files.pickle'.format(modality), 'rb') as handle:
    test_pd_files = pickle.load(handle)
permutation = np.random.permutation(range(len(test_pd_eegs)))
signal_count = 0
for signal_idx in permutation:
    signal = test_pd_eegs[signal_idx]
    patient_name = test_pd_files[signal_idx].split("(")[1].split(")")[0]
    if signal.shape[-1] >= max_time_points:
        for ch in range(len(test_pd_eegs[signal_idx])):
            plt.figure()
            _, ax = plt.subplots()
            plt.plot(T, signal[ch, :max_time_points], c="b", linewidth=0.5)
            plt.title("Patient {} from site {}".format(patient_name.split("_")[2], patient_name.split("_")[1]))
            ax.set(ylabel=r'$\mu V$', xlabel='Time (s)')
            plt.savefig('./example_events/example_pd_{}_{}.pdf'.format(signal_idx, ch),
                        bbox_inches='tight')
            plt.close()
        short_time_points = int(short_duration_seconds * fs)
        T_short = np.arange(1, short_time_points + 1) / (fs)
        for ch in range(len(test_pd_eegs[signal_idx])):
            plt.figure()
            _, ax = plt.subplots()
            plt.plot(T_short, signal[ch, :short_time_points], c="b", linewidth=0.5)
            plt.title("Patient {} from site {}".format(patient_name.split("_")[2], patient_name.split("_")[1]))
            ax.set(ylabel=r'$\mu V$', xlabel='Time (s)')
            plt.savefig('./example_events/example_pd_window_{}_{}.pdf'.format(signal_idx, ch),
                        bbox_inches='tight')
            plt.close()
        signal_count += 1
        if signal_count > number_of_signals:
            break

with open('{}_rda_eegs.pickle'.format(modality), 'rb') as handle:
    test_rda_eegs = pickle.load(handle)  # each is channels x time
with open('{}_rda_files.pickle'.format(modality), 'rb') as handle:
    test_rda_files = pickle.load(handle)
permutation = np.random.permutation(range(len(test_rda_eegs)))
signal_count = 0
for signal_idx in permutation:
    signal = test_rda_eegs[signal_idx]
    patient_name = test_rda_files[signal_idx].split("(")[1].split(")")[0]
    if signal.shape[-1] >= max_time_points:
        for ch in range(len(test_rda_eegs[signal_idx])):
            plt.figure()
            _, ax = plt.subplots()
            plt.plot(T, signal[ch, :max_time_points], c="b", linewidth=0.5)
            plt.title("Patient {} from site {}".format(patient_name.split("_")[2], patient_name.split("_")[1]))
            ax.set(ylabel=r'$\mu V$', xlabel='Time (s)')
            plt.savefig('./example_events/example_rda_{}_{}.pdf'.format(signal_idx, ch),
                        bbox_inches='tight')
            plt.close()
        short_time_points = int(short_duration_seconds * fs)
        T_short = np.arange(1, short_time_points + 1) / (fs)
        for ch in range(len(test_rda_eegs[signal_idx])):
            plt.figure()
            _, ax = plt.subplots()
            plt.plot(T_short, signal[ch, :short_time_points], c="b", linewidth=0.5)
            plt.title("Patient {} from site {}".format(patient_name.split("_")[2], patient_name.split("_")[1]))
            ax.set(ylabel=r'$\mu V$', xlabel='Time (s)')
            plt.savefig('./example_events/example_rda_window_{}_{}.pdf'.format(signal_idx, ch),
                        bbox_inches='tight')
            plt.close()
        signal_count += 1
        if signal_count > number_of_signals:
            break