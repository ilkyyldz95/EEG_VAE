import numpy as np
import os
from os import listdir
from os.path import isfile, join
from shutil import copyfile
import matplotlib.pyplot as plt
import pickle

modality = 'human'
max_time_points = 1000

# Example events
with open('{}_seizure_eegs.pickle'.format(modality), 'rb') as handle:
    test_seizure_eegs = pickle.load(handle)  # each is channels x time
with open('{}_seizure_files.pickle'.format(modality), 'rb') as handle:
    test_seizure_files = pickle.load(handle)
for signal_idx in np.random.choice(range(len(test_seizure_eegs)), 5, replace=False):
    signal = test_seizure_eegs[signal_idx]
    patient_name = test_seizure_files[signal_idx].split("(")[1].split(")")[0]
    for ch in range(len(test_seizure_eegs[signal_idx])):
        plt.figure()
        _, ax = plt.subplots()
        plt.plot(signal[ch, :max_time_points], c="b", linewidth=0.5)
        plt.title("Patient {} from site {}".format(patient_name.split("_")[2], patient_name.split("_")[1]))
        ax.set(ylabel=r'$\mu V$')
        plt.savefig('./example_events/example_seizure_{}_{}.pdf'.format(signal_idx, ch),
                    bbox_inches='tight')
        plt.close()

with open('{}_pd_eegs.pickle'.format(modality), 'rb') as handle:
    test_pd_eegs = pickle.load(handle)  # each is channels x time
with open('{}_pd_files.pickle'.format(modality), 'rb') as handle:
    test_pd_files = pickle.load(handle)
for signal_idx in np.random.choice(range(len(test_pd_eegs)), 5, replace=False):
    signal = test_pd_eegs[signal_idx]
    patient_name = test_pd_files[signal_idx].split("(")[1].split(")")[0]
    for ch in range(len(test_pd_eegs[signal_idx])):
        plt.figure()
        _, ax = plt.subplots()
        plt.plot(signal[ch, :max_time_points], c="b", linewidth=0.5)
        plt.title("Patient {} from site {}".format(patient_name.split("_")[2], patient_name.split("_")[1]))
        ax.set(ylabel=r'$\mu V$')
        plt.savefig('./example_events/example_pd_{}_{}.pdf'.format(signal_idx, ch),
                    bbox_inches='tight')
        plt.close()

with open('{}_rda_eegs.pickle'.format(modality), 'rb') as handle:
    test_rda_eegs = pickle.load(handle)  # each is channels x time
with open('{}_rda_files.pickle'.format(modality), 'rb') as handle:
    test_rda_files = pickle.load(handle)
for signal_idx in np.random.choice(range(len(test_rda_eegs)), 5, replace=False):
    signal = test_rda_eegs[signal_idx]
    patient_name = test_rda_files[signal_idx].split("(")[1].split(")")[0]
    for ch in range(len(test_rda_eegs[signal_idx])):
        plt.figure()
        _, ax = plt.subplots()
        plt.plot(signal[ch, :max_time_points], c="b", linewidth=0.5)
        plt.title("Patient {} from site {}".format(patient_name.split("_")[2], patient_name.split("_")[1]))
        ax.set(ylabel=r'$\mu V$')
        plt.savefig('./example_events/example_rda_{}_{}.pdf'.format(signal_idx, ch),
                    bbox_inches='tight')
        plt.close()