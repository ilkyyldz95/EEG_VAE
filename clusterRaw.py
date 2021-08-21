from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import os
import time
import pickle
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.cluster import AgglomerativeClustering
from scipy.signal import spectrogram
from scipy.stats import ttest_ind
from sklearn.metrics import roc_curve, accuracy_score, roc_auc_score, f1_score, precision_score, recall_score, classification_report
import matplotlib

modality = "mit"
img_size = 58368  # 58368 maximum to cover the shortest activity of 6 secs
n_channels = 38
sub_window_size = int(img_size / n_channels)  # sub_window_size / fs second window
downsample_factor = 2
print("{} channels with window size {}".format(n_channels, sub_window_size))

def extract_windows_vectorized(array, sub_window_size, downsample_factor, overlap_factor=0.5):
    # create sliding windows of size sub_window_size, downsampling by downsample_factor, and overlapping by overlap_factor percent
    sub_windows = (
        # expand_dims are used to convert a 1D array to 2D array.
        np.expand_dims(np.arange(0, sub_window_size * downsample_factor, downsample_factor), 0) +
        np.expand_dims(np.arange(0, array.shape[-1] - sub_window_size * downsample_factor + 1,
                                 int((1-overlap_factor) * sub_window_size * downsample_factor)), 0).T)
    return array[sub_windows]

def apply_sliding_window(files, eegs):
    prep_eegs = []
    prep_files = []
    for file_name, signal in zip(files, eegs):
        # signal shape: channels x time points
        print("Input signal shape channels x time points:", signal.shape)
        current_signals = []
        for channel_index in range(signal.shape[0]):
            signal_per_channel_sliding = extract_windows_vectorized(signal[channel_index], sub_window_size, downsample_factor)
            current_signals.append(signal_per_channel_sliding)
        # replicate through channels if there are less channels than max n_channels
        current_signals = np.array(current_signals)
        print("Sliding signal shape channels x batch x time points:", current_signals.shape)
        if signal.shape[0] < n_channels:
            current_signals = np.tile(current_signals,
                                      [int(np.ceil(n_channels/signal.shape[0])), 1, 1])
            current_signals = current_signals[:n_channels]
        # batch x channels x time points
        current_signals = current_signals.transpose((1, 0, 2))
        print("Sliding output signal shape batch x channels x time points:", current_signals.shape)
        # take file name only
        file_name = file_name.split("/")[-1].split(".edf")[0]
        current_file_names = np.tile([file_name], (len(current_signals),))
        prep_eegs.extend(current_signals)
        prep_files.extend(current_file_names)
    prep_eegs = np.array(prep_eegs)
    print("Dataset shape:", prep_eegs.shape)
    prep_files = np.array(prep_files)
    return prep_eegs, prep_files

# Load train EEG data with overlapping sliding windows
with open('{}_train_eegs.pickle'.format(modality), 'rb') as handle:
    train_eegs = pickle.load(handle)
with open('{}_train_files.pickle'.format(modality), 'rb') as handle:
    train_files = pickle.load(handle)
# filter nans
train_eegs_cleaned = []
train_files_cleaned = []
count = 0
for idx in range(len(train_eegs)):
    if np.any(np.isnan(train_eegs[idx])):
        count += 1
    else:
        train_eegs_cleaned.append(train_eegs[idx])
        train_files_cleaned.append(train_files[idx])
print("{} train eegs are cleaned".format(count))
train_prep_eegs, train_files = apply_sliding_window(train_files_cleaned, train_eegs_cleaned)
# load dataset via min-max normalization & add image channel dimension
train_imgs = np.array([img.flatten() for img in train_prep_eegs])  # batch x channels * time points
print("Train images shape:", train_imgs.shape)

# separate normal signals into train and test portions
shuffled_idx = range(len(train_imgs))
train_idx = shuffled_idx[:int(len(shuffled_idx)*0.8)]
test_idx = shuffled_idx[int(len(shuffled_idx)*0.8):]
test_normal_prep_eegs, test_normal_imgs, test_normal_files = \
    train_prep_eegs[test_idx], train_imgs[test_idx], train_files[test_idx]
train_prep_eegs, train_imgs, train_files = \
    train_prep_eegs[train_idx], train_imgs[train_idx], train_files[train_idx]

# Load test EEG data
with open('{}_seizure_eegs.pickle'.format(modality), 'rb') as handle:
    test_seizure_eegs = pickle.load(handle)
with open('{}_seizure_files.pickle'.format(modality), 'rb') as handle:
    test_seizure_files = pickle.load(handle)
# filter nans
test_seizure_eegs_cleaned = []
test_seizure_files_cleaned = []
count = 0
for idx in range(len(test_seizure_eegs)):
    if np.any(np.isnan(test_seizure_eegs[idx])):
        count += 1
    else:
        test_seizure_eegs_cleaned.append(test_seizure_eegs[idx])
        test_seizure_files_cleaned.append(test_seizure_files[idx])
print("{} seizure eegs are cleaned".format(count))
test_seizure_prep_eegs, test_seizure_files = \
    apply_sliding_window(test_seizure_files_cleaned, test_seizure_eegs_cleaned)
test_seizure_imgs = np.array([img.flatten() for img in test_seizure_prep_eegs])
print("Number of test normal, seizure signals:", len(test_normal_imgs), len(test_seizure_imgs))  # 93301 5511

# Dimension reduction
X = np.concatenate([test_normal_imgs, test_seizure_imgs], 0)
print("Latent space matrix original shape:", X.shape)
cluster = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='ward')
cluster.fit_predict(X)
pred_labels = cluster.labels_
test_labels = np.array([0] * len(test_normal_imgs) + [1] * len(test_seizure_imgs))
# Choose classification threshold
auc = roc_auc_score(test_labels, pred_labels)
fpr, tpr, thresholds = roc_curve(test_labels, pred_labels)
gmeans = np.sqrt(tpr * (1 - fpr))
ix = np.argmax(gmeans)
print('Normal vs. Seizure classification threshold=%f' % (thresholds[ix]))
pred_labels_thresholded = np.array(pred_labels > thresholds[ix])
report_dict = classification_report(test_labels, pred_labels_thresholded, output_dict=True)
precision = (report_dict["macro avg"]["precision"])
recall = (report_dict["macro avg"]["recall"])
accuracy = (report_dict["accuracy"])
print("Normal vs. Seizure precision, recall, accuracy, AUC", precision, recall, accuracy, auc)