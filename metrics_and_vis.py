import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

metric_names = ["Precision", "Recall", "Accuracy", "AUC"] * 5
metrics = [0.734, 0.756, 0.759, 0.8,
            0.731, 0.745, 0.762, 0.787,
            0.707, 0.714, 0.742, 0.735,
            0.849, 0.858, 0.869, 0.905,
            0.8, 0.84, 0.83, 0.9]
plt.figure()
sns.boxplot(x=metric_names, y=metrics)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.savefig("results_upenn_extended/upenn_extended_metrics_plot.pdf")
print("Average metrics over UPenn:")
metrics = np.array([[0.734, 0.756, 0.759, 0.8, 1e-162],
            [0.731, 0.745, 0.762, 0.787, 1e-162],
            [0.707, 0.714, 0.742, 0.735, 1e-101],
            [0.849, 0.858, 0.869, 0.905, 1e-303],
            [0.8, 0.84, 0.83, 0.9, 1e-303]])
metrics_mean = np.ceil(np.mean(metrics, 0) * 100).astype(int) / 100
metrics_std = np.ceil(np.std(metrics, 0) * 100).astype(int) / 100
for idx in range(5):
    print("$" + str(metrics_mean[idx]) + " \pm " + str(metrics_std[idx]) + "$")


metric_names = ["Precision", "Recall", "Accuracy", "AUC"] * 4
metrics = [0.714, 0.796, 0.783, 0.873,
           0.609, 0.642, 0.705, 0.642,
           0.575, 0.606, 0.655, 0.602,
           0.767, 0.703, 0.841, 0.537]
plt.figure()
sns.boxplot(x=metric_names, y=metrics)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.savefig("results_tuh/tuh_metrics_plot.pdf")
print("Average metrics over TUH:")
metrics = np.array([[0.714, 0.796, 0.783, 0.873, 1e-303],
            [0.609, 0.642, 0.705, 0.642, 1e-83],
            [0.575, 0.606, 0.655, 0.602, 1e-106],
            [0.767, 0.703, 0.841, 0.537, 1e-65]])
metrics_mean = np.ceil(np.mean(metrics, 0) * 100).astype(int) / 100
metrics_std = np.ceil(np.std(metrics, 0) * 100).astype(int) / 100
for idx in range(5):
    print("$" + str(metrics_mean[idx]) + " \pm " + str(metrics_std[idx]) + "$")


metric_names = ["Precision", "Recall", "Accuracy", "AUC"] * 3
metrics = [0.5118726499682063, 0.5621296578755188, 0.5250769792808309, 0.5832242014887123,
           0.5299723597382866, 0.6372739943693104, 0.6855592088095611, 0.6756119971190334,
           0.5146735484814321, 0.5765042476681734, 0.5449089295965763, 0.5945870947442446]
plt.figure()
sns.boxplot(x=metric_names, y=metrics)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.savefig("results_mit/mit_metrics_plot.pdf")
print("Average metrics over MIT:")
metrics = np.array([[0.5118726499682063, 0.5621296578755188, 0.5250769792808309, 0.5832242014887123, 1e-18],
            [0.5299723597382866, 0.6372739943693104, 0.6855592088095611, 0.6756119971190334, 1e-61],
            [0.5146735484814321, 0.5765042476681734, 0.5449089295965763, 0.5945870947442446, 1e-21]])
metrics_mean = np.ceil(np.mean(metrics, 0) * 100).astype(int) / 100
metrics_std = np.ceil(np.std(metrics, 0) * 100).astype(int) / 100
for idx in range(5):
    print("$" + str(metrics_mean[idx]) + " \pm " + str(metrics_std[idx]) + "$")