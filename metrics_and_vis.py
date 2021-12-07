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


metric_names = ["Precision", "Recall", "Accuracy", "AUC"] * 4
metrics = [0.5288325179818278, 0.6356408684121599, 0.6693805124993476, 0.6719006584658864,
           0.5144513549889942, 0.5750354614299815, 0.5561296383278534, 0.5928878905250963,
           0.5583438249228568, 0.7049228989937274, 0.7972444026929701, 0.7501257370037241,
           0.5302545962793698, 0.6398009743291088, 0.6810187359741141, 0.6770192700874208]
plt.figure()
sns.boxplot(x=metric_names, y=metrics)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.savefig("results_mit/mit_metrics_plot.pdf")
print("Average metrics over MIT:")
metrics = np.array([[0.5288325179818278, 0.6356408684121599, 0.6693805124993476, 0.6719006584658864, 1e-21],
            [0.5144513549889942, 0.5750354614299815, 0.5561296383278534, 0.5928878905250963, 1e-18],
            [0.5583438249228568, 0.7049228989937274, 0.7972444026929701, 0.7501257370037241, 1e-61],
             [0.5302545962793698, 0.6398009743291088, 0.6810187359741141, 0.6770192700874208, 1e-21]])
metrics_mean = np.ceil(np.mean(metrics, 0) * 100).astype(int) / 100
metrics_std = np.ceil(np.std(metrics, 0) * 100).astype(int) / 100
for idx in range(5):
    print("$" + str(metrics_mean[idx]) + " \pm " + str(metrics_std[idx]) + "$")