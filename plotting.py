import numpy as np
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt

numbers_string = ['is_sarcasm', 'not_sarcasm']


def plot_confusion_matrix(confusion_matrix, title, save_name):
    df_cm = pd.DataFrame(confusion_matrix, index=numbers_string, columns=numbers_string)
    ax = plt.axes()
    cm_heatmap = sn.heatmap(df_cm, ax=ax, annot=True, cmap='Blues', fmt='g')
    ax.set_title(title)
    figure = cm_heatmap.get_figure()
    figure.savefig(save_name, dpi=400)
    plt.show()


confusion_matrix = np.zeros((2, 2))

confusion_matrix[0][0] = 1950
confusion_matrix[0][1] = 307
confusion_matrix[1][0] = 316
confusion_matrix[1][1] = 1434

plot_confusion_matrix(confusion_matrix, 'Baseline - SVM ', 'CNN_0.15.png')