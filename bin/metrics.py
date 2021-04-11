import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import torch

def make_stats(y, y_hat):
    cm = confusion_matrix(y, y_hat)
    cm_df = pd.DataFrame(cm, columns=[str(i) for i in range(10)])
    report = classification_report(y, y_hat)
    return cm_df, report



