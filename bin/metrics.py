import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import torch

def make_stats(y, y_hat):
    cm = confusion_matrix(y, y_hat)
    cm_df = pd.DataFrame(cm, columns=[str(i) for i in range(10)])
    report = classification_report(y, y_hat)
    return cm_df, report



def get_model_acc(model, data_loader):
    total_samples = 0
    total_misclass = 0
    all_y = np.array([], dtype=np.uint8)
    all_y_hat = np.array([], dtype=np.uint8)
    for i, sample in enumerate(data_loader):
        y = sample['y']
        y = y.data.numpy()
        images = (sample['image'])
        images = images.float()
        images = (sample['image'])
        images = images.float()
        model = model.cpu()
        predictions = model(images)
        predictions = predictions.cpu()
        predictions = predictions.data.numpy()
        y_hat = np.argmax(predictions, axis=1)
        misclass = np.sum(np.where(y != y_hat, 1, 0))
        total_samples += y.shape[0]
        total_misclass += misclass
        all_y = np.append(all_y, y)
        all_y_hat = np.append(all_y_hat, y_hat)
        #print(f"all_y = {all_y}")
        #print(f"all_y_hat = {all_y_hat}")
        print(f"Number of Misclassifications = {misclass}")
        print(f"Sample acc = {(y.shape[0]-misclass)/y.shape[0]*100}")
    overall_acc = (total_samples - total_misclass)/total_samples
    print(f"Overall Accuracy = {overall_acc}")
    return all_y, all_y_hat, overall_acc