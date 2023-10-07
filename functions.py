import numpy as np
from sklearn.model_selection import KFold
import math
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, average_precision_score, f1_score,classification_report,confusion_matrix
from scipy.optimize import brentq
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

def evaluate(labels, scores,res_th=None):
    '''
    metric for auc/ap
    :param labels:
    :param scores:
    :param res_th:
    :param saveto:
    :return:
    '''
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    fpr, tpr, ths = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)

    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)

    best_f1 = 0
    best_threshold = 0
    for threshold in ths:
        tmp_scores = scores.copy()
        tmp_scores[tmp_scores >= threshold] = 1
        tmp_scores[tmp_scores < threshold] = 0
        cur_f1 = f1_score(labels, tmp_scores)
        if cur_f1 > best_f1:
            best_f1 = cur_f1
            best_threshold = threshold

    #threshold f1
    auc_prc=average_precision_score(labels,scores)
    return auc_prc,roc_auc,best_threshold,best_f1

def normalize(seq):
    '''
    normalize to [-1,1]
    :param seq:
    :return:
    '''
    return 2*(seq-np.min(seq))/(np.max(seq)-np.min(seq))-1

def getFloderK(data,folder,label):
    normal_cnt = data.shape[0]
    folder_num = int(normal_cnt / 5)
    folder_idx = folder * folder_num

    folder_data = data[folder_idx:folder_idx + folder_num]

    remain_data = np.concatenate([data[:folder_idx], data[folder_idx + folder_num:]])
    if label==0:
        folder_data_y = np.zeros((folder_data.shape[0], 1))
        remain_data_y=np.zeros((remain_data.shape[0], 1))
    elif label==1:
        folder_data_y = np.ones((folder_data.shape[0], 1))
        remain_data_y = np.ones((remain_data.shape[0], 1))
    else:
        raise Exception("label should be 0 or 1, get:{}".format(label))
    return folder_data,folder_data_y,remain_data,remain_data_y

def getPercent(data_x,data_y,percent,seed):
    train_x, test_x, train_y, test_y = train_test_split(data_x, data_y,test_size=percent,random_state=seed)
    return train_x, test_x, train_y, test_y
