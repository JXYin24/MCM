#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import csv
import logging
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_fscore_support

def dataLoading(path, byte_num):
    x = []
    labels = []

    with (open(path, 'r')) as data_from:
        csv_reader = csv.reader(data_from)
        for i in csv_reader:
            x.append(i[0:byte_num])
            labels.append(i[byte_num])
    for i in range(len(x)):
        for j in range(byte_num):
            x[i][j] = float(x[i][j])
    for i in range(len(labels)):
        labels[i] = float(labels[i])
    x = np.array(x)
    labels = np.array(labels)
    return x, labels


def aucPerformance(score, labels):
    roc_auc = roc_auc_score(labels, score)
    ap = average_precision_score(labels, score)
    return roc_auc, ap

def F1Performance(score, target):
    normal_ratio = (target == 0).sum() / len(target)
    score = np.squeeze(score)
    threshold = np.percentile(score, 100 * normal_ratio)
    pred = np.zeros(len(score))
    pred[score > threshold] = 1
    precision, recall, f1, _ = precision_recall_fscore_support(target, pred, average='binary')
    return f1

def writeResults(name, avg_AUC_ROC, avg_AUC_PR,
                 std_AUC_ROC, std_AUC_PR, path):
    csv_file = open(path, 'a')
    row = name + "," + avg_AUC_ROC + ',' + avg_AUC_PR + ',' + std_AUC_ROC + ',' + std_AUC_PR + "\n"
    csv_file.write(row)


def writeWeights(filename, test_label, test_fea_att_w, test_grad_att_w, print_score):
    fea_name = './att_weights/' + filename + '_fea.csv'
    grad_name = './att_weights/' + filename + '_grad.csv'
    with open(fea_name, "w", encoding="UTF-8", newline="") as csvfile1:
        writer1 = csv.writer(csvfile1)
        fea_list = test_fea_att_w.detach().cpu().numpy()[:300]
        fea_list = np.column_stack((test_label[:300], print_score[:300], fea_list))
        writer1.writerows(fea_list.tolist())
    with open(grad_name, "w", encoding="UTF-8", newline="") as csvfile2:
        writer2 = csv.writer(csvfile2)
        grad_list = test_grad_att_w.detach().cpu().numpy()[:300]
        grad_list = np.column_stack((test_label[:300], print_score[:300], grad_list))
        writer2.writerows(grad_list.tolist())


def find_best_lambda(score1, score2, y_test):
    s1= StandardScaler()
    s2 = StandardScaler()
    score1 = s1.fit_transform(score1)
    score2 = s2.fit_transform(score2)
    lambda_list = np.append(np.arange(0, 1, 0.1),np.arange(1,10,1))
    best_auc = 0
    best_pr = 0
    best_lambda = 0
    for th in lambda_list:
        auc, pr = aucPerformance(score1 + th * score2, y_test)
        if auc + pr > best_auc + best_pr:
            best_auc = auc
            best_pr = pr
            best_lambda = th
    return best_auc, best_pr, best_lambda


def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "%(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])
    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    return logger

