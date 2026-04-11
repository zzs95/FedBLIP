import numpy as np
from sklearn.metrics import f1_score as f1_score_func
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, precision_recall_curve, average_precision_score
import copy
def calculate_metrics(y_true, y_pred_proba, threshold=None, average='macro'):
    # 'micro', 'macro', 'samples', 'weighted'
    y_true[0] = 1
    y_pred_proba[0] = 1
    auc0 = roc_auc_score(y_true, y_pred_proba, average=average)
    # Calculate mean Average Precision (mAP)
    mAP = average_precision_score(y_true, y_pred_proba, average=average)
    if threshold==None:
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
        youden_index = tpr - fpr
        threshold = thresholds[np.argmax(youden_index)]
    
    # 根据设定的阈值计算预测类别
    y_pred = (y_pred_proba > threshold).astype(int)
    # 计算 AUC
    # auc_ = roc_auc_score(y_true, y_pred)
    acc = np.sum(y_pred == y_true) / len(y_true)
    
    # 计算混淆矩阵
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    # 计算敏感度（sensitivity）、特异度（specificity）、召回率（recall）
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    recall = sensitivity
    
    # 计算 F1 分数
    precision = tp / (tp + fp)
    # f1_score = 2 * (precision * recall) / (precision + recall)
    f1_score = f1_score_func(y_true, y_pred, average=average)
    
    # Calculate the Precision-Recall curve for mAP
    # precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_pred_proba)

    return auc0, acc, sensitivity, specificity, precision, f1_score, mAP, threshold, y_pred