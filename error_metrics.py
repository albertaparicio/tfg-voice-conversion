#!/usr/bin/python3

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score


def RMSE(groundtruth, prediction, mask=None):
    """
    groundtruth: matrix containing the real samples to be predicted (N samples,
                 sample dim) 
    prediction: matrix containing the prediction (N samples,
                sample dim) 
    mask: optional binary mask to not consider certain samples
          (0 in mask)
    """
    groundtruth = np.array(groundtruth, dtype=np.float32)
    prediction = np.array(prediction, dtype=np.float32)
    assert groundtruth.shape == prediction.shape
    if mask is not None:
        mask = np.array(mask)
        groundtruth = groundtruth[mask == 1]
        prediction = prediction[mask == 1]
    D = (groundtruth-prediction)**2
    D = np.mean(D, axis=0)
    return np.sqrt(D)


def AFPR(groundtruth, prediction):
    """
    Evaluate Accuracy, F-measure, Precision and Recall for binary inputs
    """
    groundtruth = np.array(groundtruth)
    prediction = np.array(prediction)
    assert groundtruth.shape == prediction.shape
    # A: accuracy
    I = np.mean(groundtruth == prediction)
    F = f1_score(groundtruth, prediction)
    P = precision_score(groundtruth, prediction)
    R = recall_score(groundtruth, prediction)
    return I, F, P, R


def MCD(gt_cep, pr_cep):
    """
    Mel Cepstral Distortion
    Input are matrices with shape (time, cc_dim)
    """
    MCD = 0
    for t in xrange(gt_cep.shape[0]):
        acum = 0
        for n in xrange(gt_cep.shape[1]):
            acum += (gt_cep[t, n]-pr_cep[t, n])**2
        MCD += np.sqrt(acum)
    # scale factor
    alpha = ((10.*np.sqrt(2))/(gt_cep.shape[0]*np.log(10)))
    MCD = alpha*MCD
    return MCD
