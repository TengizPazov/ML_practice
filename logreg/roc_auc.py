import numpy as np
import matplotlib.pyplot as plt
def build_roc_auc_curve(true_labels: np.ndarray, predicted_probas: np.ndarray) -> np.ndarray:
    if sum(true_labels) == 0 or sum(true_labels) == len(true_labels):
        raise ValueError()
    treshholds = sorted(predicted_probas)[::-1]
    result = []
    result.append((0.0, 0.0))
    #we assume that there are no treshholds with tresh == 1 and tresh == 0
    for k in range(len(treshholds)):
        predictions = np.zeros(len(true_labels))
        for i in range(len(predicted_probas)):
            if predicted_probas[i] >= treshholds[k]:
                predictions[i] = 1
            else:
                predictions[i] = 0
        tp = 0
        fp = 0
        fn = 0
        tn = 0
        for i in range(len(true_labels)):
                if true_labels[i] == 1 and predictions[i] == 1:
                     tp += 1
                elif true_labels[i] == 1 and predictions[i] == 0:
                     fn += 1
                elif true_labels[i] == 0 and predictions[i] == 1:
                     fp += 1
                elif true_labels[i] == 0 and predictions[i] == 0:
                     tn += 1
        TPR = tp / (tp + fn)
        FPR = fp / (fp + tn)
        result.append((FPR, TPR))
    result.append((1.0, 1.0))
    return result
