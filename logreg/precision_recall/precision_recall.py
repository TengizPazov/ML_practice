import numpy as np
import matplotlib.pyplot as plt
def build_precision_recall_curve(
    true_labels: np.ndarray, predicted_probas: np.ndarray
) -> np.ndarray:
    if np.sum(true_labels) == 0:
        raise ValueError()
    treshhold = sorted(predicted_probas)[::-1]
    result = []
    result.append((0.0, 1.0))
    for tresh in treshhold:
        predictions = np.zeros(len(true_labels))
        for i in range(len(predicted_probas)):
            if predicted_probas[i] >= tresh:
                predictions[i] = 1
            else:
                predictions[i] = 0
        tp = 0
        fp = 0
        fn = 0 
        for i in range(len(true_labels)):
            if predictions[i] == 1 and true_labels[i] == 1:
                tp += 1
            elif predictions[i] == 1 and true_labels[i] == 0:
                fp += 1
            elif predictions[i] == 0 and true_labels[i] == 1:
                fn += 1
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        result.append((recall, precision))
    return np.array(result)
np.random.seed(42)
true_labels = np.array([0, 0, 1, 1, 1, 1, 1, 0, 1, 0])
predicted_probas = np.array([0.1, 0.2, 0.15, 0.9, 0.43, 0.79, 0.75, 0.3, 0.7, 0.25])
x = []
y = []
result = build_precision_recall_curve(true_labels, predicted_probas)
for value in result:
    x.append(value[0])
    y.append(value[1])
plt.plot(x, y)
plt.show()
