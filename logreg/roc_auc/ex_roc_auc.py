from roc_auc import build_roc_auc_curve
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np
def calculate_auc(x, y):
    """Вычисление AUC методом трапеций"""
    auc = 0.0
    for i in range(1, len(x)):
        width = x[i] - x[i-1]
        height = (y[i] + y[i-1]) / 2
        auc += width * height
    return auc
#загрузка данных
url = "https://raw.githubusercontent.com/Statology/Python-Guides/main/default.csv"
data = pd.read_csv(url)
X = data[['student', 'balance', 'income']]
y = data['default']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

log_regression = LogisticRegression()
log_regression.fit(X_train, y_train)

y_pred_proba = log_regression.predict_proba(X_test)[:,1]
y_pred_proba = np.array(y_pred_proba)
y_test = np.array(y_test)
result = build_roc_auc_curve(y_test, y_pred_proba)
x = []
y = []
for value in result:
    x.append(value[0])
    y.append(value[1])
AUC = calculate_auc(x, y)
print(AUC)
plt.plot(x, y, label=f'ROC curve (AUC = {round(AUC, 3)})')
x0 = [0.0, 1.0]
y0 = [0.0, 1.0]
plt.plot(x0, y0, linestyle='--', label='random model')
plt.legend()
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('ROC-AUC curve')
plt.savefig('ROC-AUC.png')