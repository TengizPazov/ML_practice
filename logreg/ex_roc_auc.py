from roc_auc import build_roc_auc_curve
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np
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
plt.plot(x, y)
plt.show()