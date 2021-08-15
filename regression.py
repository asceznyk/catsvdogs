import pickle

import numpy as np
import pandas as pd

import sklearn

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, accuracy_score

x = np.load(f'/content/trainfeatures.npy')
y = np.load(f'/content/trainlabels.npy')

print(x.shape, y.shape)

x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.001, random_state=1337)
clf = LogisticRegression(max_iter=2000)
clf.fit(x_train, y_train)

p_val = clf.predict_proba(x_val)[:, 1]
loss = log_loss(y_val, p_val)
acc = clf.score(x_val, y_val)

print(f'log loss (validation): {loss}')
print(f'accuaracy score (validation): {acc}')

x_test = np.load('/content/testfeatures.npy')

print(x_test.shape)

p_test = clf.predict_proba(x_test)[:, 1]
df = pd.DataFrame({'id': np.arange(1, 12501), 'label': np.clip(p_test, 0.005, 0.995)})
df.to_csv('yosubmission.csv', index=0)
print(f'saving nsubmission.csv file...')

filename = 'clf.log.regressor'
pickle.dump(clf, open(filename, 'wb'))
print(f'saving classifier to file {filename}...')
