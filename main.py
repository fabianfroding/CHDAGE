import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn import linear_model
import matplotlib.pyplot as plt

# Show original data
df = pd.read_csv("data/CHD.csv", header=0)
plt.figure()
plt.axis([0, 70, -0.2, 1.2])
plt.title('Original data')
plt.scatter(df['age'], df['chd'])
plt.show()

# Create logistic regression model
logistic = linear_model.LogisticRegression(C=1e5)
logistic.fit(df['age'].values.reshape(100, 1), df['chd'].values.reshape(100, 1))

# LogisticRegression(C = 100000.0, class_weight = None, dual = False, fit_intercept = True, intercept_scaling = 1, max_iter = 100, multi_class = 'ovr', n_jobs = 1, penalty = '12', random_state = None, solver = 'liblinear', tol = 0.0001, verbose = 0, warm_start = False)

print(df.shape)

x_plot = np.linspace(10, 90, 100)
oneprob = []
zeroprob = []
predict = []
plt.figure(figsize=(10, 10))
for i in x_plot:
    oneprob.append(logistic.predict_proba(i)[0][1])
    zeroprob.append(logistic.predict_proba(i)[0][0])
    predict.append(logistic.predict(i)[0])

plt.plot(x_plot, oneprob)
plt.plot(x_plot, zeroprob)
plt.plot(x_plot, predict)
plt.scatter(df['age'], df['chd'])
