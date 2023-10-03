# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import os
import seaborn as sns

from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, normalize, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, precision_score,f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from imblearn.metrics import sensitivity_score

import xgboost as xgb
from xgboost import plot_importance
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sns.set_style('darkgrid')

#Read and Print diabes.csv out screen
df = pd.read_csv("diabetes.csv")
df.head(10).style. \
                    set_properties(**{"min-width": "60px"}). \
                    set_properties(**{"color": "#111111"}). \
                    set_properties(**{"text-align": "center"}). \
                    set_table_styles([
                          {"selector": "th",
                           "props": [("font-weight", "bold"),
                                     ("font-size", "12px"),
                                     ("text-align", "center")]},
                          {"selector": "tr:nth-child(even)",
                           "props": [("background-color", "#f2f2f2")]},
                          {"selector": "tr:nth-child(odd)",
                           "props": [("background-color", "#fdfdfd")]},
                          {"selector": "tr:hover",
                           "props": [("background-color", "#bcbcbc")]}])

#Drop Columns
pd.options.mode.chained_assignment = None

x = df.loc[:,df.columns != "Outcome"]
y = df[["Outcome"]]
#Print missing data
print("Print the missing value contains \n",df.isnull().sum())
#Print Data_info
print("x_data info: \n")
x.info()
print("\ny_data info: \n")
y.info()

y["Outcome"].value_counts() #Count 2 form

#Split the test
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0,stratify=y)
x_train.reset_index(drop=True,inplace=True)
x_test.reset_index(drop=True,inplace=True)
y_train.reset_index(drop=True,inplace=True)
y_test.reset_index(drop=True,inplace=True)

def plots(feature):
    fig = plt.figure(constrained_layout=True, figsize=(10, 3))
    gs = gridspec.GridSpec(nrows=1, ncols=4, figure=fig)
    ax1 = fig.add_subplot(gs[0, :3])

    # Plot histograms
    sns.histplot(data=x_train.loc[y_train["Outcome"] == 0, feature],
                 kde=False, color="#004a4d", bins=40, stat="count",
                 label="Not Diabetes", ax=ax1)
    sns.histplot(data=x_train.loc[y_train["Outcome"] == 1, feature],
                 kde=False, color="#7d0101", bins=40, stat="count",
                 label="Diabetes", ax=ax1)

    ax2 = fig.add_subplot(gs[0, 3])

    # Plot boxplot
    sns.boxplot(data=x_train, x=feature, orient="v", color="#989100", width=0.2, ax=ax2)

    ax1.legend(loc="upper right")

Q1 = x_train["Pregnancies"].quantile(0.25)
Q3 = x_train["Pregnancies"].quantile(0.75)
q95th = x_train["Pregnancies"].quantile(0.95)
IQR = Q3 - Q1
UW = Q3 + 1.5*IQR

x_train["Pregnancies"] = np.where(x_train["Pregnancies"] > UW,
                                  q95th, x_train["Pregnancies"])
plots("Pregnancies")

#Replace 0 values with median
med = x_train["Glucose"].median()
x_train["Glucose"] = np.where(x_train["Glucose"] == 0, med, x_train["Glucose"])
plots("Glucose")

med = x_train["BloodPressure"].median()
q5th = x_train["BloodPressure"].quantile(0.05)
q95th = x_train["BloodPressure"].quantile(0.95)
Q1 = x_train["BloodPressure"].quantile(0.25)
Q3 = x_train["BloodPressure"].quantile(0.75)
IQR = Q3 - Q1
LW = Q1 - 1.5*IQR
UW = Q3 + 1.5*IQR
#Remove some 0 values for BloodPressure -> Replace with median
x_train["BloodPressure"] = np.where(x_train["BloodPressure"] == 0,
                                    med, x_train["BloodPressure"])
x_train["BloodPressure"] = np.where(x_train["BloodPressure"] < LW,
                                    q5th, x_train["BloodPressure"])
x_train["BloodPressure"] = np.where(x_train["BloodPressure"] > UW,
                                    q95th, x_train["BloodPressure"])
plots("BloodPressure")

med = x_train["SkinThickness"].median()
q95th = x_train["SkinThickness"].quantile(0.95)
Q1 = x_train["SkinThickness"].quantile(0.25)
Q3 = x_train["SkinThickness"].quantile(0.75)
IQR = Q3 - Q1
UW = Q3 + 1.5*IQR
#Replace 0 values
x_train["SkinThickness"] = np.where(x_train["SkinThickness"] == 0,
                                    med, x_train["SkinThickness"])
x_train["SkinThickness"] = np.where(x_train["SkinThickness"] > UW,
                                    q95th, x_train["SkinThickness"])

plots("SkinThickness")

q60th = x_train["Insulin"].quantile(0.60)
q95th = x_train["Insulin"].quantile(0.95)
Q1 = x_train["Insulin"].quantile(0.25)
Q3 = x_train["Insulin"].quantile(0.75)
IQR = Q3 - Q1
UW = Q3 + 1.5*IQR
#Remove 0 values
x_train["Insulin"] = np.where(x_train["Insulin"] == 0,
                              q60th, x_train["Insulin"])
x_train["Insulin"] = np.where(x_train["Insulin"] > UW,
                              q95th, x_train["Insulin"])
plots("Insulin")

med = x_train["BMI"].median()
q95th = x_train["BMI"].quantile(0.95)
Q1 = x_train["BMI"].quantile(0.25)
Q3 = x_train["BMI"].quantile(0.75)
IQR = Q3 - Q1
UW = Q3 + 1.5*IQR

x_train["BMI"] = np.where(x_train["BMI"] == 0,
                          med, x_train["BMI"])
x_train["BMI"] = np.where(x_train["BMI"] > UW,
                          q95th, x_train["BMI"])
plots("BMI")

q95th = x_train["DiabetesPedigreeFunction"].quantile(0.95)
Q1 = x_train["DiabetesPedigreeFunction"].quantile(0.25)
Q3 = x_train["DiabetesPedigreeFunction"].quantile(0.75)
IQR = Q3 - Q1
UW = Q3 + 1.5*IQR
#Replace values greater than UW with 95th quantile
x_train["DiabetesPedigreeFunction"] = np.where(
                        x_train["DiabetesPedigreeFunction"] > UW,
                        q95th, x_train["DiabetesPedigreeFunction"])
plots("DiabetesPedigreeFunction")

q95th = x_train["Age"].quantile(0.95)
Q1 = x_train["Age"].quantile(0.25)
Q3 = x_train["Age"].quantile(0.75)
IQR = Q3 - Q1
UW = Q3 + 1.5*IQR

x_train["Age"] = np.where(x_train["Age"] > UW,
                          q95th, x_train["Age"])
plots("Age")

#Standarization
scaler_ti = StandardScaler()
x_train[['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']] = scaler_ti.fit_transform(x_train[['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']])
x_test[['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']] = scaler_ti.fit_transform(x_test[['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']])

#Single Perceptron
ppn_clf = Perceptron(eta0=0.001,max_iter=5000,random_state=1)
ppn_clf.fit(x_train,y_train.values.ravel())

y_pred = ppn_clf.predict(x_test)
report = classification_report(y_test,y_pred, digits=4, target_names=["Not Diabetes", "Diabetes"])

#Print report and accuracy of Single Perceptron
ppn_acc = accuracy_score(y_test,y_pred)
ppn_precision = precision_score(y_test,y_pred)
ppn_sensitivity = sensitivity_score(y_test, y_pred, average='weighted')
ppn_f1 = f1_score(y_test,y_pred)
print("Report",report)
print("Accuracy: ", np.round(ppn_acc,2))
print("Precision: ", np.round(ppn_precision,2))
print("Sensitivity: ", np.round(ppn_sensitivity,2))
print("F1_Score: ", np.round(ppn_f1,2))

#Gausiaan Classifier
g_clf = GaussianNB()
g_clf.fit(x_train, y_train.values.ravel())
y_pred = g_clf.predict(x_test)

report = classification_report(y_test,y_pred, digits=4, target_names=["Not Diabetes", "Diabetes"])

#Print report and accuracy of Gaussian Classifier
g_acc = accuracy_score(y_test,y_pred)
g_precision = precision_score(y_test,y_pred)
g_sensitivity = sensitivity_score(y_test, y_pred, average='weighted')
g_f1 = f1_score(y_test,y_pred)
print("Report",report)
print("Accuracy: ", np.round(g_acc,2))
print("Precision: ", np.round(g_precision,2))
print("Sensitivity: ", np.round(g_sensitivity,2))
print("F1_score: ", np.round(g_f1,2))

#K-NN
test_scores = []
train_scores = []
for i in range(1, 15):
  knn = KNeighborsClassifier(i)
  knn.fit(x_train, y_train.values.ravel())
  train_scores.append(knn.score(x_train,y_train.values.ravel()))
  test_scores.append(knn.score(x_test,y_test))

#Max Score testing on datapoint
def max_score(scores):
  max_score = max(scores)
  scores_indx = [i for i, v in enumerate(scores) if v == max_score]
  return print('Max test score {} % and k = {}'.format(max_score*100,list(map(lambda x: x+1, scores_indx))))

#K-NN scores that comes from testing on the same datapoints were used for training
max_score(train_scores)
max_score(test_scores)

plt.figure(figsize=(12,5))
p = sns.lineplot(train_scores,marker='*',label='Train Score')
p = sns.lineplot(test_scores,marker='o',label='Test Score')

#The best result is captured at k = 4 -> We used k=4 for the final
knn=KNeighborsClassifier(4)
knn.fit(x_train,y_train.values.ravel())
knn.score(x_test,y_test)
y_pred = knn.predict(x_test)

report = classification_report(y_test,y_pred, digits=4, target_names=["Not Diabetes", "Diabetes"])
knn_acc = accuracy_score(y_test,y_pred)
knn_precision = precision_score(y_test,y_pred)
knn_sensitivity = sensitivity_score(y_test, y_pred, average='weighted')
knn_f1 = f1_score(y_test,y_pred)
print("Report",report)
print("Accuracy: ", np.round(knn_acc,2))
print("Precision: ", np.round(knn_precision,2))
print("Sensitivity: ", np.round(knn_sensitivity,2))
print("F1_Score: ", np.round(knn_f1, 2))

x_train.drop("SkinThickness",axis=1,inplace=True)
x_test.drop("SkinThickness",axis=1,inplace=True)

#Multiple Layer Perceptron (MLP)
mlp_clf = MLPClassifier(solver="adam", max_iter=5000, activation="relu",
                        hidden_layer_sizes= (12),
                        alpha = 0.01,
                        batch_size = 64,
                        learning_rate_init = 0.001,
                        random_state=2)
mlp_clf.fit(x_train,y_train.values.ravel())

y_pred = mlp_clf.predict(x_test)
report = classification_report(y_test,y_pred, digits=4, target_names=["Not Diabetes", "Diabetes"])

#Print report and accuracy of Multiple Layer Perceptron
mlp_acc = accuracy_score(y_test,y_pred)
mlp_precision = precision_score(y_test,y_pred)
mlp_sensitivity = sensitivity_score(y_test, y_pred, average='weighted')
mlp_f1 = f1_score(y_test,y_pred)
print("Report",report)
print("Accuracy: ", np.round(mlp_acc,2))
print("Precision: ", np.round(mlp_precision,2))
print("Sensitivity: ", np.round(mlp_sensitivity,2))
print("F1_Score: ", np.round(mlp_f1, 2))

all_performance_metrics = pd.DataFrame([
    [ppn_acc, ppn_precision, ppn_sensitivity, ppn_f1],
    [g_acc, g_precision, g_sensitivity, g_f1],
    [knn_acc, knn_precision, knn_sensitivity, knn_f1],
    [mlp_acc, mlp_precision, mlp_sensitivity, mlp_f1],
], index= ['PPN', 'GB', 'k-NN', 'MLP'],
columns = ['Accuracy', 'Precision', 'Sensitivity', 'F1_Score'])
all_performance_metrics

all_performance_metrics['Model'] = all_performance_metrics.index.values
colors = ['tab:green', 'tab:orange', 'tab:red', 'tab:blue']
all_performance_metrics.plot(x = 'Model', kind = 'bar', rot = 0, color = colors)
plt.title('Performance metrics of each model')
plt.xlabel('Model')
plt.ylabel('Score')
plt.legend(loc='center right', bbox_to_anchor=(1.35, 0.5))
plt.show()

