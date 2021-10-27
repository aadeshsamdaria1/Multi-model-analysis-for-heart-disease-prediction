#!/usr/bin/env python
# coding: utf-8

# Load libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.metrics import r2_score

df = pd.read_csv('C:/Users/Aadesh/Downloads/heart.csv', header=0)
label_encoder =LabelEncoder()
df['ChestPainType']= label_encoder.fit_transform(df['ChestPainType'])
df['Sex']= label_encoder.fit_transform(df['Sex'])
df['RestingECG']= label_encoder.fit_transform(df['RestingECG'])
df['ExerciseAngina']= label_encoder.fit_transform(df['ExerciseAngina'])
df['ST_Slope']= label_encoder.fit_transform(df['ST_Slope'])

#split dataset in features and target variable
X = df[["Age","Sex", "ChestPainType", "RestingBP", "Cholesterol", "FastingBS", "RestingECG", "MaxHR", "ExerciseAngina", "Oldpeak", "ST_Slope"]].values # Features
y = df[["HeartDisease"]].values # Target variable

#splitting dataset to train and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test
y_train = y_train.reshape(-1)

# importing all classifiers
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier

classifier_names = ["Gaussian Naive Bayes Classifer", "Decision Tree Classifier", "Logistic Regression", "Support Vector Machine", "KNN Classifier" ]
classifiers = [GaussianNB(), DecisionTreeClassifier(criterion='gini', max_depth=4, random_state=1), LogisticRegression(random_state=0, max_iter=140), svm.SVC(kernel='linear', C=1.0), KNeighborsClassifier(n_neighbors = 8) ]
for i in range(0,len(classifiers)):
    clf = classifiers[i]
    # Train Classifer
    clf = clf.fit(X_train,y_train)

    #Predict the response for test dataset
    y_pred = clf.predict(X_test)
    print("\n" + classifier_names[i] + ":\n")
    print("Classification report: ")
    print(classification_report(y_test, y_pred))
    print("R2 value: " + str(r2_score(y_test, y_pred)))
    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred)
    auc = metrics.roc_auc_score(y_test, y_pred)
    plt.plot(fpr,tpr,label= classifier_names[i] + ", auc="+str(auc))

# plot all model results
plt.legend(loc='best', bbox_to_anchor=(1.0, 0.5))
plt.show()
