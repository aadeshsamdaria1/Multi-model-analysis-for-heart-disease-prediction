# Multi model heart disease prediction

## Description: 
  An analysis and comparison between various supervised machine learning models was performed on an extensive data set of heart-disease patients. SVM, KNN, Gaussian NB classifier,  Decision tree classifier and Logistic regression are the models that were used in the analysis. 
  
## Model Specifications:
<p>Decision tree classifier :- Max depth = 4 , random state = 1 </p>
<p>Logistic regression :- Max iterations = 140, random state = 0 </p>
<p>SVM :- kernel = linear </p>
<p>KNN classifier :-  number of neighbors = 8 </p>
<p>Gaussian NB :- default </p>

## Dataset:  
Dataset source: https://www.kaggle.com/datasets. 
11 predictor variables like RestingBP, Cholesterol, MaxHR etc were used to predict if an individual is prone to a heart disease.
  
## Output :  
  The SVM and the Decision tree classifier have the same high accuracy and the KNN classifier has the least accuracy.  
  A greater area under the curve would imply that the Decision tree model has lower chance of prediciting false positives.
  
  <img src="output_graph.png">




