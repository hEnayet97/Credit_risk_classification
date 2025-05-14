# Credit_risk_classification
## Overview of the Analysis

The purpse of this analysis is to use supervised machine learning to create a model to predict the credibility of the borrowers and if the loan is healthy (0) or high risk(1) based on different financial parameters. The financial information included in this dataset are: loan size, interest rate, borrower income, debt-to-income ratio, number of accounts, derogatory marks, and total debt. We are trying to predict the loan status which is binary, i.e if it is a healthy loan (0) and will be returned by borrower, or if it is a high risk loan (1) and there is risk of defaulting. The lending dataset had 77,536 data values with 75,036 being healthy loans and 2,500 being high risk loans.

The stages of machine learning  process used in this analysis are:
* The data was pre-processed to create the labels and features.The 'loan_status' column was set as y, and then features (X) DataFrame was created from the remaining columns.
* The data was split into training and testing datasets by using train_test_split.
* A Logestic Regression Model was created from the original dataset as our data in continuous and this method can be used in predicting binary outcomes from data
  * The logistic regression model had max_iter=200, random_state=1 and was fit by using the training data (X_train and y_train).
* Predictions were made using the testing feature data (X_test) and the fitted model.
* The model was evaluated using a confusion matrix and classification report (precision, recall, accuracy) scores.
  
* For comparison, I also created a Random Forest Model from the original dataset as the lending dataset had class imbalance.
  * Similar steps are used for this machine learning model except after splitting the data StandardScaler was used to scale the features data (only X_train and  X_testing).
  * Once the data was scaled, a random forest instance was created using n_estimators=500, random_state=78. The model was fit on training data and predictions were made, same as the other model.

## Results

* Logestic Regression Model :
    * Healthly Loan (0) - precision = 1.00, recall = 1.00
    * High Risk Loan (1)- precision = 0.87, recall = 0.95
    * Overall model accuracy is 99%
      
* Random Forest Classification Model:
    * Healthly Loan (0) - precision = 1.00, recall = 1.00
    * High Risk Loan (1)- precision = 0.85, recall = 0.88
    * Overall model accuracy is 99%

## Summary

Although, both models have an overall accuracy of 99%, the logestic regression model is better than the random forest for credit risk analysis.
The logistic regression model does a good job in predicting Healthy Loans with 100% precision and High-Risk Loans with 87% precision. It has a low false negative as well and most high risk loans are actually caught by the model (95% recall).

However, there is a class imbalance of 18759 healthy loans, and only 625 high risk ones used in the model. So, the high accuracy may be misleading as there are few instances of the high risk loans and the model can misclassifying healthy loans as high risk ones. This can be improved by adding more of the high risk loans in the training or resampling to improve the precision of high risk loans. 
It is more important to predict the high risk loans as it can lead to finanicial loss if the borrower defaults if not prevented on time.

## References

The assignment was completed with the help of inclass resources. A random forest classification was done for model comparison
