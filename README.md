# Customer Churn Prediction Project Report

## 1. Introduction

This report details a project focused on analyzing customer churn for a telecom company, "Leo." The project aims to identify factors contributing to customer churn and develop predictive models to prevent it. We utilize Python, Pandas, Matplotlib, Seaborn, and Keras for data manipulation, visualization, and model building.

## 2. Objectives

The primary objectives were:

* To perform data manipulation to extract relevant customer information.
* To visualize customer churn and internet service distribution.
* To build and evaluate sequential models using Keras to predict customer churn.

## 3. Methodology

The project followed these steps:

1.  **Data Loading and Initial Inspection:**
    * Imported necessary Python libraries: Pandas, NumPy, Matplotlib, and Seaborn.
    * Loaded the "customer\_churn\_ai.csv" dataset into a Pandas DataFrame.
    * Displayed the DataFrame to understand the data's structure.

2.  **Data Manipulation:**
    * Calculated the total number of male customers.
    * Calculated the total number of customers with 'DSL' internet service.
    * Extracted female senior citizens with 'Mailed check' payment method.
    * Extracted customers with tenure less than 10 months or total charges less than 500.

3.  **Data Visualization:**
    * Created a pie chart to show the distribution of customer churn.
    * Created a bar plot to show the distribution of 'Internet Service'.

4.  **Model Building:**
    * Built three sequential models using Keras:
        * Model 1: Predict churn using 'tenure' as a feature.
        * Model 2: Model 1 with dropout layers.
        * Model 3: Predict churn using 'tenure', 'Monthly Charges', and 'Total Charges' as features.
    * Used 'Relu' activation, 'Adam' optimizer, and trained for 150 epochs.
    * Predicted values on the test set and built confusion matrices.
    * Plotted 'Accuracy vs Epochs' graphs.

5.  **Data Preprocessing for Model Building:**
    * Converted the Churn column to numerical values.
    * Handled the 'TotalCharges' column, converting it to numeric and filling any errors with 0.
    * Split the data into training and testing sets.
    * Scaled the numerical features.
    * Converted the scaled dataframes to numpy arrays for Keras.

## 4. Dataset Description

The "customer\_churn\_ai.csv" dataset contains the following columns:

* **customerID:** Unique customer identifier.
* **gender:** Customer's gender.
* **SeniorCitizen:** Whether the customer is a senior citizen (1) or not (0).
* **Partner:** Whether the customer has a partner.
* **Dependents:** Whether the customer has dependents.
* **tenure:** Number of months the customer has stayed with the company.
* **PhoneService:** Whether the customer has phone service.
* **MultipleLines:** Whether the customer has multiple lines.
* **InternetService:** Customer's internet service provider.
* **OnlineSecurity:** Whether the customer has online security.
* ... (other service-related columns)
* **Contract:** Customer's contract type.
* **PaperlessBilling:** Whether the customer has paperless billing.
* **PaymentMethod:** Customer's payment method.
* **MonthlyCharges:** Amount charged to the customer monthly.
* **TotalCharges:** Total amount charged to the customer.
* **Churn:** Whether the customer churned.

## 5. Results and Observations

* Data manipulation tasks were completed successfully.
* Data visualizations provided insights into customer churn and internet service distribution.
* Sequential models were built and evaluated, showing varying degrees of accuracy.
* The models using multiple features performed better than the model using only tenure.
* Dropout layers impacted the model performance.
* Accuracy vs Epochs graphs were plotted, showing the model's learning progress.

## 6. Conclusion

This project successfully analyzed customer churn data, performed data manipulation, visualized key features, and built predictive models using Keras. The models provide a foundation for predicting customer churn, which can help the telecom company take proactive measures to retain customers.

## 7. Future Work

Future work on this project could include:

* Performing more extensive feature engineering and selection.
* Trying different machine learning algorithms and deep learning architectures.
* Hyperparameter tuning to improve model performance.
* Analyzing the impact of different customer demographics and services on churn.
* Implementing methods to handle imbalanced datasets.
* Deploying the best model for real-time churn prediction.
* Further exploring the relationships between the features and the target.
* Performing more in depth EDA.
