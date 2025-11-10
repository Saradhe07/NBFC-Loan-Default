# NBFC-Loan-Default
A machine learning model to predict loan default risk based on  customer financial and credit features. It uses LightGBM, a powerful gradient boosting  framework, combined with feature engineering, cross-validation, and scaling techniques to  ensure robust and generalizable results.

# Technical Workflow:
1. Library Imports

   • pandas, numpy → data manipulation and numerical operations
   
   • scikit-learn → preprocessing, model evaluation, scaling
   
   • lightgbm → model training (LGBMClassifier)
   
   • StratifiedKFold → maintains class distribution during cross-validation
   
   • StandardScaler → feature normalization
   
2. Data Loading
   • Train_set.csv → contains features and the target column default
   
   • Test_set.csv → used for final prediction and submission

3. Preprocessing
   
   • Target Separation: The default column was extracted as y, and X retained feature columns.
   
   • Missing Value Handling: All missing values were replaced with -999 (a placeholder value).
   
   • Categorical Encoding: Used LabelEncoder for categorical columns to convert them into numeric form.
                           Encoder fitted on combined train + test columns to ensure consistency.
   
4. Feature Engineering

<img width="1329" height="234" alt="Screenshot 2025-11-08 204454" src="https://github.com/user-attachments/assets/24cd368d-aa34-41fe-b563-8df1fba63d50" />

