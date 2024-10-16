import sys
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

# # Basic Import
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt 
# import seaborn as sns
# # Modelling
# from sklearn.metrics import mean_squared_error, r2_score
# from sklearn.neighbors import KNeighborsRegressor
# from sklearn.tree import DecisionTreeRegressor
# from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor
# from sklearn.svm import SVR
# from sklearn.linear_model import LinearRegression, Ridge,Lasso
# from xgboost import XGBRegressor
# from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
# from sklearn.model_selection import RandomizedSearchCV
# from sklearn.model_selection import train_test_split

# from sklearn.linear_model import LogisticRegression
# from sklearn.tree import DecisionTreeClassifier

# from sklearn.ensemble import RandomForestClassifier
# from sklearn.ensemble import GradientBoostingClassifier
# from sklearn.ensemble import AdaBoostClassifier
# from xgboost import XGBClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.model_selection import StratifiedKFold

# from sklearn.model_selection import GridSearchCV

# from sklearn.metrics import accuracy_score, classification_report,ConfusionMatrixDisplay, precision_score, recall_score, f1_score, roc_auc_score,roc_curve 


import warnings
import scipy
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from src.exception import CustomException
from src.logger import get_logger
from src.utils import save_object
logger=get_logger()


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts',"preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()
    
    def get_data_transformer_object(Self):
        '''
        This function is responsible for data transformation
        
        '''
        try:
            numeric_features= ['Age', 'AnnualIncome', 'CreditScore', 'LoanAmount', 'LoanDuration',
                                'NumberOfDependents', 'MonthlyDebtPayments',
                                'CreditCardUtilizationRate', 'NumberOfOpenCreditLines',
                                'NumberOfCreditInquiries', 'DebtToIncomeRatio', 'BankruptcyHistory',
                                'PreviousLoanDefaults', 'PaymentHistory', 'LengthOfCreditHistory',
                                'SavingsAccountBalance', 'CheckingAccountBalance', 'TotalAssets',
                                'TotalLiabilities', 'UtilityBillsPaymentHistory', 'JobTenure',
                                'BaseInterestRate', 'InterestRate', 'MonthlyLoanPayment',
                                'TotalDebtToIncomeRatio', 'RiskScore']
            categorical_features= ['EmploymentStatus', 'EducationLevel', 'MaritalStatus',
                                    'HomeOwnershipStatus', 'LoanPurpose']
            
            num_pipeline=Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                     ("scaler", StandardScaler() )
                ]
            )

            cat_pipeline=Pipeline(
                steps=[("imputer", SimpleImputer(strategy="most_frequent")),
                       ("one_hot_encoder", OneHotEncoder()),
                       
                       

                ]
            )
            logger.info("Numerical columns standard scaling completed")

            logger.info("Categorical columns encoding completed")



            preprocessor=ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline,numeric_features),
                    ("cat_pipeline", cat_pipeline, categorical_features),

                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e,sys)
        

    def initiate_data_transformation(self, train_path,test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logger.info("Read train and test data completed")

            logger.info("Obtaining preprocessing object")

            preprocessing_obj=self.get_data_transformer_object()

            target_column_name="LoanApproved"
            input_feature_train_df=train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df=test_df[target_column_name]

            logger.info(
                f"Applying preprocessing object on training dataframe and testing dataframe"
            )
            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

            train_arr=np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr=np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logger.info(f"Saved preprocessing object")

            save_object(
                file_path=  self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            raise CustomException(e,sys)






