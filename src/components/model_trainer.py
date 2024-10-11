# Basic Import
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
import os
import sys


# Modelling
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, Ridge,Lasso
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import accuracy_score, classification_report,ConfusionMatrixDisplay, precision_score, recall_score, f1_score, roc_auc_score,roc_curve 


from dataclasses import dataclass

from src.logger import get_logger
from src.exception import CustomException
from src.utils import save_object, evaluate_models

logger=get_logger()

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logger.info("Split training and test input data")
            X_train, y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]

            )
            models = {
            # "Logistic Regression": LogisticRegression(),
            "Decision Tree": DecisionTreeClassifier(),
            "Random Forest": RandomForestClassifier(),
            "Gradient Boost": GradientBoostingClassifier(),
            "Adaboost": AdaBoostClassifier(),
            "Xgboost": XGBClassifier()
        }

            params = {
                # "Logistic Regression": {  # Fixed name to match the dictionary keys
                #     "penalty": ['l1', 'l2'],
                #     "C": [100, 10, 1.0, 0.1, 0.01],
                #     "solver": ['liblinear', 'saga']
                # },
                "Decision Tree": {
                    'criterion': ['gini', 'entropy', 'log_loss'],
                    # 'splitter': ['best', 'random'],
                    # 'max_depth': [None, 2, 5, 10],
                    # 'max_features': [None, 'sqrt', 'log2']
                },
                "Gradient Boost": {
                    # "loss": ['deviance', 'exponential'],
                    "criterion": ['friedman_mse'],
                    # "min_samples_split": [2, 8, 15],
                    "n_estimators": [100, 200, 300],
                    # "max_depth": [3, 5, 8, 10, None]
                },
                "Random Forest": {
                    # "max_depth": [None, 10, 15, 20],
                    # "max_features": ['sqrt', 'log2', 0.5],
                    # "min_samples_split": [2, 8, 15],
                    "n_estimators": [100, 200, 500]
                },
                "Adaboost": {
                    "n_estimators": [50, 100, 150],
                    # "algorithm": ['SAMME', 'SAMME.R']
                },
                "Xgboost": {
                    # "learning_rate": [0.01, 0.05, 0.1],
                    # "max_depth": [3, 5, 7, 10],
                    "n_estimators": [100, 200, 300],
                    # "colsample_bytree": [0.5, 0.7, 1]
                }
            }



            


            model_report:dict=evaluate_models(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, models=models, param=params)

            ## To get best model score from dict
            best_model_score=max(sorted(model_report.values()))

            ## To get best model name from dict
            best_model_name=list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model=models[best_model_name]

            if best_model_score<0.6:
                raise CustomException("No best model found")
            
            logger.info(f"Best found model on both training and testing dataset")
            logger.info(f"Best model found is {best_model}")
            

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted=best_model.predict(X_test)

            r2_square=r2_score(y_test,predicted)
            return r2_square

        except Exception as e:
            raise CustomException(e,sys)
        

            


