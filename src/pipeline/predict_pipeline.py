import sys
import pandas as pd

from src.exception import CustomException
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            model_path='artifacts\model.pkl'
            preprocessor_path='artifacts\preprocessor.pkl'
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            data_scaled=preprocessor.transform(features)
            preds=model.predict(data_scaled)
            return preds
        except Exception as e:
            raise CustomException(e,sys)
    


class CustomData:
    def __init__(self,
        Age: int, 
        AnnualIncome: int, 
        CreditScore : int, 
        EmploymentStatus:int,
        EducationLevel: int,
        LoanAmount : int, 
        LoanDuration: int,
        MaritalStatus: int,
        NumberOfDependents : int, 
        HomeOwnershipStatus: int,
        MonthlyDebtPayments : int,
        CreditCardUtilizationRate : float, 
        NumberOfOpenCreditLines: int,
        NumberOfCreditInquiries: int , 
        DebtToIncomeRatio: float,
        BankruptcyHistory: int,
        LoanPurpose: int,
        PreviousLoanDefaults: int, 
        PaymentHistory: int, 
        LengthOfCreditHistory: int,
        SavingsAccountBalance : int, 
        CheckingAccountBalance : int, 
        TotalAssets : int,
        TotalLiabilities : int, 
        UtilityBillsPaymentHistory : float, 
        JobTenure : int,
        BaseInterestRate : float, 
        InterestRate : float, 
        MonthlyLoanPayment : float,
        TotalDebtToIncomeRatio : float,
        RiskScore: float):
        
        self.Age=Age,
        self.AnnualIncome= AnnualIncome,
        self.CreditScore = CreditScore,
        self.EmploymentStatus=EmploymentStatus,
        self.EducationLevel=EducationLevel,
        self.LoanAmount =LoanAmount,
        self.MaritalStatus= MaritalStatus,
        self.LoanDuration = LoanDuration,
        self.NumberOfDependents =NumberOfDependents, 
        self.HomeOwnershipStatus=HomeOwnershipStatus,
        self.MonthlyDebtPayments = MonthlyDebtPayments,
        self.CreditCardUtilizationRate =CreditCardUtilizationRate, 
        self.NumberOfOpenCreditLines = NumberOfOpenCreditLines,
        self.NumberOfCreditInquiries =NumberOfCreditInquiries , 
        self.DebtToIncomeRatio = DebtToIncomeRatio,
        self.BankruptcyHistory= BankruptcyHistory,
        self.LoanPurpose=LoanPurpose,
        self.PreviousLoanDefaults = PreviousLoanDefaults, 
        self.PaymentHistory = PaymentHistory, 
        self.LengthOfCreditHistory = LengthOfCreditHistory,
        self.SavingsAccountBalance =SavingsAccountBalance, 
        self.CheckingAccountBalance = CheckingAccountBalance, 
        self.TotalAssets = TotalAssets,
        self.TotalLiabilities = TotalLiabilities, 
        self.UtilityBillsPaymentHistory =UtilityBillsPaymentHistory, 
        self.JobTenure =JobTenure,
        self.BaseInterestRate =BaseInterestRate, 
        self.InterestRate = InterestRate, 
        self.MonthlyLoanPayment =MonthlyLoanPayment,
        self.TotalDebtToIncomeRatio =TotalDebtToIncomeRatio,
        self.RiskScore=RiskScore
    

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict={
                "Age": self.Age,
                 
                "AnnualIncome": self.AnnualIncome, 
                "CreditScore" : self.CreditScore, 
                "EmploymentStatus":self.EmploymentStatus,
                "EducationLevel":self.EducationLevel,
                "LoanAmount" : self.LoanAmount, 
                "MaritalStatus": self.MaritalStatus,
                "LoanDuration": self.LoanDuration,
                "NumberOfDependents" : self.NumberOfDependents, 
                "HomeOwnershipStatus": self.HomeOwnershipStatus,
                "MonthlyDebtPayments" : self.MonthlyDebtPayments,
                "CreditCardUtilizationRate" : self.CreditCardUtilizationRate, 
                "NumberOfOpenCreditLines": self.NumberOfOpenCreditLines,
                "NumberOfCreditInquiries": self.NumberOfCreditInquiries , 
                "DebtToIncomeRatio": self.DebtToIncomeRatio,
                "BankruptcyHistory": self.BankruptcyHistory,
                "LoanPurpose": self.LoanPurpose,
                "PreviousLoanDefaults": self.PreviousLoanDefaults, 
                "PaymentHistory": self.PaymentHistory, 
                "LengthOfCreditHistory": self.LengthOfCreditHistory,
                "SavingsAccountBalance" : self.SavingsAccountBalance, 
                "CheckingAccountBalance" : self.CheckingAccountBalance, 
                "TotalAssets" : self.TotalAssets,
                "TotalLiabilities" : self.TotalLiabilities, 
                "UtilityBillsPaymentHistory" : self.UtilityBillsPaymentHistory, 
                "JobTenure" : self.JobTenure,
                "BaseInterestRate" : self.BaseInterestRate, 
                "InterestRate" : self.InterestRate, 
                "MonthlyLoanPayment" : self.MonthlyLoanPayment,
                "TotalDebtToIncomeRatio" : self.TotalDebtToIncomeRatio,
                "RiskScore":self.RiskScore
                        
            }
            return pd.DataFrame(custom_data_input_dict)
        except Exception as e:
            raise CustomException(e, sys)
            




