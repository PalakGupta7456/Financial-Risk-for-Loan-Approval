from flask import Flask, request, render_template

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application=Flask(__name__)

app=application

## route for a home page

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata', methods=['GET','POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('home.html')
    else:
        data = CustomData(
        Age=int(request.form.get('age')),
        AnnualIncome=int(request.form.get('annual_income')),
        CreditScore=int(request.form.get('credit_score')),
        EmploymentStatus=(request.form.get('employment_status')),
        EducationLevel=(request.form.get('education_level')),
        LoanAmount=int(request.form.get('loan_amount')),
        MaritalStatus=(request.form.get('marital_status')),
        LoanDuration=int(request.form.get('loan_duration')),
        NumberOfDependents=int(request.form.get('number_of_dependents')),
        HomeOwnershipStatus=(request.form.get('home_status')),
        MonthlyDebtPayments=float(request.form.get('monthly_debt_payments')),
        CreditCardUtilizationRate=float(request.form.get('credit_card_utilization_rate')),
        NumberOfOpenCreditLines=int(request.form.get('number_of_open_credit_lines')),
        NumberOfCreditInquiries=int(request.form.get('number_of_credit_inquiries')),
        DebtToIncomeRatio=float(request.form.get('debt_to_income_ratio')),
        BankruptcyHistory=int(request.form.get('bankruptcy_history')),
        LoanPurpose=(request.form.get('loan_purpose')),
        PreviousLoanDefaults=int(request.form.get('previous_loan_defaults')),
        PaymentHistory=int(request.form.get('payment_history')),
        LengthOfCreditHistory=int(request.form.get('length_of_credit_history')),
        SavingsAccountBalance=int(request.form.get('savings_account_balance')),
        CheckingAccountBalance=int(request.form.get('checking_account_balance')),
        TotalAssets=int(request.form.get('total_assets')),
        TotalLiabilities=int(request.form.get('total_liabilities')),
        UtilityBillsPaymentHistory=float(request.form.get('utility_bills_payment_history')),
        JobTenure=int(request.form.get('job_tenure')),
        BaseInterestRate=float(request.form.get('base_interest_rate')),
        InterestRate=float(request.form.get('interest_rate')),
        MonthlyLoanPayment=float(request.form.get('monthly_loan_payment')),
        TotalDebtToIncomeRatio=float(request.form.get('total_debt_to_income_ratio')),
        RiskScore=float(request.form.get('risk_score'))
    )


        pred_df=data.get_data_as_data_frame()
        print(pred_df)

        predict_pipeline=PredictPipeline()
        results= predict_pipeline.predict(pred_df)

        return render_template('home.html', results=results[0])
    
if __name__=="__main__":
    app.run(host="0.0.0.0")
    

    






    
