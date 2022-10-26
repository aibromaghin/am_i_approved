# import dependencies
import streamlit as st
import pickle
import pandas as pd
import numpy as np
import sklearn
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


# generate user interface
st.write("# Consumer Loan Approval Prediction")

# gather user data
col1, col2, col3 = st.columns(3)

loan_product = col1.selectbox("Enter loan category:", ['Unsecured', 'Credit card', 'Auto', 'Recreational'])
amount = col2.number_input("How much would you like to borrow?")
credit_score = col3.number_input("What is your credit score?")
rent = col1.number_input("How much rent do you pay each month?")
monthly_income = col2.number_input("How much do you earn from your primary job each month?")
other_income = col3.number_input("Enter any other monthly income (Side hustle, rental income, etc)")
debts = col1.number_input("Enter your total monthly payments (Credit cards, car loan, etc)")

# input user data to dataframe 
df_pred = pd.DataFrame(data=[[loan_product, amount, credit_score, rent, monthly_income, other_income, debts]],\
     columns=['loan product', 'amount', 'credit score', 'rent', 'monthly income', 'other income', 'debts'])

# encode loan categories
df_pred['loan product'].replace({'Unsecured': 1, 'Credit card': 2, 'Auto': 3, 'Recreational': 4}, inplace=True)
# round credit score
df_pred['credit score'] = df_pred['credit score'].apply(lambda x: round(x/100)*100)

# helper function to calculate dti
def calculate_dti(row):
    debt = row['debts']
    income = row['monthly income'] + row['other income']
    if income == 0:
        return 100
    else:
        return round(debt/income*100, 2)

# add features
df_pred['has debts'] = np.where(df_pred['debts'] > 0, 1, 0)
df_pred['pays rent'] = np.where(df_pred['rent'] > 0, 1, 0)
df_pred['has credit'] = np.where(df_pred['credit score'] > 0, 1, 0)
df_pred['dti'] = df_pred.apply(lambda row: calculate_dti(row), axis=1)

# import training data to apply scaling
df = pd.read_csv('final_consumer_loans_data.csv', index_col=0,)
target = 'status'
feature_cols = [col for col in df.columns if col != target]
features = df[feature_cols]
target = df[target]

# split into train and test sets
x_train, x_test, y_train, y_test = train_test_split(features, target, test_size = 0.2)

# scale numerical data
scaler = MinMaxScaler(feature_range=(0, 1))
cols = ['amount', 'credit score', 'rent', 'monthly income', 'other income', 'debts']
x_train[cols] = scaler.fit_transform(x_train[cols])

# scale user data
df_pred[cols] = scaler.transform(df_pred[cols])
user_data = df_pred.to_numpy()

# generate prediction
with open('random_forest.pkl','rb') as f:
     model = pickle.load(f)
prediction = model.predict(user_data)
# get class probabilities,
proba = model.predict_proba(user_data)
prob_rejected = round(proba[0][0] * 100, 2)
prob_approved = round(proba[0][1] * 100, 2)


# display results
if st.button('Predict approval'):
    if prediction == 0:
        st.write('<p><big><strong>Your application is likely to be rejected</big></strong></p>', unsafe_allow_html=True)
        st.write('<p><big>Your application has a ', str(prob_rejected), '%', ' chance of being rejected</p>', unsafe_allow_html=True)
    if prediction ==  1:
        st.write('<p><big><strong>Your application is likely to be approved</big></strong></p>', unsafe_allow_html=True)
        st.write('<p><big>Your application has a ', str(prob_approved), '%', ' chance of being approved</big></p>', unsafe_allow_html=True)
    
