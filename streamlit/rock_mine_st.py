import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

st.title("Rock vs Mine")

# with st.echo():
# Read data as pandas dataframe from csv 
df = pd.read_csv('../data/sonar_data.csv', header=None)
st.subheader('The Sonar dataset')
st.dataframe(df)
st.write('The first 5 Rows of Data: ',df.head())
st.write('Shape of the Data: ',df.shape)

st.write('Statistical summary of data: ',df.describe())

st.write('Number of unique values in Label: ', df[60].value_counts())

st.write('Data types of each columns: ', df.dtypes)

st.write('Groupby label column by its mean', df.groupby(60).mean())

# Seprating features and labels
X = df.drop(60, axis=1)
Y = df[60]

# Spliting train and test data from the original data set
X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size=0.2, stratify=Y, random_state=1)

st.write('Original data shape: ',X.shape) 
st.write('Train data shape: ', X_train.shape)
st.write('Test data shape: ', X_test.shape)

# Using ML model
model = LogisticRegression()

# Training the model
st.write('Training the model: ',model.fit(X_train, y_train))
# Check training accuracy
X_train_prediction = model.predict(X_train)
train_accuracy = accuracy_score(X_train_prediction, y_train)

st.write('training accuracy score: ',train_accuracy)

# Check testing accuracy
X_test_prediction = model.predict(X_test)
test_accuracy = accuracy_score(X_test_prediction, y_test)
st.write('testing accuracy score: ',test_accuracy)

# input_data = st.text_input('Input your data to predict')

# Use an input data for prediction 

input_data = (0.0454,0.0472,0.0697,0.1021,0.1397,0.1493,0.1487,0.0771,0.1171,0.1675,0.2799,0.3323,0.4012,0.4296,0.5350,0.5411,0.6870,0.8045,0.9194,0.9169,1.0000,0.9972,0.9093,0.7918,0.6705,0.5324,0.3572,0.2484,0.3161,0.3775,0.3138,0.1713,0.2937,0.5234,0.5926,0.5437,0.4516,0.3379,0.3215,0.2178,0.1674,0.2634,0.2980,0.2037,0.1155,0.0919,0.0882,0.0228,0.0380,0.0142,0.0137,0.0120,0.0042,0.0238,0.0129,0.0084,0.0218,0.0321,0.0154,0.0053)

# Converting input to numpy array

input_data_as_numpy_array = np.asarray(input_data)

# Checking the current shape of data 

input_data_as_numpy_array.shape

# Since our data contain 60 rows and no columns we can Reshaping data to 60 columns in 1 row 

input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

# Checking the current shape of data 

input_data_reshaped.shape

# Predicting using the trained model

prediction = model.predict(input_data_reshaped)
st.write(prediction)

if(prediction[0] == 'R'):
    st.write('The Object is a Rock')
else:
    st.write('The Object is a Mine')

st.balloons()