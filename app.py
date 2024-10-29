import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import pandas_datareader as data
import yfinance as yf
from keras.models import load_model
import streamlit as st


start = '2010-01-01'
end = '2019-12-31'

# df = data.DataReader('AAPL', 'yahoo', start, end)
# df.head()
st.title('Stock Trend Prediction')

user_input = st.text_input('Enter Stock Ticker', 'AAPL')
df = yf.download(user_input, start, end)

# Describing Data
st.subheader('Data from 2010 - 2019')
st.write(df.describe())

# Visualizations
st.subheader('Closing Price cs Time Chart')
fig = plt.figure(figsize=(12,6))
plt.plot(df.Close)
st.pyplot(fig)

st.subheader('Closing Price cs Time Chart with 100MA')
ma100 = df.Close.rolling(100).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(ma100)
plt.plot(df.Close)
st.pyplot(fig)

st.subheader('Closing Price cs Time Chart with 100MA & 200MA')
ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(ma100, 'r')
plt.plot(ma200, 'g')
plt.plot(df.Close, 'b')
st.pyplot(fig)


# Splitting data into training and testing
data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)]) # 70% data for training
data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70): int(len(df))]) # 30% data for testing

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))

data_training_array = scaler.fit_transform(data_training)

# Splitting data into x_train and y_train
# x_train = []
# y_train = []

# for i in range(100, data_training_array.shape[0]):
#     x_train.append(data_training_array[i - 100:i]) # should start from 1, go till i
#     y_train.append(data_training_array[i, 0]) # considering only one column

# x_train, y_train = np.array(x_train), np.array(y_train)

# Load my model
model = load_model('keras_model.h5')

# Testing Part
past_100_days = data_training.tail(100)
final_df = pd.concat([past_100_days, data_testing], ignore_index=True)
input_data = scaler.fit_transform(final_df)

x_test = []
y_test = []

for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100: i])
    y_test.append(input_data[i, 0])

x_test, y_test = np.array(x_test), np.array(y_test)
y_predicted = model.predict(x_test)
scaler = scaler.scale_

scale_factor = 1 / scaler[0]
y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor

# Final Graph
st.subheader('Predictions vs Original')
fig2 = plt.figure(figsize=(12,6))
plt.plot(y_test, 'b', label="Original Price")
plt.plot(y_predicted, 'r', label="Predicted Price")
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)

# streamlit run app.py