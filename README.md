# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY

Explain the problem statement

Neural networks consist of simple input/output units called neurons (inspired by neurons of the human brain). These input/output units are interconnected and each connection has a weight associated with it. Neural networks are flexible and can be used for both classification and regression. In this article, we will see how neural networks can be applied to regression problems.

Regression helps in establishing a relationship between a dependent variable and one or more independent variables. Regression models work well only when the regression equation is a good fit for the data. Most regression models will not fit the data perfectly. Although neural networks are complex and computationally expensive, they are flexible and can dynamically pick the best type of regression, and if that is not enough, hidden layers can be added to improve prediction.

First import the libraries which we will going to use and Import the dataset and check the types of the columns and Now build your training and test set from the dataset Here we are making the neural network 2 hidden layer(1st hidden layer contains 7 neurons and 2nd hidden layer contains 14 neurons) with activation layer as relu and with their nodes in them. Now we will fit our dataset and then predict the value.
## Neural Network Model

Include the neural network model diagram.

![image](https://github.com/user-attachments/assets/155d6fb0-a3cb-4935-b9e6-ad9cc5ac63e9)

## DESIGN STEPS

### STEP 1:

Loading the dataset

### STEP 2:

Split the dataset into training and testing

### STEP 3:

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4:

Build the Neural Network Model and compile the model.

### STEP 5:

Train the model with the training data.

### STEP 6:

Plot the performance plot

### STEP 7:

Evaluate the model with the testing data.

## PROGRAM
### Name: A.ARUVI
### Register Number: 212222230014
```python


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from google.colab import auth
import gspread
from google.auth import default
auth.authenticate_user()
creds, _ = default()
gc = gspread.authorize(creds)
worksheet = gc.open('dl_ex1').sheet1
data = worksheet.get_all_values()
dataset1 = pd.DataFrame(data[1:], columns=data[0])
dataset1 = dataset1.astype({'Input':'float'})
dataset1 = dataset1.astype({'Output':'float'})
dataset1.head()
X = dataset1[['Input']].values
y = dataset1[['Output']].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33,random_state=33)
Scaler = MinMaxScaler()
Scaler.fit(X_train)
X_train1 = Scaler.transform(X_train)
ai_brain = Sequential([
    Dense(8,activation = 'relu'),
    Dense(10,activation = 'relu'),
    Dense(1)
    ])
ai_brain.compile(optimizer='rmsprop',loss='mse')
ai_brain.fit(X_train1,y=y_train,epochs=2000)
loss_df = pd.DataFrame(ai_brain.history.history)
loss_df.plot()
X_test1 = Scaler.transform(X_test)
ai_brain.evaluate(X_test1,y_test)
X_n1 = [[4]]
X_n1_1 = Scaler.transform(X_n1)
ai_brain.predict(X_n1_1)






```

## Dataset Information

![image](https://github.com/user-attachments/assets/4d630e2b-f652-4929-9a1e-63b83390c08d)


## OUTPUT


### Training Loss Vs Iteration Plot

![image](https://github.com/user-attachments/assets/c66ce4b4-4bab-4ef1-861f-5d5639fd6006)



### Test Data Root Mean Squared Error

Find the test data root mean squared error


![image](https://github.com/user-attachments/assets/e8a1aac5-6e0a-4bb7-a9ec-6993993f79c0)


### New Sample Data Prediction

![image](https://github.com/user-attachments/assets/a5dd89f5-c0b4-47f0-b006-67aa3bcd43a7)



## RESULT

Thus a neural network regression model for the given dataset is written and executed successfully.
