# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Load the dataset Salary.csv using pandas and view the first few rows.

2.Check dataset information and identify any missing values.

3.Encode the categorical column "Position" into numerical values using LabelEncoder.

4.Define feature variables x as "Position" and "Level", and target variable y as "Salary".

5.Split the dataset into training and testing sets using an 80-20 split.

6.Create a DecisionTreeRegressor model instance.

7.Train the model using the training data.

8.Predict the salary values using the test data.

9.Evaluate the model using Mean Squared Error (MSE) and R² Score.

10.Use the trained model to predict salary for a new input [5, 6].

## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: Vedagiri Indu Sree
RegisterNumber:  212223230236
*/
```
```
import pandas as pd
data=pd.read_csv("Salary.csv")
data.head()

data.info

data.isnull().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["Position"]=le.fit_transform(data["Position"])
data.head()

x=data[["Position","Level"]]
x.head()

y=data[["Salary"]]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.2,random_state=2)

from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor()
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)

from sklearn import metrics
mse=metrics.mean_squared_error(y_test, y_pred)
mse

r2=metrics.r2_score(y_test,y_pred)
r2

dt.predict([[5,6]])
```

## Output:
## DATA HEAD:
![image](https://github.com/user-attachments/assets/00a3014d-f11e-4638-8b0a-fa908edabb75)
## Data Info:
![image](https://github.com/user-attachments/assets/d0dced48-d0b8-43d5-b59e-caf694d0e922)
## is null() sum():
![image](https://github.com/user-attachments/assets/eaf99ce7-25e1-42f3-a64a-832608e09662)
## Data Head for salary:
![image](https://github.com/user-attachments/assets/13489eb2-b4fc-41bb-ab04-f8f3ba5d2287)
## Mean Sqaured Error:
![image](https://github.com/user-attachments/assets/61713009-49ef-45d0-9482-e334df0c698f)
## r2 value:
![image](https://github.com/user-attachments/assets/f304b467-004e-4d45-a34e-2c396f229a5b)
## Data Prediction:
![image](https://github.com/user-attachments/assets/0c9792d7-2735-428b-b8c7-43be5a28906a)

## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
