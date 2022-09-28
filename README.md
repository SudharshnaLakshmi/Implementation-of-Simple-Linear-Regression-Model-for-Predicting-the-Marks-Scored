# Implementation of Simple Linear Regression Model for Predicting the Marks Scored

## AIM
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the needed packages
2. Assigning hours To X and Scores to Y
3. Plot the scatter plot
4. Use mse,rmse,mae formmula to find

## Program
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Sudharshna Lakshmi S
RegisterNumber:  212221230110
*/
```
```

#import files
import numpy as np
import pandas as pd
df=pd.read_csv('student_scores.csv')

#assigning hours To X and Scores to Y
X=df.iloc[:,:-1].values
Y=df.iloc[:,1].values
print("X=",X)
print("Y=",Y)

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/3,random_state=0)

from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(X_train,Y_train)

Y_pred=reg.predict(X_test)
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error

plt.scatter(X_train,Y_train,color='brown')
plt.plot(X_train,reg.predict(X_train),color='orange')
plt.title('Training set(H vs S)',color='green')
plt.xlabel('Hours',color='pink')
plt.ylabel('Scores',color='pink')

plt.scatter(X_test,Y_test,color='brown')
plt.plot(X_test,reg.predict(X_test),color='violet')
plt.title('Test set(H vs S)',color='brown')
plt.xlabel('Hours',color='grey')
plt.ylabel('Scores',color='grey')

mse=mean_squared_error(Y_test,Y_pred)
print('MSE = ',mse)

mae=mean_absolute_error(Y_test,Y_pred)
print('MAE = ',mae)

rmse=np.sqrt(mse)
print('RMSE = ',rmse)

```

## Output
![simple](./1.png)

![output](2.png)

![output](3.png)

![output](4.png)

![output](5.png)

![output](6.png)

![output](7.png)

![out](./8.png)

![out](9.png)

![out](./10.png)

![out](11.png)

![out](12.png)

![out](13.png)


## Result
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
