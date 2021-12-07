#Steps:- 
#1 Import all lib. 
#2 Get the data in the form of array with the help of numpy
#3 Create Dataframe from that array with the help of Pandas
#4 Create reg model and fit the data frame in req for example "model.fit(df[['Area']].values,df.Price)"
#5 before prediction make sure you're providing 2d value for the convert 1d array to 2d i.e 2D_Array = 1D_array[np.newaxis, :]
#6 predict the price model.predict(2d_array)

import numpy as np
import pandas as pd
from sklearn import linear_model
import matplotlib.pyplot as plt

df=pd.read_csv("Raw data.csv")
print(df)

plt.scatter(df.Area, df.Price)
plt.xlabel('Area')
plt.ylabel('Price')
plt.show()

reg= linear_model.LinearRegression()
reg.fit(df[['Area']].values,df.Price)

test=np.array([125])
Test=test[np.newaxis, :]
print(reg.predict(Test))

#-----------------------------with csv data file---------------------------------------

df=pd.read_csv("Raw data.csv")
print(df)

plt.scatter(df.Area, df.Price)
plt.xlabel('Area')
plt.ylabel('Price')
plt.show()

reg= linear_model.LinearRegression()
reg.fit(df[['Area']].values,df.Price)

test=np.array([600])
Test=test[np.newaxis, :]
reg.predict(Test)

d = pd.read_csv("Area.csv") 
print(d)
predicted_price =reg.predict(d.values)

d['Prices']=predicted_price #TO ADD NEW COULUMN FOR PRICES AGAINST AREA


plt.xlabel('Area')
plt.ylabel('Price')
plt.scatter(df.Area, df.Price)
plt.plot(d.Area, d.Prices)

####----------------------------without CSV data file----------------------------------

a = np.arange(100,500,30)
b=np.arange(2000,10000,600)

# intialise data of lists.
data = {'Area':a,
        'Price':b}
 
# Create DataFrame
df = pd.DataFrame(data)
 
# Print the output.
print(df)

model=linear_model.LinearRegression()
model.fit(df[['Area']].values,df.Price)

test=np.array([100])
Test=test[np.newaxis, :]
model.predict(Test)

T = np.arange(100,1000,33)
# intialise data of lists.
data1 = {'Area1':T}
 
# Create DataFrame
D = pd.DataFrame(data1)
 
# Print the output.
#print(D)

predicted_price =model.predict(D.values)

D['Price1']=predicted_price

plt.scatter(df.Area,df.Price)
plt.plot(D.Area1,D.Price1)
