
#import some important libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error,r2_score
#load the dataset

url=r'C:\Users\User\Desktop\ML with simplilearn\chatbot\auto_mpg_sample.csv'
column_names=['mpg','cylinders','displacement','horsepower','weight','acceleration','model_year','origin','car_name']
data=pd.read_csv(url)
print("loading the data:")
print(data.head())
#data preprocessing (removing  missing values)

data.dropna(inplace=True)
data=data[['horsepower','mpg']]
print("data after removing missing values:")
print(data.head())

#prepare data for the linear regression

X=data['horsepower'].values.reshape(-1,1) #independent ya explanantory
y=data['mpg'].values   #dependent ya response

#fit linear regression model 

linear_model=LinearRegression()
linear_model.fit(X,y) #fit the model 
y_pred_linear=linear_model.predict(X) 
#plot linear regression results

plt.figure(figsize=(10,6))
plt.scatter(X,y_pred_linear,color='green',label='actual data')
plt.plot(X,y_pred_linear,color='red',linewidth=2,label='linear regression line ')
plt.title('regression line mpg vs horse power')
plt.xlabel('horsepower')
plt.ylabel('mpg')
plt.legend()
plt.show()
#evaluate linear regression model 

mse_linear=mean_squared_error(y,y_pred_linear)
r2_linear=r2_score(y,y_pred_linear)
print(f"mean square error {mse_linear:.2f}")
print(f"r square error {r2_linear:.2f}")
#prepare data for the polynomial regression

poly=PolynomialFeatures(degree=2)
X_poly=poly.fit_transform(X)
#fit polynomail regression model

poly_model=LinearRegression()
poly_model.fit(X_poly,y)
y_pred_poly=poly_model.predict(X_poly)
#plot polynomial regression results
plt.figure(figsize=(10,6))
plt.scatter(X,y,color='blue', label='actual data') #actual data
plt.plot(X,y_pred_poly,color='green',linewidth=2,label='quadratic regression line')
plt.title("quadratic regression:MPG vs Horsepower")
plt.xlabel('horsepower')
plt.ylabel('mpg')
plt.legend()
plt.show()
#evaluate polynomial regression model 

mse_poly=mean_squared_error(y,y_pred_poly)
r2_poly=r2_score(y,y_pred_poly)
print(f"quadratic mean error:{mse_poly:.2f}")
print(f"r squared:{r2_poly:.2f}")
#compare model performance 

print(f"improvement in r sqyuared from linear to quadratic regression:{r2_poly-r2_linear:.2f}")
print(f"reduction in mse from linear to quadratic equatoion:{mse_linear-mse_poly:.2f} ")
while True:
    #prediction using linear regression model 
    horsepower_input=float(input("enter the horsepower for linear regression prediction"))
    predicted_mpg_linear=linear_model.predict([[horsepower_input]])
    
    print(f"predicted mpg using linear regression for horsepower={horsepower_input}:{predicted_mpg_linear[0]:.2f}")
    #prediction using the polynomail regression model 
    predicted_mpg_poly=poly_model.predict(poly.transform([[horsepower_input]]))
    print(f"predicted mpg using quadratic regression for horsepower={horsepower_input}:{predicted_mpg_linear[0]:.2f}")
    doagain=input("if you want to predict again for some value press  y otherwise n:")
    if doagain=='n':
        break