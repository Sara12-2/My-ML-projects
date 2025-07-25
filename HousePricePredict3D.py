#import different libraries
#jb 1 s ziada independent variable hun er wo y(dependent) pr influence kr rhy hon isko multiple regression khty hain
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Load the dataset
data = pd.read_csv("chatbot/house_data_multiple_regression.csv")

#data preprocessing ,selecting relevent columns and handling missing data
data = data[['sqft_living', 'sqft_lot', 'price']]  
data.dropna(inplace=True)

#prepare data for multiple regression 
X=data[['sqft_living','sqft_lot']].values 
y=data['price'].values
 
#fit multiple regression model

multiple_regress_model = LinearRegression()
multiple_regress_model.fit(X, y)

# make predictions
y_pred = multiple_regress_model.predict(X)

#plot result 3D scatter and regression plane
fig = plt.figure(figsize=(12,8))
ax=fig.add_subplot(111,projection='3d') #axis 3D

#scatter plot
ax.scatter(data['sqft_living'],data['sqft_lot'],y,color='blue',label='Actual data')

# meshgrid for plotting regtression line 
x_surf,y_surf=np.meshgrid(np.linspace(X[:,0].min(),X[:,0].max(),100),
                          np.linspace(X[:,1].min(),X[:,1].max(),100))
z_surf=multiple_regress_model.predict(np.c_[x_surf.ravel(),y_surf.ravel()]).reshape(x_surf.shape)
 
 
#plot regression plane
ax.plot_surface(x_surf,y_surf,z_surf,color='red',alpha=0.5,rstride=100,cstride=100)

#set label and title
ax.set_xlabel('square foot living area')
ax.set_ylabel('square foot lot area')
ax.set_zlabel('house price')
ax.set_title('multiple regression:house vs living area vs lot area')
plt.show()


#take the input from independent and predict price
sqft_living_input=float(input("enter the square footage of living area:"))
sqft_lot_input=float(input("enter the square footage of lot area:"))

#predict price using the fitted moddel
predicted_price=multiple_regress_model.predict([[sqft_living_input,sqft_lot_input]])
print(f"output for living area predicted={sqft_living_input} and lot area is {sqft_lot_input}: is {predicted_price[0]}")
