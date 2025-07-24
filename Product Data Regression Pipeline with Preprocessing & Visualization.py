#importing libraries
import sklearn #implementing ML models
import pandas as pd
import numpy as np #data processing and numerical 
from sklearn.preprocessing import MinMaxScaler,LabelEncoder,StandardScaler
from sklearn.impute import SimpleImputer #filling missing values
import statsmodels.api as sm #statistical calculation
from statsmodels.api import add_constant #in linear equation
import matplotlib.pyplot as plt #plotting
#main part of the function
def main():
    # preparing the data
    #load the dataset
    
    file_path='chatbot/sample_products.csv'
    df=pd.read_csv(file_path)
    print("\n initial data :")
    print(df.head(8))  
    print(f"\ntotal row and column before cleaning :{df.shape}")
    
    
    #data cleaning
    df_cleaned=df.dropna() #drop rows with missing values(na mean not avaible)
    print("\n data after dropping missing values:")
    print(df_cleaned.head(8))
    print(f"\n rows and columns after dropping missing values:{df_cleaned.shape}")
    
    
    #use simple imputer for filling the missing values
    
    imputer=SimpleImputer(strategy='mean')
    df[['Age','Price','TotalAmount','Quantity']]=imputer.fit_transform(df[['Age','Price','TotalAmount','Quantity']])
    print("\n After imputation:")
    print(df.head(8))
    print(f"\nafter all the imputation number of rows and columns are:{df.shape}")
    
    
    #handling categorical data 
    # #label encoding for gender
    
    le=LabelEncoder()
    df['Gender']=le.fit_transform(df['Gender'])
    print("encoding:")
    print(df['Gender'].head())
    print(f"\nafter all the encoding number of rows and columns:{df.shape}")
    #minmax scaler
    
    scaler=MinMaxScaler()
    df[['Price','TotalAmount','Quantity']]=scaler.fit_transform( df[['Price','TotalAmount','Quantity']])
    print("\n",df[['Price','TotalAmount','Quantity']].head())
    print(f"\nafter all the scaling:{df.shape}")
    #standarized scaler
    
    standard_scaler=StandardScaler()
    df[['Price','TotalAmount','Quantity']]=standard_scaler.fit_transform( df[['Price','TotalAmount','Quantity']])
    print("\n",df[['Price','TotalAmount','Quantity']].head())
    print(f"\nafter all the standard scaling:{df.shape}")
    #preparaing data for OLS regression
    
    
    X=df['Price'] #independent variable
    y=df['TotalAmount'] #dependent
    # adding constant term for intercept
    
    X=sm.add_constant(X)
    #fitting the OS model 
    
    model=sm.OLS(y,X).fit()
    print("\n press any key to continue....")
    input()
    #summary of the model 
    
    print(model.summary())
    print("press any key to continue.....")
    input()
    # Plotting regression line and scatter
    
    plt.scatter(df['Price'], df['TotalAmount'], color='skyblue', label='Data Points')
    plt.plot(df['Price'], model.predict(sm.add_constant(df['Price'])), color='red', linewidth=2, label='OLS Regression Line')
    plt.title('OLS Regression: TotalAmount vs Price')
    plt.xlabel('Price (Standard Scaled)')
    plt.ylabel('TotalAmount (Standard Scaled)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
if __name__ == "__main__":
    main()