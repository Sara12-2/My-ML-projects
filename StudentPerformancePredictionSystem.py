import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,classification_report

df=pd.read_csv(r'chatbot\StudentsPerformance_1000.csv')
df.head()
#create average score and pass fail column
df['average']=df[['math score','reading score','writing score']].mean(axis=1)
df['pass']=df['average'].apply(lambda x: 'pass' if x>=50 else 'fail')
df.head()
#encode categorical columns 
le = LabelEncoder()

categorical_cols = ['gender', 'race/ethnicity', 'parental level of education', 'lunch', 'test preparation course']

for columns in categorical_cols:
  df[columns]=le.fit_transform(df[columns])
df.head()
#define features and targets
X=df[categorical_cols] #input
y=df['pass'] #output
#split dataset
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=42,test_size=0.2)
#train classifier
model=RandomForestClassifier()
model.fit(X_train,y_train)
#make predictions
y_pred=model.predict(X_test)
print(accuracy_score(y_test,y_pred))
print(classification_report(y_test,y_pred))
gender = int(input("Gender (0=female, 1=male): "))
race = int(input("Race (0=A, 1=B, ..., 4=E): "))
education = int(input("Education (0=associate's, ..., 5=some high school): "))
lunch = int(input("Lunch (0=free/reduced, 1=standard): "))
prep = int(input("Test Prep (0=completed, 1=none): "))

custom_input = [[gender, race, education, lunch, prep]]
prediction = model.predict(custom_input)
result = "Pass" if prediction[0] == 1 else "Fail"
print("Prediction Result:", result)
