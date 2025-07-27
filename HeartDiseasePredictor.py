#classification
#a binary logistic regression model
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns #built on the functionality of matplotlib
from sklearn.model_selection import train_test_split #(to split data)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score,roc_curve,auc
#auc mean area under the curve
#loading data
url="chatbot\heart_disease_large.csv"
data=pd.read_csv(url)
#data preprocessiong 
#dropping missing values
data.dropna(inplace=True) #inplace true mtlab original data main change krta h copy modified nhi lata

#converting target variable to binary classification (0:No heart disease ,1:heart diaease)
data['target']=data['target'].apply(lambda x:1 if x>0 else 0)

#splitting the data into features and target
X=data.drop('target',axis=1)
y=data['target']

#splitting the dataset into training and testing sets
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

#creating and traing the logistic regression model 
model=LogisticRegression(max_iter=1000)
model.fit(X_train,y_train)

#predicting on the test set
y_pred=model.predict(X_test)

#evalating the model performance 
print("classification report\n",classification_report(y_test,y_pred))
print("confusion matrix\n",confusion_matrix(y_test,y_pred))
print("accuracy score\n",accuracy_score(y_test,y_pred))


# ROC Curve and AUC
y_pred_proba = model.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()

# Visualizing the classification
plt.figure(figsize=(10, 6))
sns.scatterplot(x='age', y='chol', hue='target', data=data, palette='coolwarm')
plt.title('Heart Disease Classification (Age vs Cholesterol)')
plt.show()

# Predicting a new record
new_record = pd.DataFrame([[60, 1, 3, 130, 206, 0, 0, 132, 1, 2.4, 1, 0, 3]], columns=X.columns)
prediction = model.predict(new_record)
print(f"the predicted class for the new result is is:{'Heart_disease' if prediction[0]==1 else 'no heart disease'}")
#displaying classification results in a table 
results=pd.DataFrame({'actual':y_test,'predicted':y_pred})
print("\n sample classification results:\n",results.head())