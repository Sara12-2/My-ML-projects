# 1. Required libraries import karein
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
# 2. CSV file load karein
df = pd.read_csv('chatbot\ecommerce_sample_data.csv')
df.head()
# 3. Categorical columns ko encode karein
le_location = LabelEncoder()
le_category = LabelEncoder()
le_device = LabelEncoder()

df['user_location'] = le_location.fit_transform(df['user_location'])
df['product_category'] = le_category.fit_transform(df['product_category'])
df['device_type'] = le_device.fit_transform(df['device_type'])
# 4. Features (X) aur target (y) define karein
X = df.drop('purchased', axis=1)  # sab columns except 'purchased'
y = df['purchased']               # target variable
# 5. Data ko train aur test sets mein split karein (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# 6. Model define aur train karein (Random Forest Classifier)
model = RandomForestClassifier()
model.fit(X_train, y_train)
# 7. Prediction karein aur results evaluate karein
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
# 8. Naye user ke liye prediction karein
new_user = pd.DataFrame([{
    'user_age': 30,
    'user_location': le_location.transform(['California'])[0],
    'product_price': 75.0,
    'time_on_site': 10.5,
    'pages_viewed': 6,
    'previous_purchases': 1,
    'product_category': le_category.transform(['Electronics'])[0],
    'device_type': le_device.transform(['Mobile'])[0],
    'clicked_ad': 1
}])

# 9. Prediction
prediction = model.predict(new_user)
print("✅ Will Purchase" if prediction[0] == 1 else "❌ Will Not Purchase")
