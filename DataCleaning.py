import pandas as pd

# Load the dataset
df = pd.read_csv("used_cars_dirty_dataset.csv")

# Initial exploration
print(df.shape)           # Display initial shape of the dataset
print(df.columns)         # Display column names
df.info()                 # Display data types and non-null counts
df.head()                # Preview the first few rows

# Check for missing values
df.isnull().sum().sort_values(ascending=False)

# Remove duplicate rows
df = df.drop_duplicates()
print("After removing duplicates:", df.shape)

# Convert relevant columns to numeric, coercing errors to NaN
df['price'] = pd.to_numeric(df['price'], errors='coerce')
df['odometer'] = pd.to_numeric(df['odometer'], errors='coerce')
df['year'] = pd.to_numeric(df['year'], errors='coerce')

# Identify and remove suspiciously cheap and expensive entries
df[df['price'] < 500]         # View suspiciously cheap cars
df[df['price'] > 100000]      # View suspiciously expensive cars
df = df[(df['price'] >= 500) & (df['price'] <= 100000)]

# Remove cars with unrealistic odometer readings
df = df[df['odometer'] <= 300000]

# Summary statistics for price and odometer
print(df['price'].describe())
print(df['odometer'].describe())

# Clean text columns by making them lowercase and stripping whitespace
df['manufacturer'] = df['manufacturer'].str.lower().str.strip()
df['model'] = df['model'].str.lower().str.strip()
df['condition'] = df['condition'].str.lower().str.strip()

# Fill missing values in categorical columns with mode
df['fuel'] = df['fuel'].fillna(df['fuel'].mode()[0])
df['transmission'] = df['transmission'].fillna(df['transmission'].mode()[0])

# Drop rows with missing essential numeric values
df = df.dropna(subset=['year', 'price', 'odometer'])

# Final check for remaining missing values
df.isnull().sum()

# Export the cleaned dataset to a new CSV file (without index)
df.to_csv("used_cars_cleaned.csv", index=False)

# Preview the cleaned dataset
df.head()
