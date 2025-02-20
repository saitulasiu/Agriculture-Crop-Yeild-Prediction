import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler

# Sample dataset (replace with actual dataset)
data = {
    'Temperature': [25, 30, 28, 32, 26, 29, 33, 27, 31, 30, 24, 35, 22, 36, 23],
    'Rainfall': [200, 150, 180, 120, 220, 140, 100, 210, 130, 160, 250, 90, 270, 80, 260],
    'Soil Quality': [7, 6.5, 7.2, 6.8, 7.5, 6.3, 6.9, 7.1, 6.7, 7.3, 7.8, 6.1, 8.0, 5.9, 8.2],
    'Crop Yield': [3000, 2500, 2800, 2200, 3200, 2400, 2100, 3100, 2300, 2600, 3400, 2000, 3500, 1900, 3600]
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Data visualization
sns.pairplot(df)
plt.show()

# Correlation heatmap
plt.figure(figsize=(8,6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.show()

# Features and target
X = df[['Temperature', 'Rainfall', 'Soil Quality']]
y = df['Crop Yield']

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train a Random Forest model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Model evaluation
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error: {mae}")
print(f"Mean Squared Error: {mse}")
print(f"Root Mean Squared Error: {rmse}")
print(f"R2 Score: {r2}")

# Feature importance
feature_importances = model.feature_importances_
feature_names = X.columns

plt.figure(figsize=(8,6))
sns.barplot(x=feature_importances, y=feature_names, palette='viridis')
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Feature Importance in Crop Yield Prediction")
plt.show()
