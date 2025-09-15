# =============================================================================
# Airbnb Price Prediction Model
# Complete pipeline from data cleaning to model training
# =============================================================================

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Load data (replace with your file path)
df = pd.read_csv('AB_NYC_2019.csv')  # Update with your actual file name

# =============================================================================
# DATA CLEANING & PREPROCESSING
# =============================================================================

# Handle missing values
df['host_name'].fillna('Unknown_user', inplace=True)
df['name'].fillna('Unknown_user', inplace=True)
df['reviews_per_month'].fillna(0, inplace=True)

# Convert last_review to datetime and create days_since_last_review
df['last_review'] = pd.to_datetime(df['last_review'])
df['days_since_last_review'] = (pd.Timestamp('today') - df['last_review']).dt.days.fillna(9999)

# =============================================================================
# FEATURE ENGINEERING
# =============================================================================

from sklearn.preprocessing import LabelEncoder

# Encode categorical variables
le = LabelEncoder()
df['neighbourhood_group_encoded'] = le.fit_transform(df['neighbourhood_group'])
df['room_type_encoded'] = le.fit_transform(df['room_type'])
df['neighbourhood_encoded'] = le.fit_transform(df['neighbourhood'])

# Create new features
df['price_per_night'] = df['price'] / df['minimum_nights'].replace(0, 1)
df['occupancy_rate'] = (365 - df['availability_365']) / 365
df['total_host_experience'] = df['number_of_reviews'] * df['reviews_per_month'].replace(0, 1)
df['recently_reviewed'] = (df['days_since_last_review'] < 30).astype(int)
df['review_active'] = (df['days_since_last_review'] < 90).astype(int)

# =============================================================================
# FEATURE SELECTION & OUTLIER REMOVAL
# =============================================================================

# Define features
features = [
    'neighbourhood_group_encoded', 'neighbourhood_encoded', 'room_type_encoded',
    'latitude', 'longitude', 'minimum_nights', 'number_of_reviews',
    'reviews_per_month', 'calculated_host_listings_count', 'availability_365',
    'total_host_experience', 'price_per_night', 'occupancy_rate',
    'recently_reviewed', 'review_active'
]

# Create X and y
X = df[features]
y = df['price']

# Handle NaN values
X = X.fillna(X.median())
y = y.fillna(y.median())

# Remove outliers using IQR method
Q1 = y.quantile(0.25)
Q3 = y.quantile(0.75)
IQR = Q3 - Q1
y_clean = y[(y >= (Q1 - 1.5 * IQR)) & (y <= (Q3 + 1.5 * IQR))]
X_clean = X.loc[y_clean.index]

# =============================================================================
# TRAIN-TEST SPLIT & SCALING
# =============================================================================

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X_clean, y_clean, test_size=0.2, random_state=42
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# =============================================================================
# MODEL TRAINING & EVALUATION
# =============================================================================

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns

# Initialize models
models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(alpha=1.0),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(random_state=42),
    'SVR': SVR(kernel='rbf')
}

# Train and evaluate models
results = {}
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    results[name] = {
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2
    }
    
    print(f"{name}: RMSE = {rmse:.2f}, MAE = {mae:.2f}, R2 = {r2:.4f}")

# Convert results to DataFrame
results_df = pd.DataFrame(results).T
print("\nModel Comparison:")
print(results_df.sort_values('R2', ascending=False))

# =============================================================================
# HYPERPARAMETER TUNING (RANDOM FOREST)
# =============================================================================

from sklearn.model_selection import GridSearchCV

# Tune Random Forest (typically performs well)
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

rf = RandomForestRegressor(random_state=42)
grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=5,
    scoring='neg_mean_squared_error',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train_scaled, y_train)

# Best parameters
print("Best parameters:", grid_search.best_params_)
print("Best score:", np.sqrt(-grid_search.best_score_))

# Train with best parameters
best_rf = grid_search.best_estimator_

# =============================================================================
# FEATURE IMPORTANCE
# =============================================================================

# Get feature importance
feature_importance = best_rf.feature_importances_

# Create DataFrame for visualization
importance_df = pd.DataFrame({
    'feature': features,
    'importance': feature_importance
}).sort_values('importance', ascending=False)

# Plot feature importance
plt.figure(figsize=(10, 6))
sns.barplot(x='importance', y='feature', data=importance_df)
plt.title('Feature Importance for Price Prediction')
plt.tight_layout()
plt.savefig('feature_importance.png')  # Save for GitHub
plt.show()

# =============================================================================
# SAVE MODEL & PREPROCESSING ARTIFACTS
# =============================================================================

import joblib

# Save the best model and preprocessing objects
joblib.dump(best_rf, 'airbnb_price_predictor.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(le, 'label_encoder.pkl')

print("Model and preprocessing objects saved successfully!")

# =============================================================================
# CREATE PREDICTION FUNCTION
# =============================================================================

def predict_price(new_listing):
    """
    Predict price for a new Airbnb listing
    new_listing: Dictionary with feature values
    """
    # Load artifacts
    model = joblib.load('airbnb_price_predictor.pkl')
    scaler = joblib.load('scaler.pkl')
    
    # Convert to DataFrame
    listing_df = pd.DataFrame([new_listing])
    
    # Apply same preprocessing
    listing_scaled = scaler.transform(listing_df)
    
    # Make prediction
    prediction = model.predict(listing_scaled)
    
    return prediction[0]

# Example usage
example_listing = {
    'neighbourhood_group_encoded': 1,
    'neighbourhood_encoded': 25,
    'room_type_encoded': 2,
    'latitude': 40.7128,
    'longitude': -74.0060,
    'minimum_nights': 2,
    'number_of_reviews': 15,
    'reviews_per_month': 1.5,
    'calculated_host_listings_count': 3,
    'availability_365': 200,
    'total_host_experience': 22.5,
    'price_per_night': 0,  # This will be calculated
    'occupancy_rate': 0.45,
    'recently_reviewed': 1,
    'review_active': 1
}

# Note: For a real application, you'd need to preprocess the input data
# similar to how we processed the training data

print("\nModel training completed successfully!")