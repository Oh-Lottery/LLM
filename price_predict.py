import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor

def prepare_prediction_features(df):
    """
    Prepare features for predicting next round's 1st prize amount
    """
    # Create lag features
    df['prev_1st_prize'] = df['1st Prize Amount'].shift(1)
    df['prev_winners'] = df['Number of 1st Prize Winner'].shift(1)
    
    # Enhanced rolling statistics
    for window in [3, 5, 10]:
        df[f'rolling_{window}_prize_mean'] = df['1st Prize Amount'].rolling(window=window).mean()
        df[f'rolling_{window}_prize_std'] = df['1st Prize Amount'].rolling(window=window).std()
        df[f'rolling_{window}_winners_mean'] = df['Number of 1st Prize Winner'].rolling(window=window).mean()
    
    # Carryover features
    df['is_carryover'] = (df['Number of 1st Prize Winner'].shift(1) == 0).astype(int)
    df['consecutive_carryover'] = ((df['Number of 1st Prize Winner'].shift(1) == 0) & 
                                 (df['Number of 1st Prize Winner'].shift(2) == 0)).astype(int)
    
    # Prize trend features
    df['prize_momentum'] = df['1st Prize Amount'] - df['1st Prize Amount'].shift(3)
    df['winner_trend'] = df['Number of 1st Prize Winner'] - df['Number of 1st Prize Winner'].shift(3)
    
    # Drop rows with NaN values created by shifting
    df = df.dropna()
    return df

def build_prediction_model(df):
    """
    Build and evaluate a model to predict 1st prize amount
    """
    # Prepare features
    df = prepare_prediction_features(df)
    
    # Define features and target
    features = ['prev_1st_prize', 'prev_winners', 
           'rolling_3_prize_mean', 'rolling_5_prize_mean', 'rolling_10_prize_mean',
           'rolling_3_prize_std', 'rolling_5_prize_std', 'rolling_10_prize_std',
           'rolling_3_winners_mean', 'rolling_5_winners_mean', 'rolling_10_winners_mean',
           'is_carryover', 'consecutive_carryover',
           'prize_momentum', 'winner_trend']
    
    X = df[features]
    y = df['1st Prize Amount']
    
    # Split data
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=21)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    # model = RandomForestRegressor(n_estimators=100, random_state=42)
    model = RandomForestRegressor(n_estimators=300)
    model.fit(X_train_scaled, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test_scaled)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Get feature importance
    feature_importance = dict(zip(features, model.feature_importances_))
    
    return model, scaler, mse, r2, feature_importance, df

def predict_next_prize(model, scaler, df):
    """
    Predict the next round's 1st prize amount using the most recent data
    """
    # Get the last row of data
    features = ['prev_1st_prize', 'prev_winners', 
           'rolling_3_prize_mean', 'rolling_5_prize_mean', 'rolling_10_prize_mean',
           'rolling_3_prize_std', 'rolling_5_prize_std', 'rolling_10_prize_std',
           'rolling_3_winners_mean', 'rolling_5_winners_mean', 'rolling_10_winners_mean',
           'is_carryover', 'consecutive_carryover',
           'prize_momentum', 'winner_trend']
    
    latest_data = df[features].iloc[-1:]
    
    # Transform and predict
    scaled_features = scaler.transform(latest_data)
    prediction = model.predict(scaled_features)
    return prediction[0]

# Load and process data
df = pd.read_csv('oh-lottery.csv')
model, scaler, mse, r2, feature_importance, processed_df = build_prediction_model(df)

# Print model evaluation metrics
print("Model Evaluation Metrics:")
print(f"Mean Squared Error: {mse:,.2f}")
print(f"R² Score: {r2:.4f}")
print("\nFeature Importance:")
for feature, importance in sorted(feature_importance.items(), key=lambda x: x[1], reverse=True):
    print(f"{feature}: {importance:.4f}")

# Make prediction for next round
next_prize_prediction = predict_next_prize(model, scaler, processed_df)
print(f"\nPredicted Next Round 1st Prize Amount: ₩{next_prize_prediction:,.0f}")