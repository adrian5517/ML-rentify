import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
import logging

logging.basicConfig(level=logging.INFO)

def remove_outliers(df, columns, method='iqr', threshold=1.5):
    """
    Remove outliers using IQR (Interquartile Range) method
    
    Args:
        df: DataFrame
        columns: list of columns to check for outliers
        method: 'iqr' for IQR method
        threshold: IQR multiplier (default 1.5)
    
    Returns:
        DataFrame without outliers
    """
    df_clean = df.copy()
    initial_count = len(df_clean)
    
    for col in columns:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        
        # Filter outliers
        df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
    
    removed = initial_count - len(df_clean)
    if removed > 0:
        logging.info(f"🧹 Removed {removed} outliers from {initial_count} records ({removed/initial_count*100:.1f}%)")
    
    return df_clean

def preprocess_data(df, scaler=None, fit=False):
    """
    Preprocess property data with improved cleaning and scaling
    
    Args:
        df: raw DataFrame from MongoDB
        scaler: existing scaler (optional, for consistency)
        fit: whether to fit a new scaler (True) or use existing (False)
    
    Returns:
        scaled_data: numpy array of scaled features
        scaler: fitted scaler object
        df_clean: cleaned DataFrame
    """
    # Drop rows with missing critical fields
    df = df.dropna(subset=['price', 'location'])
    
    # Extract location coordinates
    df['latitude'] = df['location'].apply(lambda loc: loc.get('latitude') if isinstance(loc, dict) else None)
    df['longitude'] = df['location'].apply(lambda loc: loc.get('longitude') if isinstance(loc, dict) else None)
    
    # Drop rows with missing coordinates
    df = df.dropna(subset=['latitude', 'longitude'])
    
    # Select features for ML
    df_features = df[['price', 'latitude', 'longitude']].copy()
    
    # Convert to numeric and handle any remaining issues
    df_features['price'] = pd.to_numeric(df_features['price'], errors='coerce')
    df_features['latitude'] = pd.to_numeric(df_features['latitude'], errors='coerce')
    df_features['longitude'] = pd.to_numeric(df_features['longitude'], errors='coerce')
    
    # Drop any rows that couldn't be converted
    df_features = df_features.dropna()
    
    # Remove outliers (especially important for price)
    df_features = remove_outliers(df_features, ['price'], threshold=1.5)
    
    if len(df_features) == 0:
        logging.warning("⚠️ No valid data after preprocessing")
        return np.array([]), scaler, pd.DataFrame()
    
    # Scale features for better ML performance
    # RobustScaler is less sensitive to outliers than StandardScaler
    if fit or scaler is None:
        scaler = RobustScaler()
        scaled_data = scaler.fit_transform(df_features)
        logging.info(f"✅ Fitted new scaler on {len(df_features)} samples")
        # Defensive: ensure scaler knows the feature names, useful if sklearn behavior differs by version
        try:
            scaler.feature_names_in_ = df_features.columns.to_numpy()
        except Exception:
            # not critical — sklearn usually sets this automatically when fitting on a DataFrame
            logging.debug("Could not set scaler.feature_names_in_, proceeding anyway")
    else:
        scaled_data = scaler.transform(df_features)
        logging.info(f"✅ Applied existing scaler on {len(df_features)} samples")
    
    logging.info(f"📊 Feature ranges: Price [{df_features['price'].min():.0f} - {df_features['price'].max():.0f}]")
    
    return scaled_data, scaler, df_features
