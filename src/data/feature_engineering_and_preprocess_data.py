import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib


from config import project_config

def create_time_series_features(df, freq='5min'):
    """
    Aggregates raw log data into a multivariate time series.
    """
    print(f"Resampling data to {freq} frequency...")
    
    df = df.set_index('timestamp')
    df['is_error'] = (df['status_code'] != 200).astype(int)
    
    # Define aggregations, including 'size' to get the request count
    aggs = {
        'latency_ms': ['mean', 'std', 'max'],
        'prompt_tokens': ['mean', 'std', 'sum'],
        'completion_tokens': ['mean', 'std', 'sum'],
        'user_id': ['nunique'],
        'is_error': ['sum'],
        'status_code': ['size'] # Use 'size' on any non-null column to get the count
    }
    
    # **FIX:** Apply aggregation *after* resampling
    df_resampled = df.resample(freq).agg(aggs)
    
    # Flatten the multi-level column names
    df_resampled.columns = ['_'.join(col).strip() for col in df_resampled.columns.values]
    
    # **FIX:** Rename the 'size' column to 'request_count' for clarity
    df_resampled.rename(columns={'status_code_size': 'request_count'}, inplace=True)
    
    # --- Create Ratio Features ---
    # To avoid division by zero, we add a small epsilon
    epsilon = 1e-6
    # This line will now work correctly
    df_resampled['error_rate'] = df_resampled['is_error_sum'] / (df_resampled['request_count'] + epsilon)
    
    # Fill any potential NaN values that result from empty windows
    df_resampled.fillna(0, inplace=True)
    
    return df_resampled


def start_feature_engineering_and_data_preprocessing():

    RAW_DATA_DIR = project_config.RAW_DATA_DIR
    INTERIM_DATA_DIR = project_config.INTERIM_DATA_DIR
    os.makedirs(INTERIM_DATA_DIR, exist_ok=True)

    # anomaly_labels_df.to_csv(os.path.join(RAW_DATA_DIR, 'llm_anomaly_labels.csv'), index=False)
    # --- Load the Datasets ---
    print("Loading train, validation, test datasets...")
    # DATA_DIR = "llm_ops_data"
    train_df = pd.read_csv(os.path.join(RAW_DATA_DIR, 'llm_logs_train.csv'), parse_dates=['timestamp'])
    val_df = pd.read_csv(os.path.join(RAW_DATA_DIR, 'llm_logs_val.csv'), parse_dates=['timestamp'])
    test_df = pd.read_csv(os.path.join(RAW_DATA_DIR, 'llm_logs_test.csv'), parse_dates=['timestamp'])

    # --- Apply the 5 min frequency create_time_series_features function to our datasets ---
    train_ts = create_time_series_features(train_df)
    val_ts = create_time_series_features(val_df)
    test_ts = create_time_series_features(test_df)

    print("\n--- Engineered Features (Training Set Head) ---")
    print(train_ts.head())
    print(f"\nShape of training time series: {train_ts.shape}")

    print("\n--- Engineered Features (Validation Set Head) ---")
    print(val_ts.head())
    print("\nShape of validation time series: ", val_ts.shape)

    print("\n--- Engineered Features (Testing Set Head) ---")
    print(test_ts.head())
    print("\nShape of testing time series: ", test_ts.shape)


    # --- Scale the Features ---
    scaler = StandardScaler()
    print("\nFitting scaler on training data...")
    scaler.fit(train_ts)
    '''
    Why are we fitting the scaler only on the training data?
    This is not a mistake; it is a deliberate and critical step to ensure the integrity of the entire project.
    The Analogy: Creating a Ruler
    Think of the StandardScaler as a special ruler you are building to measure your data.

    scaler.fit(train_ts): Building the Ruler
    What it does: When you call .fit(), the scaler looks at all the data in train_ts and calculates two crucial numbers for each feature (each column): its mean (the average value) and its standard deviation (a measure of how spread out the data is).
    The Purpose: These two numbers now define your "ruler." The mean becomes the "zero point" of your ruler, and the standard deviation becomes the "unit markings" (like inches or centimeters). This ruler represents the definition of "normal" because it was built exclusively from your 100% normal training data.

    scaler.transform(...): Using the Ruler
    What it does: When you call .transform(), the scaler takes a dataset and uses the ruler it has already built. For each data point, it subtracts the mean it learned from the training set and divides by the standard deviation it learned from the training set.
    The Purpose: This ensures that all of your data—training, validation, and testing—is measured with the exact same, consistent ruler.
    Why We ONLY Fit on the Training Data
    This is the most important concept. Your test_ts (and validation_ts) datasets are like a final exam for your model. They must remain completely unseen and unknown during the "learning" phase.
    '''
    print("Transforming training and testing data...")
    train_scaled = scaler.transform(train_ts)
    val_scaled = scaler.transform(val_ts)  # Scale validation set as well
    test_scaled = scaler.transform(test_ts)

    print("\n--- Scaled Data ---")
    print(f"Shape of scaled training data: {train_scaled.shape}")
    print("Sample of scaled training data (first 5 rows):")
    print(train_scaled[:5])

    # Save all necessary artifacts
    # --- Save the processed arrays for the next phase (This part remains the same) ---
    train_scaled_path = os.path.join(INTERIM_DATA_DIR, 'train_scaled.npy')
    val_scaled_path = os.path.join(INTERIM_DATA_DIR, 'val_scaled.npy')
    test_scaled_path = os.path.join(INTERIM_DATA_DIR, 'test_scaled.npy')
    # Also save the scaler itself - this is a best practice for production
    scaler_path = os.path.join(INTERIM_DATA_DIR, 'scaler.joblib')

    # --- CRITICAL ADDITION ---
    # Save the feature names to a file for the final analysis script
    feature_names = train_ts.columns.tolist()
    feature_names_path = os.path.join(INTERIM_DATA_DIR, 'feature_names.joblib')
    joblib.dump(feature_names, feature_names_path)
    # --- END ADDITION ---

    np.save(train_scaled_path, train_scaled)
    np.save(val_scaled_path, val_scaled)  # Save validation set as well
    np.save(test_scaled_path, test_scaled)
    joblib.dump(scaler, scaler_path)

    print(f"\nProcessed and scaled data, and the scaler object, have been saved to '{INTERIM_DATA_DIR}'")


