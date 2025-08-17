import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import os

from config import project_config

# --- Default Global Parameters ---
# These can be overridden by passing them as arguments to the main function
# DEFAULT_CONFIG = {
#     "BASE_DIR": ".",
#     "START_DATE": datetime(2025, 1, 1),
#     "TRAIN_MONTHS": 4,
#     "VALIDATION_MONTHS": 1,
#     "TEST_MONTHS": 1,
#     "NUM_USERS": 500,
#     "USER_PERSONAS": {
#         'casual': {'id_range': (1, 400), 'requests_per_day': 50, 'prompt_tokens': (50, 20), 'completion_tokens': (100, 40)},
#         'power': {'id_range': (401, 500), 'requests_per_day': 400, 'prompt_tokens': (200, 80), 'completion_tokens': (500, 150)}
#     },
#     "LATENCY_NORMAL": (250, 50),
#     "ERROR_RATE_NORMAL": 0.01,
#     "LATENCY_DRIFT_PER_DAY": 0.1,
#     "TOKEN_DRIFT_PER_DAY": 0.05,
#     "ABUSIVE_USER_ID": 450,
#     "ABUSIVE_TOKEN_MULTIPLIER": 15,
#     "LATENCY_SPIKE_MULTIPLIER": 4,
#     "DOS_ATTACK_MULTIPLIER": 50
# }

# --- Helper Functions ---
def _get_user_persona(user_id, config):
    """Internal helper to find a user's persona."""
    for persona, details in config["USER_PERSONAS"].items():
        if details['id_range'][0] <= user_id <= details['id_range'][1]:
            return details
    return config["USER_PERSONAS"]['casual']

def _generate_production_data(start_date, num_days, config):
    """Internal helper to generate a chunk of time-series data."""
    logs = []
    current_time = start_date
    end_date = start_date + timedelta(days=num_days)
    while current_time < end_date:
        days_passed = (current_time - config["START_DATE"]).days
        hour_of_day = current_time.hour
        diurnal_multiplier = (np.sin((hour_of_day - 8) * (2 * np.pi / 24)) + 1.1) 
        weekly_multiplier = 0.6 if current_time.weekday() >= 5 else 1.0
        activity_level = diurnal_multiplier * weekly_multiplier
        total_base_requests = sum(p['requests_per_day'] * (p['id_range'][1] - p['id_range'][0] + 1) for p in config["USER_PERSONAS"].values())
        num_requests_this_minute = np.random.poisson((total_base_requests / (24 * 60)) * activity_level)
        current_latency_drift = days_passed * config["LATENCY_DRIFT_PER_DAY"]
        current_token_drift = days_passed * config["TOKEN_DRIFT_PER_DAY"]
        for _ in range(num_requests_this_minute):
            user_id = np.random.randint(1, config["NUM_USERS"] + 1)
            persona = _get_user_persona(user_id, config)
            latency = np.random.normal(config["LATENCY_NORMAL"][0] + current_latency_drift, config["LATENCY_NORMAL"][1])
            prompt_tokens = np.maximum(10, np.random.normal(persona['prompt_tokens'][0] + current_token_drift, persona['prompt_tokens'][1]))
            completion_tokens = np.maximum(10, np.random.normal(persona['completion_tokens'][0] + current_token_drift, persona['completion_tokens'][1]))
            status_code = 200 if np.random.rand() > config["ERROR_RATE_NORMAL"] else 500
            logs.append([current_time, user_id, latency, prompt_tokens, completion_tokens, status_code])
        current_time += timedelta(minutes=1)
    df = pd.DataFrame(logs, columns=['timestamp', 'user_id', 'latency_ms', 'prompt_tokens', 'completion_tokens', 'status_code'])
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df

def _inject_anomalies(df, config):
    """Internal helper to inject anomalies into a dataframe."""
    anomaly_labels = []
    test_df = df.copy()
    start_time_1 = test_df['timestamp'].min() + timedelta(days=5, hours=2); end_time_1 = start_time_1 + timedelta(hours=3)
    anomaly_labels.append({'start': start_time_1, 'end': end_time_1, 'type': 'Token Abuse'})
    abuse_mask = (test_df['timestamp'] >= start_time_1) & (test_df['timestamp'] < end_time_1) & (test_df['user_id'] == config["ABUSIVE_USER_ID"])
    test_df.loc[abuse_mask, 'prompt_tokens'] *= config["ABUSIVE_TOKEN_MULTIPLIER"]; test_df.loc[abuse_mask, 'completion_tokens'] *= config["ABUSIVE_TOKEN_MULTIPLIER"]
    start_time_2 = test_df['timestamp'].min() + timedelta(days=12, hours=8); end_time_2 = start_time_2 + timedelta(hours=4)
    anomaly_labels.append({'start': start_time_2, 'end': end_time_2, 'type': 'Latency Degradation'})
    latency_mask = (test_df['timestamp'] >= start_time_2) & (test_df['timestamp'] < end_time_2)
    test_df.loc[latency_mask, 'latency_ms'] *= config["LATENCY_SPIKE_MULTIPLIER"]
    start_time_3 = test_df['timestamp'].min() + timedelta(days=20, hours=14); end_time_3 = start_time_3 + timedelta(minutes=30)
    anomaly_labels.append({'start': start_time_3, 'end': end_time_3, 'type': 'DoS Attack'})
    attacker_logs = []; attack_time = start_time_3; attacker_id = 999
    while attack_time < end_time_3:
        num_attack_requests = np.random.poisson((config["USER_PERSONAS"]['casual']['requests_per_day'] / (24*60)) * config["DOS_ATTACK_MULTIPLIER"])
        for _ in range(num_attack_requests):
            attacker_logs.append([attack_time, attacker_id, np.random.normal(*config["LATENCY_NORMAL"]), 10, 10, 503])
        attack_time += timedelta(minutes=1)
    attacker_df = pd.DataFrame(attacker_logs, columns=test_df.columns)
    test_df = pd.concat([test_df, attacker_df]).sort_values('timestamp').reset_index(drop=True)
    return test_df, pd.DataFrame(anomaly_labels)

# --- Main Orchestration Function ---
def create_synthetic_dataset(config=None):
    """
    Generates and saves a complete synthetic dataset for the LLM Ops anomaly detection project.
    
    This function creates and saves three datasets (train, validation, test) and a
    file with the ground-truth anomaly labels for the test set.
    
    Args:
        config (dict, optional): A dictionary of parameters to override the defaults.
                                 If None, DEFAULT_CONFIG is used.
    """
    DEFAULT_CONFIG = {
    # --- Project and Time Settings ---
    "BASE_DIR": ".",
    "START_DATE": datetime(2024, 1, 1),
    "TRAIN_MONTHS": 9,
    "VALIDATION_MONTHS": 1,
    "TEST_MONTHS": 2,

    # --- User Population Settings ---
    "NUM_USERS": 5000,
    "USER_PERSONAS": {
        'casual': {'id_range': (1, 4000), 'requests_per_day': 50, 'prompt_tokens': (50, 20), 'completion_tokens': (100, 40)},
        'power':  {'id_range': (4001, 5000), 'requests_per_day': 400, 'prompt_tokens': (200, 80), 'completion_tokens': (500, 150)}
    },

    # --- System Behavior Settings ---
    "LATENCY_NORMAL": (250, 50),
    "ERROR_RATE_NORMAL": 0.01,
    "LATENCY_DRIFT_PER_DAY": 0.1,
    "TOKEN_DRIFT_PER_DAY": 0.05,

    # --- Anomaly Injection Settings ---
    "ABUSIVE_USER_ID": 4500,  # A power user
    "ABUSIVE_TOKEN_MULTIPLIER": 15,
    "LATENCY_SPIKE_MULTIPLIER": 4,
    "DOS_ATTACK_MULTIPLIER": 50
    }

    if config is None:
        config = DEFAULT_CONFIG

    # --- Setup Directories and Durations ---
    RAW_DATA_DIR = project_config.RAW_DATA_DIR
    PROCESSED_DATA_DIR = project_config.PROCESSED_DATA_DIR
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    
    START_DATE = config["START_DATE"]
    TRAIN_DAYS = config["TRAIN_MONTHS"] * 30
    VALIDATION_DAYS = config["VALIDATION_MONTHS"] * 30
    TEST_DAYS = config["TEST_MONTHS"] * 30
    
    # --- Generate All Three Datasets Chronologically ---
    print(f"--- Starting Dataset Generation ---")
    
    # 1. Training Set
    print(f"Generating {config['TRAIN_MONTHS']} months of training data...")
    train_df = _generate_production_data(START_DATE, TRAIN_DAYS, config)
    print(f"  - Generated {len(train_df):,} log entries.")

    # 2. Validation Set
    validation_start_date = START_DATE + timedelta(days=TRAIN_DAYS)
    print(f"\nGenerating {config['VALIDATION_MONTHS']} month of validation data...")
    validation_df = _generate_production_data(validation_start_date, VALIDATION_DAYS, config)
    print(f"  - Generated {len(validation_df):,} log entries.")

    # 3. Test Set
    test_start_date = validation_start_date + timedelta(days=VALIDATION_DAYS)
    print(f"\nGenerating {config['TEST_MONTHS']} month of test data...")
    base_test_df = _generate_production_data(test_start_date, TEST_DAYS, config)
    test_df, anomaly_labels_df = _inject_anomalies(base_test_df, config)
    print(f"  - Generated {len(test_df):,} log entries (after injecting anomalies).")

    # --- Save all datasets ---
    print("\nSaving all datasets to CSV files...")
    train_df.to_csv(os.path.join(RAW_DATA_DIR, 'llm_logs_train.csv'), index=False)
    validation_df.to_csv(os.path.join(RAW_DATA_DIR, 'llm_logs_val.csv'), index=False)
    test_df.to_csv(os.path.join(RAW_DATA_DIR, 'llm_logs_test.csv'), index=False)
    anomaly_labels_df.to_csv(os.path.join(RAW_DATA_DIR, 'llm_anomaly_labels.csv'), index=False)

    print(f"\n✅ Dataset generation complete. All files saved in '{RAW_DATA_DIR}'.")