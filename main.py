import pandas as pd
import numpy as np
import xgboost as xgb
import optuna
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)


# ==========================================
# 1. DATA LOADING & PREPARATION
# ==========================================
def load_and_prep_data(file_path='Datathon Dataset.xlsx'):
    print(">>> 1. Loading and Cleaning Data...")

    try:
        main_df = pd.read_excel(file_path, sheet_name='Data - Main')
        country_map = pd.read_excel(file_path, sheet_name='Others - Country Mapping')
        cat_link = pd.read_excel(file_path, sheet_name='Others - Category Linkage')
        cash_bal = pd.read_excel(file_path, sheet_name='Data - Cash Balance')
    except FileNotFoundError:
        print(f"Error: '{file_path}' not found. Please ensure the file is in the folder.")
        return None, None

    # Date Conversion
    main_df['Pstng Date'] = pd.to_datetime(main_df['Pstng Date'])

    # Fix Typos
    main_df['Category'] = main_df['Category'].replace({
        'Non Netting AP': 'Non-Netting AP',
        'Non Netting AR': 'Non-Netting AR',
        'Dividend payout': 'Dividend Payout'
    })

    # Merge Mappings
    main_df = pd.merge(main_df, country_map[['Code', 'Country']], left_on='Name', right_on='Code', how='left')

    cat_link = cat_link.rename(columns={'Category Names': 'Cat_Key', 'Category': 'Flow_Type'})
    main_df = pd.merge(main_df, cat_link[['Cat_Key', 'Flow_Type']], left_on='Category', right_on='Cat_Key', how='left')

    # Handle "Other" Logic (Sign based)
    main_df.loc[(main_df['Category'] == 'Other') & (main_df['Amount in USD'] < 0), 'Flow_Type'] = 'Outflow'
    main_df.loc[(main_df['Category'] == 'Other') & (main_df['Amount in USD'] >= 0), 'Flow_Type'] = 'Inflow'

    # Cashflow Bucketing
    operating = ['AP', 'AR', 'Payroll', 'Tax payable', 'Statutory contribution', 'Custom and Duty', 'Netting AP',
                 'Netting AR']
    financing = ['Loan payment', 'Loan receipt', 'Interest charges', 'Interest income', 'Dividend Payout']

    def get_bucket(cat):
        if cat in operating: return 'Operating Activities'
        if cat in financing: return 'Financing Activities'
        return 'Investing/Other'

    main_df['Cashflow_Bucket'] = main_df['Category'].apply(get_bucket)

    return main_df, cash_bal


# ==========================================
# 2. WEEKLY AGGREGATION & FEATURE ENGINEERING
# ==========================================
def create_weekly_features(main_df):
    print(">>> 2. Aggregating to Weekly Data & Creating Features...")

    # Create Week Column (Monday Start)
    main_df['Week_Ending'] = main_df['Pstng Date'].dt.to_period('W-MON').apply(lambda r: r.start_time)

    # Group by Name + Week
    weekly_flow = main_df.groupby(['Name', 'Week_Ending'])['Amount in USD'].sum().reset_index()

    # FILL GAPS: Resample to ensure every week exists
    full_weeks = []
    for name, group in weekly_flow.groupby('Name'):
        group = group.set_index('Week_Ending')
        resampled = group['Amount in USD'].resample('W-MON').sum().fillna(0).reset_index()
        resampled['Name'] = name
        full_weeks.append(resampled)

    df = pd.concat(full_weeks).sort_values(['Name', 'Week_Ending'])
    df = df.rename(columns={'Amount in USD': 'Net_Flow'})

    # --- Feature Engineering ---
    df['Week_of_Year'] = df['Week_Ending'].dt.isocalendar().week.astype(int)
    df['Month'] = df['Week_Ending'].dt.month
    df['Is_Quarter_End'] = df['Month'].isin([3, 6, 9, 12]) & (df['Week_Ending'].dt.day >= 20)

    # Lag Features (History)
    for lag in [1, 4, 12]:
        df[f'Lag_{lag}'] = df.groupby('Name')['Net_Flow'].shift(lag)

    # Volatility Features
    df['Rolling_Mean_4'] = df.groupby('Name')['Net_Flow'].transform(lambda x: x.rolling(4).mean())
    df['Rolling_Std_4'] = df.groupby('Name')['Net_Flow'].transform(lambda x: x.rolling(4).std())

    df_clean = df.dropna().copy()

    return df, df_clean


# ==========================================
# 3. OPTUNA OPTIMIZATION (FIXED)
# ==========================================
def optimize_xgboost(X, y):
    print(">>> 3. Running Optuna Optimization (Finding best parameters)...")

    def objective(trial):
        param = {
            'n_estimators': 1000,
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 9),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'subsample': trial.suggest_float('subsample', 0.6, 0.9),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 0.9),
            'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 1.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 1.0, log=True),
            'n_jobs': -1,
            'random_state': 42
        }

        # Walk-Forward Validation
        tscv = TimeSeriesSplit(n_splits=3)
        scores = []

        for train_idx, val_idx in tscv.split(X):
            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

            # FIX: Pass early_stopping_rounds in the constructor
            model = xgb.XGBRegressor(**param, early_stopping_rounds=50)

            model.fit(
                X_tr, y_tr,
                eval_set=[(X_val, y_val)],
                verbose=False
            )

            preds = model.predict(X_val)
            mae = mean_absolute_error(y_val, preds)
            scores.append(mae)

        return np.mean(scores)

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=15)
    print(f"    Best MAE: {study.best_value:,.2f}")
    print(f"    Best Params: {study.best_params}")
    return study.best_params


# ==========================================
# 4. TRAINING & FORECASTING
# ==========================================
def train_and_forecast(df_clean, full_df, best_params, cash_bal):
    print(">>> 4. Training Final Model & Forecasting Next 6 Months...")

    features = ['Week_of_Year', 'Month', 'Is_Quarter_End', 'Lag_1', 'Lag_4', 'Lag_12', 'Rolling_Mean_4',
                'Rolling_Std_4']
    target = 'Net_Flow'

    X = df_clean[features]
    y = df_clean[target]

    # Train Final Model on ALL data
    # Note: We do NOT use early_stopping here as we want to use the full dataset
    final_model = xgb.XGBRegressor(**best_params, n_estimators=1000, n_jobs=-1, random_state=42)
    final_model.fit(X, y)

    # --- Generate Future Data (Recursive Forecasting) ---
    future_weeks = 26  # 6 Months
    last_date = df_clean['Week_Ending'].max()
    entities = df_clean['Name'].unique()

    forecast_rows = []

    for entity in entities:
        # Get last known data for this entity
        entity_data = full_df[full_df['Name'] == entity].sort_values('Week_Ending')

        for i in range(1, future_weeks + 1):
            next_date = last_date + pd.Timedelta(weeks=i)

            # Update Time Features
            next_row = pd.DataFrame({'Name': [entity], 'Week_Ending': [next_date]})
            next_row['Week_of_Year'] = next_date.isocalendar().week
            next_row['Month'] = next_date.month
            next_row['Is_Quarter_End'] = (next_date.month in [3, 6, 9, 12]) and (next_date.day >= 20)

            # Recalculate Lags (Recursive Step)
            forecast_2d = np.array(forecast_rows).reshape(-1, 11)
            history = pd.concat([entity_data, pd.DataFrame(forecast_2d, columns=entity_data.columns)])
            history = history[history['Name'] == entity].sort_values('Week_Ending')

            # Calculate features dynamically based on updated history
            next_row['Lag_1'] = history['Net_Flow'].iloc[-1]
            next_row['Lag_4'] = history['Net_Flow'].iloc[-4] if len(history) >= 4 else history['Net_Flow'].mean()
            next_row['Lag_12'] = history['Net_Flow'].iloc[-12] if len(history) >= 12 else history['Net_Flow'].mean()
            next_row['Rolling_Mean_4'] = history['Net_Flow'].rolling(4).mean().iloc[-1]
            next_row['Rolling_Std_4'] = history['Net_Flow'].rolling(4).std().iloc[-1]

            # Predict
            pred_flow = final_model.predict(next_row[features])[0]

            # Store Result
            next_row['Net_Flow'] = pred_flow
            forecast_rows.append(next_row)

            # Update entity_data wrapper strictly for next lag calc in loop
            entity_data = pd.concat([entity_data, next_row[['Name', 'Week_Ending', 'Net_Flow']]])

    forecast_df = pd.concat(forecast_rows).reset_index(drop=True)

    # --- Calculate Ending Balance ---
    initial_balances = cash_bal.set_index('Name')['Carryforward Balance (USD)'].to_dict()

    final_results = []
    for entity in entities:
        start_bal = initial_balances.get(entity, 0)

        # Historical Cumulative
        hist_flow = full_df[full_df['Name'] == entity]['Net_Flow'].sum()
        current_balance = start_bal + hist_flow

        # Forecast Cumulative
        ent_forecast = forecast_df[forecast_df['Name'] == entity].copy()
        ent_forecast['Cumulative_Forecast'] = ent_forecast['Net_Flow'].cumsum()
        ent_forecast['Ending_Balance'] = current_balance + ent_forecast['Cumulative_Forecast']

        final_results.append(ent_forecast)

    final_df = pd.concat(final_results)
    return final_df


# ==========================================
# 5. KEY INSIGHTS & REPORTING
# ==========================================
def generate_insights(forecast_df):
    print("\n" + "=" * 40)
    print("      ASTRAZENECA DATATHON - KEY INSIGHTS")
    print("=" * 40)

    # 1. Cash Drainers
    total_flow = forecast_df.groupby('Name')['Net_Flow'].sum().sort_values()
    print("\n[ALERT] Top 3 Entities with Highest Cash Burn (Next 6 Months):")
    for name, val in total_flow.head(3).items():
        print(f"  - {name}: ${val:,.0f}")

    # 2. Liquidity Risk
    min_balances = forecast_df.groupby('Name')['Ending_Balance'].min()
    negative_bal = min_balances[min_balances < 0]
    if not negative_bal.empty:
        print("\n[CRITICAL] Entities Projected to Hit Negative Balance:")
        for name, val in negative_bal.items():
            print(f"  - {name}: Hits low of ${val:,.0f}")
    else:
        print("\n[INFO] No entities projected to go into overdraft.")

    # 3. Volatility Watch
    volatility = forecast_df.groupby('Name')['Net_Flow'].std().sort_values(ascending=False)
    print("\n[WATCH] Most Volatile Entities (Hardest to Predict):")
    for name, val in volatility.head(3).items():
        print(f"  - {name}: Weekly Swing ~${val:,.0f}")

    # 4. Global Stress Week
    global_weekly = forecast_df.groupby('Week_Ending')['Net_Flow'].sum()
    worst_week = global_weekly.idxmin()
    print(f"\n[PLANNING] Worst Global Cash Flow Week: {worst_week.date()}")
    print(f"  - Total Net Outflow: ${global_weekly.min():,.0f}")
    print("=" * 40)


# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    main_df, cash_bal = load_and_prep_data()

    if main_df is not None:
        full_df, df_clean = create_weekly_features(main_df)

        features = ['Week_of_Year', 'Month', 'Is_Quarter_End', 'Lag_1', 'Lag_4', 'Lag_12', 'Rolling_Mean_4',
                    'Rolling_Std_4']
        X = df_clean[features]
        y = df_clean['Net_Flow']

        best_params = optimize_xgboost(X, y)

        forecast_results = train_and_forecast(df_clean, full_df, best_params, cash_bal)

        forecast_results.to_csv("Final_AZ_Forecast.csv", index=False)
        generate_insights(forecast_results)
        print("\n>>> Success! Forecast saved to 'Final_AZ_Forecast.csv'")

