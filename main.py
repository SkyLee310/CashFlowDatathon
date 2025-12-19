import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
import optuna
from sklearn.ensemble import IsolationForest
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import LabelEncoder
import warnings

# set up and branding
warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

# AZ Brand Colour
az_colors = {
    'Primary': '#830051',  # Mulberry
    'Accent': '#C4D600',  # Lime Green
    'Interaction': '#D0006F',  # Magenta
    'Supporting': '#F0AB00',  # Gold
    'Grey': '#333333',  # Dark Grey
    'LightGrey': '#E0E0E0'  # Light Grey
}

# Plotting style
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['axes.prop_cycle'] = plt.cycler(
    color=[az_colors['Primary'], az_colors['Accent'], az_colors['Supporting'], az_colors['Interaction']])
plt.rcParams['text.color'] = az_colors['Grey']
plt.rcParams['axes.labelcolor'] = az_colors['Grey']
plt.rcParams['xtick.color'] = az_colors['Grey']
plt.rcParams['ytick.color'] = az_colors['Grey']
plt.rcParams['grid.color'] = az_colors['LightGrey']

# data loading and cleaning
file_path = "Datathon Dataset.xlsx"
print(">>> Processing data...")

main_df = pd.read_excel(file_path, sheet_name='Data - Main')
cash_balance_df = pd.read_excel(file_path, sheet_name='Data - Cash Balance')
country_mapping_df = pd.read_excel(file_path, sheet_name='Others - Country Mapping')
category_linkage_df = pd.read_excel(file_path, sheet_name='Others - Category Linkage')


main_df['Pstng Date'] = pd.to_datetime(main_df['Pstng Date'])
main_df = pd.merge(main_df, country_mapping_df[['Code', 'Country', 'Currency']],
                   left_on='Name', right_on='Code', how='left')

# 类别清洗
fix_map = {'Non Netting AP': 'Non-Netting AP', 'Non Netting AR': 'Non-Netting AR', 'Dividend payout': 'Dividend Payout'}
main_df['Category'] = main_df['Category'].replace(fix_map)

category_linkage_clean = category_linkage_df.rename(columns={'Category Names': 'Category_Key', 'Category': 'Flow_Type'})
main_df = pd.merge(main_df, category_linkage_clean[['Category_Key', 'Flow_Type']],
                   left_on='Category', right_on='Category_Key', how='left')

# Flow Type
main_df['Flow_Type'] = main_df['Flow_Type'].fillna('Other')
main_df.loc[(main_df['Flow_Type'] == 'Other') & (main_df['Amount in USD'] < 0), 'Flow_Type'] = 'Outflow'
main_df.loc[(main_df['Flow_Type'] == 'Other') & (main_df['Amount in USD'] >= 0), 'Flow_Type'] = 'Inflow'
main_df = main_df.drop(columns=['Code', 'Category_Key'])

# export the cleaned data
cleaned_file = "Processed_Cleaned_Data.xlsx"
main_df.to_excel(cleaned_file, index=False)
print(f"cleaned data exported: {cleaned_file}")

# fraud detection
print("\n>>> fraud detecting...")
df_fraud = main_df.copy()
le = LabelEncoder()

# feature preparation
df_fraud['Name_Encoded'] = le.fit_transform(df_fraud['Name'])
df_fraud['Flow_Type_Encoded'] = le.fit_transform(df_fraud['Flow_Type'])
df_fraud['Week_Day'] = df_fraud['Pstng Date'].dt.dayofweek

# A. Isolation Forest
features_fraud = ['Amount in USD', 'Name_Encoded', 'Flow_Type_Encoded', 'Week_Day']
iso_forest = IsolationForest(n_estimators=100, contamination=0.01, random_state=42)
df_fraud['Anomaly_Score'] = iso_forest.fit_predict(df_fraud[features_fraud])
ml_anomalies = df_fraud[df_fraud['Anomaly_Score'] == -1]


# B. Benford's Law
def get_first_digit(x):
    try:
        return int(str(abs(int(x)))[0])
    except:
        return 0


benford_df = df_fraud[df_fraud['Amount in USD'] != 0].copy()
benford_df['First_Digit'] = benford_df['Amount in USD'].apply(get_first_digit)
benford_df = benford_df[benford_df['First_Digit'] > 0]
actual_counts = benford_df['First_Digit'].value_counts(normalize=True).sort_index()
benford_probs = np.array([np.log10(1 + 1 / d) for d in range(1, 10)])

# C. 周末大额规则
df_fraud['Is_Weekend'] = df_fraud['Pstng Date'].dt.dayofweek >= 5
rule_anomalies = df_fraud[(df_fraud['Is_Weekend'] == True) &
                          (df_fraud['Flow_Type'] == 'Outflow') &
                          (df_fraud['Amount in USD'].abs() > df_fraud['Amount in USD'].abs().quantile(0.95))]

# --- 绘制欺诈仪表盘 ---
plt.figure(figsize=(14, 10))

# Subplot 1: ML Anomalies
plt.subplot(2, 2, 1)
plt.scatter(df_fraud[df_fraud['Anomaly_Score'] == 1]['Pstng Date'],
            df_fraud[df_fraud['Anomaly_Score'] == 1]['Amount in USD'],
            c=az_colors['Grey'], s=10, alpha=0.1, label='Normal')
plt.scatter(ml_anomalies['Pstng Date'], ml_anomalies['Amount in USD'],
            c=az_colors['Interaction'], s=50, marker='D', label='ML Anomaly')
plt.title('Isolation Forest Anomalies', fontweight='bold', color=az_colors['Primary'])
plt.legend()

# Subplot 2: Benford's Law
plt.subplot(2, 2, 2)
digits = range(1, 10)
plt.bar(digits, actual_counts, alpha=0.7, label='Actual Data', color=az_colors['Primary'])
plt.plot(digits, benford_probs, color=az_colors['Interaction'], marker='o', linestyle='--', linewidth=2,
         label='Benford Law')
plt.title("Benford's Law Analysis", fontweight='bold', color=az_colors['Primary'])
plt.legend()

# Subplot 3: Top Categories
plt.subplot(2, 2, 3)
if not ml_anomalies.empty:
    top_fraud = ml_anomalies['Category'].value_counts().head(5)
    sns.barplot(y=top_fraud.index, x=top_fraud.values, color=az_colors['Primary'])
    plt.title('Top Categories in Anomalies', fontweight='bold', color=az_colors['Primary'])

plt.tight_layout()
plt.savefig('fraud_dashboard_branded.png')
print("已保存欺诈仪表盘: fraud_dashboard_branded.png")

# ==========================================
# 3. 高级特征工程 (ADVANCED FEATURE ENG)
# ==========================================
print("\n>>> 处理时间序列与特征工程...")
# 按周聚合 (分开计算 Inflow/Outflow 以便后续精确训练)
main_df['Week_Ending'] = main_df['Pstng Date'].dt.to_period('W').apply(lambda r: r.start_time)

# 聚合逻辑：分别计算 Weekly_Inflow, Weekly_Outflow, 以及总 Net
weekly_agg = main_df.groupby(['Name', 'Week_Ending']).agg(
    Weekly_Inflow=('Amount in USD', lambda x: x[x > 0].sum()),
    Weekly_Outflow=('Amount in USD', lambda x: x[x < 0].sum())
).reset_index()
weekly_agg['Amount in USD'] = weekly_agg['Weekly_Inflow'] + weekly_agg['Weekly_Outflow']

# 填充缺失周
entities = weekly_agg['Name'].unique()
all_weeks = pd.date_range(start=weekly_agg['Week_Ending'].min(), end=weekly_agg['Week_Ending'].max(), freq='W-MON')
idx = pd.MultiIndex.from_product([entities, all_weeks], names=['Name', 'Week_Ending'])
full_df = pd.DataFrame(index=idx).reset_index()
weekly_df = pd.merge(full_df, weekly_agg, on=['Name', 'Week_Ending'], how='left').fillna(0)

weekly_df = weekly_df.sort_values(['Name', 'Week_Ending'])

# --- 核心特征构造 ---
weekly_df['Month'] = weekly_df['Week_Ending'].dt.month
weekly_df['Week_Num'] = weekly_df['Week_Ending'].dt.isocalendar().week.astype(int)

# 业务逻辑特征 (捕捉季度末/月末波动)
weekly_df['Is_Quarter_End'] = (
            (weekly_df['Week_Ending'].dt.month.isin([3, 6, 9, 12])) & (weekly_df['Week_Ending'].dt.day >= 21)).astype(
    int)
weekly_df['Is_Month_End'] = (weekly_df['Week_Ending'].dt.day >= 21).astype(int)

# 滞后特征 (基于 Net Flow)
for lag in range(1, 5):
    weekly_df[f'Lag_{lag}'] = weekly_df.groupby('Name')['Amount in USD'].shift(lag)

# 滚动特征
weekly_df['Rolling_Mean_4'] = weekly_df.groupby('Name')['Amount in USD'].transform(
    lambda x: x.shift(1).rolling(window=4).mean())
weekly_df['Rolling_Std_4'] = weekly_df.groupby('Name')['Amount in USD'].transform(
    lambda x: x.shift(1).rolling(window=4).std())

model_data = weekly_df.dropna().copy()
model_data['Name_Encoded'] = le.fit_transform(model_data['Name'])

# 准备训练集
split_date = pd.Timestamp('2025-05-01')
train_df = model_data[model_data['Week_Ending'] < split_date].copy()
test_df = model_data[model_data['Week_Ending'] >= split_date].copy()

features = ['Name_Encoded', 'Month', 'Week_Num', 'Is_Quarter_End', 'Is_Month_End',
            'Lag_1', 'Lag_2', 'Lag_3', 'Lag_4', 'Rolling_Mean_4', 'Rolling_Std_4']

# ==========================================
# 4. OPTUNA 优化 (针对高精度模型)
# ==========================================
print("\n>>> 开始 Optuna 超参数搜索 (Inflow Model)...")
# 为了节省时间，我们针对 "Inflow" (通常更难预测) 进行优化，并将参数应用于 Outflow
X_opt = train_df[features]
y_opt = np.log1p(train_df['Weekly_Inflow'])  # 使用 Log 目标


def objective(trial):
    param = {
        'n_estimators': trial.suggest_int('n_estimators', 300, 1000),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'max_depth': trial.suggest_int('max_depth', 4, 9),  # 允许更深，捕捉尖峰
        'subsample': trial.suggest_float('subsample', 0.6, 0.9),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 0.9),
        'random_state': 42,
        'n_jobs': -1
    }

    # 简单的 TimeSeriesSplit 模拟 (取后20%验证)
    split_idx = int(len(X_opt) * 0.8)
    X_tr, X_val = X_opt.iloc[:split_idx], X_opt.iloc[split_idx:]
    y_tr, y_val = y_opt.iloc[:split_idx], y_opt.iloc[split_idx:]

    model = xgb.XGBRegressor(**param)
    model.fit(X_tr, y_tr)
    preds = model.predict(X_val)

    # --- [修正] 兼容新版 Scikit-Learn: 手动计算 RMSE ---
    mse = mean_squared_error(y_val, preds)
    return np.sqrt(mse)


study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=20)  # 跑20次以保持脚本速度
best_params = study.best_params
print(f"最佳参数: {best_params}")

# ==========================================
# 5. 双模型训练 (Split Inflow/Outflow)
# ==========================================
print("\n>>> 启动高精度双模型训练 (Split Model Strategy)...")

# 准备对数目标 (Log Transformation)
y_in_train = np.log1p(train_df['Weekly_Inflow'])
y_out_train = np.log1p(train_df['Weekly_Outflow'].abs())  # Outflow 取绝对值后 Log

# 使用最佳参数 (稍作调整适配 Outflow)
final_params = best_params.copy()
final_params.update({'n_jobs': -1, 'random_state': 42})

# 训练 Inflow 模型
model_in = xgb.XGBRegressor(**final_params)
model_in.fit(train_df[features], y_in_train)

# 训练 Outflow 模型
model_out = xgb.XGBRegressor(**final_params)
model_out.fit(train_df[features], y_out_train)

# --- 测试集评估 ---
# 预测并还原 (Expm1)
test_df['Pred_In'] = np.expm1(model_in.predict(test_df[features]))
test_df['Pred_Out'] = np.expm1(model_out.predict(test_df[features]))
test_df['Prediction'] = test_df['Pred_In'] - test_df['Pred_Out']  # Net = In - Out

mae = mean_absolute_error(test_df['Amount in USD'], test_df['Prediction'])
rmse = np.sqrt(mean_squared_error(test_df['Amount in USD'], test_df['Prediction']))
print(f"Test Set Performance -> MAE: ${mae:,.0f} | RMSE: ${rmse:,.0f}")

# 绘图：Actual vs Forecast (Branded)
global_test = test_df.groupby('Week_Ending')[['Amount in USD', 'Prediction']].sum().reset_index()
plt.figure(figsize=(12, 6))
plt.plot(global_test['Week_Ending'], global_test['Amount in USD'],
         label='Actual', marker='o', color=az_colors['Primary'], linewidth=2)
plt.plot(global_test['Week_Ending'], global_test['Prediction'],
         label='High-Precision Forecast', marker='x', linestyle='--', color=az_colors['Accent'], linewidth=2)
plt.title(f'Test Performance (Split Model) - MAE: {mae:,.0f}', fontweight='bold', color=az_colors['Primary'])
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('test_performance_branded.png')

# ==========================================
# 6. 未来 6 个月递归预测 (RECURSIVE FORECAST)
# ==========================================
print("\n>>> 生成未来 6 个月预测...")
last_date = model_data['Week_Ending'].max()
future_weeks = pd.date_range(start=last_date + pd.DateOffset(weeks=1), periods=26, freq='W-MON')
last_known_data = model_data.groupby('Name').tail(1).set_index('Name')
future_preds_list = []

for entity in entities:
    current_row = last_known_data.loc[entity].copy()
    entity_code = current_row['Name_Encoded']
    # 提取最近历史数据 (Net Flow)
    history_values = model_data[model_data['Name'] == entity].sort_values('Week_Ending')['Amount in USD'].tail(
        4).tolist()

    for date in future_weeks:
        # 1. 构造特征
        is_q_end = 1 if (date.month in [3, 6, 9, 12] and date.day >= 21) else 0
        is_m_end = 1 if date.day >= 21 else 0

        feat = {
            'Name_Encoded': entity_code,
            'Month': date.month,
            'Week_Num': date.week,
            'Is_Quarter_End': is_q_end,
            'Is_Month_End': is_m_end,
            'Lag_1': history_values[-1],
            'Lag_2': history_values[-2],
            'Lag_3': history_values[-3],
            'Lag_4': history_values[-4],
            'Rolling_Mean_4': np.mean(history_values[-4:]),
            'Rolling_Std_4': np.std(history_values[-4:])
        }
        feat_df = pd.DataFrame([feat])

        # 2. 分别预测 In 和 Out
        pred_in = np.expm1(model_in.predict(feat_df)[0])
        pred_out = np.expm1(model_out.predict(feat_df)[0])
        net_flow = pred_in - pred_out

        future_preds_list.append({
            'Name': entity,
            'Week_Ending': date,
            'Predicted_Net_Flow': net_flow
        })

        # 3. 滚动历史
        history_values.append(net_flow)
        history_values.pop(0)

future_df = pd.DataFrame(future_preds_list)
future_df = future_df.sort_values(['Name', 'Week_Ending'])
future_df['Cumulative_Flow_Change'] = future_df.groupby('Name')['Predicted_Net_Flow'].cumsum()

# ==========================================
# 7. 数据导出与最终可视化 (EXPORTS & VISUALS)
# ==========================================
print("\n>>> 正在导出预测结果...")
# 导出 6 个月
file_6m = "Forecast_Next_6_Months.xlsx"
future_df.to_excel(file_6m, index=False)
print(f"✅ 已导出 6 个月预测: {file_6m}")

# 导出 1 个月
cutoff_1m = future_df['Week_Ending'].min() + pd.DateOffset(weeks=4)
forecast_1m = future_df[future_df['Week_Ending'] <= cutoff_1m]
file_1m = "Forecast_Next_1_Month.xlsx"
forecast_1m.to_excel(file_1m, index=False)
print(f"✅ 已导出 1 个月预测: {file_1m}")

# --- 最终可视化 ---
# 1. 类别深度分析 (Category Deep Dive)
top_categories = main_df.groupby('Category')['Amount in USD'].std().sort_values(ascending=False).head(2).index.tolist()
plt.figure(figsize=(12, 5))
c1 = main_df[main_df['Category'] == top_categories[0]].groupby('Week_Ending')['Amount in USD'].sum().reset_index()
c2 = main_df[main_df['Category'] == top_categories[1]].groupby('Week_Ending')['Amount in USD'].sum().reset_index()
plt.plot(c1['Week_Ending'], c1['Amount in USD'], label=top_categories[0], color=az_colors['Primary'], linewidth=2)
plt.plot(c2['Week_Ending'], c2['Amount in USD'], label=top_categories[1], color=az_colors['Supporting'], linewidth=2,
         linestyle='-.')
plt.title(f'Category Trends: {top_categories[0]} vs {top_categories[1]}', fontweight='bold', color=az_colors['Primary'])
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('category_deep_dive_branded.png')

# 2. 全球 6 个月趋势 (Global 6-Month Forecast)
global_future = future_df.groupby('Week_Ending')['Predicted_Net_Flow'].sum().reset_index()
historical_global = main_df.groupby('Week_Ending')['Amount in USD'].sum().reset_index()

plt.figure(figsize=(12, 6))
plt.plot(historical_global['Week_Ending'].tail(12), historical_global['Amount in USD'].tail(12),
         label='Historical (Last 3 Months)', marker='o', color=az_colors['Primary'])
plt.plot(global_future['Week_Ending'], global_future['Predicted_Net_Flow'],
         label='6-Month Forecast', linestyle='--', marker='x', color=az_colors['Accent'], linewidth=2.5)
plt.fill_between(global_future['Week_Ending'], global_future['Predicted_Net_Flow'] * 0.9,
                 global_future['Predicted_Net_Flow'] * 1.1,
                 color=az_colors['Accent'], alpha=0.1)
plt.title('Global Net Cash Flow: Historical + 6-Month Forecast', fontweight='bold', color=az_colors['Primary'])
plt.axhline(0, color=az_colors['Grey'], linewidth=0.8)
plt.legend(loc='upper left')
plt.grid(True, alpha=0.2)
plt.savefig('6_month_forecast_branded.png')

print("\nAll Done! Congratulations!!")