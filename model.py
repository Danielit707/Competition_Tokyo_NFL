# ============================================================
# NFL DRAFT PREDICTION — HIGH-ACCURACY GPU ENSEMBLE 
# ============================================================

import pandas as pd
import numpy as np
import warnings

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import TargetEncoder, StandardScaler
from scipy.optimize import minimize
from scipy.stats import rankdata

from lightgbm import LGBMClassifier, early_stopping
from catboost import CatBoostClassifier
from xgboost import XGBClassifier

warnings.filterwarnings('ignore')

# ============================================================
# 1. DATA LOADING
# ============================================================

TRAIN_PATH = 'data/train.csv'
TEST_PATH  = 'data/test.csv'

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

y = train_df['Drafted']
test_ids = test_df['Id']

train_df.drop(columns=['Id', 'Drafted'], inplace=True)
test_df.drop(columns=['Id'], inplace=True)

n_train = len(train_df)

# ============================================================
# 2. ADVANCED FEATURE ENGINEERING WITH UNIFIED PROFILING
# ============================================================
df_all = pd.concat([train_df, test_df], axis=0).reset_index(drop=True)

def add_features(df):
    df = df.copy()
    physical_tests = ['Sprint_40yd', 'Vertical_Jump', 'Bench_Press_Reps',
                      'Broad_Jump', 'Agility_3cone', 'Shuttle']

    df['Total_Missing_Tests'] = df[physical_tests].isnull().sum(axis=1)

    for col in physical_tests:
        df[f'{col}_is_missing'] = df[col].isnull().astype(int)

    df['BMI'] = df['Weight'] / (df['Height']**2)
    df['SpeedScore'] = (df['Weight'] * 200) / (df['Sprint_40yd']**4)
    df['ExplosionScore'] = df['Vertical_Jump'] + df['Broad_Jump']
    df['StrengthScore'] = df['Bench_Press_Reps'] / df['Weight']
    df['AgilityScore'] = df['Agility_3cone'] + df['Shuttle']
    df['HeightWeight'] = df['Height'] * df['Weight']
    
    df['Power_Index'] = df['Weight'] * df['Vertical_Jump']
    df['Size_Adjusted_Agility'] = df['AgilityScore'] * df['BMI']
    df['Catch_Radius_Proxy'] = df['Height'] + df['Vertical_Jump']

    df['Speed_x_Weight'] = df['Sprint_40yd'] * df['Weight']
    df['BMI_x_Strength'] = df['BMI'] * df['StrengthScore']
    df['Explosion_per_Weight'] = df['ExplosionScore'] / df['Weight']

    for col in physical_tests:
        df[f'{col}_x_Year'] = df[col] * df['Year']

    df['School_Position'] = df['School'].astype(str) + "_" + df['Position'].astype(str)

    return df

df_all = add_features(df_all)

# Compute both Year-Cohort and Position-Cohort metrics globally to eliminate test drift
physical_tests = ['Sprint_40yd', 'Vertical_Jump', 'Bench_Press_Reps', 'Broad_Jump', 'Agility_3cone', 'Shuttle']
for col in physical_tests:
    # Year Cohorts
    year_means = df_all.groupby('Year')[col].transform('mean')
    year_stds = df_all.groupby('Year')[col].transform('std').fillna(1e-6).replace(0, 1e-6)
    df_all[f'{col}_year_zscore'] = (df_all[col] - year_means) / year_stds
    
    # Position Cohorts (Unified calculation)
    pos_means = df_all.groupby('Position_Type')[col].transform('mean')
    pos_stds = df_all.groupby('Position_Type')[col].transform('std').fillna(1e-6).replace(0, 1e-6)
    pos_meds = df_all.groupby('Position_Type')[col].transform('median')
    
    df_all[f'{col}_z_pos'] = (df_all[col] - pos_means) / pos_stds
    df_all[f'{col}_diff_pos'] = df_all[col] - pos_meds

# Resplit frames cleanly
train_df = df_all.iloc[:n_train].copy().reset_index(drop=True)
test_df = df_all.iloc[n_train:].copy().reset_index(drop=True)

# ============================================================
# 3. ATTRIBUTES CONFIGURATION
# ============================================================
categorical_features = ['School', 'Player_Type', 'Position_Type', 'Position', 'School_Position']

numeric_features = ['Age', 'Weight', 'Height', 'Year', 'BMI', 'Total_Missing_Tests',
                    'SpeedScore', 'ExplosionScore', 'StrengthScore', 'AgilityScore', 
                    'HeightWeight', 'Power_Index', 'Size_Adjusted_Agility', 'Catch_Radius_Proxy',
                    'Sprint_40yd', 'Vertical_Jump', 'Bench_Press_Reps', 'Broad_Jump', 
                    'Agility_3cone', 'Shuttle', 'Speed_x_Weight', 'BMI_x_Strength', 
                    'Explosion_per_Weight']

numeric_features += [f'{col}_x_Year' for col in physical_tests]
numeric_features += [f'{col}_year_zscore' for col in physical_tests]
numeric_features += [f'{col}_z_pos' for col in physical_tests]
numeric_features += [f'{col}_diff_pos' for col in physical_tests]

skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# OOF Arrays store raw structural probabilities to protect baseline calibration
oof_lgb = np.zeros(len(train_df))
oof_cat = np.zeros(len(train_df))
oof_xgb = np.zeros(len(train_df))
oof_xgb_deep = np.zeros(len(train_df))

test_lgb_raw = np.zeros(len(test_df))
test_cat_raw = np.zeros(len(test_df))
test_xgb_raw = np.zeros(len(test_df))
test_xgb_deep_raw = np.zeros(len(test_df))

# ============================================================
# 4. MAIN TRAINING LOOP
# ============================================================
for fold, (train_idx, val_idx) in enumerate(skf.split(train_df, y)):
    X_train = train_df.iloc[train_idx].copy()
    X_val = train_df.iloc[val_idx].copy()
    y_train = y.iloc[train_idx]
    y_val = y.iloc[val_idx]
    X_test_fold = test_df.copy()

    # 1. Target Encoding
    te = TargetEncoder(smooth="auto", cv=5)
    te_cols = ['School', 'Position', 'Player_Type', 'School_Position']
    te_feat_names = [f"{c}_te" for c in te_cols]
    X_train[te_feat_names] = te.fit_transform(X_train[te_cols], y_train)
    X_val[te_feat_names] = te.transform(X_val[te_cols])
    X_test_fold[te_feat_names] = te.transform(X_test_fold[te_cols])

    # 2. Standard Scaling (Preserving Structure)
    scaler = StandardScaler()
    X_train[numeric_features] = scaler.fit_transform(X_train[numeric_features])
    X_val[numeric_features] = scaler.transform(X_val[numeric_features])
    X_test_fold[numeric_features] = scaler.transform(X_test_fold[numeric_features])

    # 3. Frequency Coding & Categorical Cleaning
    for col in categorical_features:
        freq = X_train[col].value_counts()
        X_train[f'{col}_freq'] = X_train[col].map(freq)
        X_val[f'{col}_freq'] = X_val[col].map(freq).fillna(1)
        X_test_fold[f'{col}_freq'] = X_test_fold[col].map(freq).fillna(1)

        X_train[col] = X_train[col].astype(str).replace(['nan', 'None', 'NaN'], 'missing')
        X_val[col] = X_val[col].astype(str).replace(['nan', 'None', 'NaN'], 'missing')
        X_test_fold[col] = X_test_fold[col].astype(str).replace(['nan', 'None', 'NaN'], 'missing')

        cats = X_train[col].unique()
        if "missing" not in cats:
            cats = np.append(cats, "missing")

        X_val[col] = X_val[col].where(X_val[col].isin(cats), "missing")
        X_test_fold[col] = X_test_fold[col].where(X_test_fold[col].isin(cats), "missing")

        X_train[col] = pd.Categorical(X_train[col], categories=cats)
        X_val[col] = pd.Categorical(X_val[col], categories=cats)
        X_test_fold[col] = pd.Categorical(X_test_fold[col], categories=cats)

    X_val = X_val[X_train.columns]
    X_test_fold = X_test_fold[X_train.columns]

    # ========================================================
    # A. LIGHTGBM (Balanced Complexity Variant)
    # ========================================================
    lgb_model = LGBMClassifier(
        boosting_type='gbdt',
        n_estimators=3000,
        learning_rate=0.01,
        num_leaves=42,
        max_depth=7,
        min_child_samples=25,
        subsample=0.8,
        colsample_bytree=0.7,
        reg_alpha=1.5,
        reg_lambda=2.5,
        class_weight='balanced',
        metric='auc',
        random_state=42,
        verbose=-1,
        n_jobs=-1
    )
    lgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[early_stopping(150, verbose=False)])

    # ========================================================
    # B. CATBOOST (High-Stability GPU)
    # ========================================================
    cat_model = CatBoostClassifier(
        iterations=3000,
        learning_rate=0.015,
        depth=6,
        l2_leaf_reg=5,
        auto_class_weights='Balanced',
        eval_metric='AUC',
        early_stopping_rounds=150,
        task_type='GPU',
        random_seed=42,
        verbose=False
    )
    cat_model.fit(X_train, y_train, eval_set=(X_val, y_val), cat_features=categorical_features, use_best_model=True)

    # ========================================================
    # C. XGBOOST
    # ========================================================
    scale_pos_weight = (len(y_train) - sum(y_train)) / sum(y_train)
    xgb_model = XGBClassifier(
        n_estimators=3000,
        learning_rate=0.01,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.7,
        reg_alpha=2.0,
        reg_lambda=6.0,
        scale_pos_weight=scale_pos_weight,
        tree_method='hist',
        device='cuda',
        enable_categorical=True,
        early_stopping_rounds=150,
        eval_metric='auc',
        random_state=42
    )
    xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    # ========================================================
    # D. XGBOOST DEEP
    # ========================================================
    xgb_deep_model = XGBClassifier(
        n_estimators=3000,
        learning_rate=0.008,
        max_depth=8,
        subsample=0.75,
        colsample_bytree=0.5,
        reg_alpha=1.5,
        reg_lambda=12.0,
        scale_pos_weight=scale_pos_weight,
        tree_method='hist',
        device='cuda',
        enable_categorical=True,
        early_stopping_rounds=150,
        eval_metric='auc',
        random_state=2026
    )
    xgb_deep_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    # Store raw probabilities to prevent intra-fold distribution destruction
    oof_lgb[val_idx] = lgb_model.predict_proba(X_val)[:, 1]
    oof_cat[val_idx] = cat_model.predict_proba(X_val)[:, 1]
    oof_xgb[val_idx] = xgb_model.predict_proba(X_val)[:, 1]
    oof_xgb_deep[val_idx] = xgb_deep_model.predict_proba(X_val)[:, 1]

    test_lgb_raw += lgb_model.predict_proba(X_test_fold)[:, 1] / skf.n_splits
    test_cat_raw += cat_model.predict_proba(X_test_fold)[:, 1] / skf.n_splits
    test_xgb_raw += xgb_model.predict_proba(X_test_fold)[:, 1] / skf.n_splits
    test_xgb_deep_raw += xgb_deep_model.predict_proba(X_test_fold)[:, 1] / skf.n_splits

    print(f"--> FOLD {fold+1}/10 | LGBM: {roc_auc_score(y_val, oof_lgb[val_idx]):.5f} | CAT: {roc_auc_score(y_val, oof_cat[val_idx]):.5f} | XGB: {roc_auc_score(y_val, oof_xgb[val_idx]):.5f} | XGB_DEEP: {roc_auc_score(y_val, oof_xgb_deep[val_idx]):.5f}")

# ============================================================
# 5. GLOBAL RANK TRANSFORMATIONS (Executed Exactly Once)
# ============================================================
oof_lgb_rank = rankdata(oof_lgb) / len(oof_lgb)
oof_cat_rank = rankdata(oof_cat) / len(oof_cat)
oof_xgb_rank = rankdata(oof_xgb) / len(oof_xgb)
oof_xgb_deep_rank = rankdata(oof_xgb_deep) / len(oof_xgb_deep)

test_lgb_rank = rankdata(test_lgb_raw) / len(test_lgb_raw)
test_cat_rank = rankdata(test_cat_raw) / len(test_cat_raw)
test_xgb_rank = rankdata(test_xgb_raw) / len(test_xgb_raw)
test_xgb_deep_rank = rankdata(test_xgb_deep_raw) / len(test_xgb_deep_raw)

def auc_loss(weights_raw):
    exp_w = np.exp(weights_raw - np.max(weights_raw))
    w = exp_w / np.sum(exp_w)

    blend = w[0] * oof_lgb_rank + w[1] * oof_cat_rank + w[2] * oof_xgb_rank + w[3] * oof_xgb_deep_rank
    return -roc_auc_score(y, blend)

opt_res = minimize(auc_loss, x0=[0.0, 0.0, 0.0, 0.0], method='Powell')

exp_w_opt = np.exp(opt_res.x - np.max(opt_res.x))
w_lgb, w_cat, w_xgb, w_xgb_deep = exp_w_opt / np.sum(exp_w_opt)

final_oof = w_lgb * oof_lgb_rank + w_cat * oof_cat_rank + w_xgb * oof_xgb_rank + w_xgb_deep * oof_xgb_deep_rank
final_score = roc_auc_score(y, final_oof)

print('\n==================================================')
print(f'OPTIMAL RANK WEIGHTS -> LGBM: {w_lgb:.4f} | CAT: {w_cat:.4f} | XGB: {w_xgb:.4f} | XGB_DEEP: {w_xgb_deep:.4f}')
print(f'CORRECTED GLOBAL ROC AUC (10-Fold CV Rank Ensemble): {final_score:.5f}')
print('==================================================')

# ============================================================
# 6. EXPORT SUBMISSION
# ============================================================
final_test_preds = np.clip(w_lgb * test_lgb_rank + w_cat * test_cat_rank + w_xgb * test_xgb_rank + w_xgb_deep * test_xgb_deep_rank, 0.0, 1.0)

submission = pd.DataFrame({
    'Id': test_ids,
    'Drafted': final_test_preds
})

OUTPUT_PATH = 'data/submission_advanced_ensemble.csv'
submission.to_csv(OUTPUT_PATH, index=False)
print(f'\nSuccess! Calibrated high-performance submission saved to:\n{OUTPUT_PATH}')