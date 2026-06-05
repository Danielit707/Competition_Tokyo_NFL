# ============================================================
# NFL DRAFT PREDICTION — HIGH-ACCURACY GPU ENSEMBLE (v5.5 Pro)
# ============================================================

import pandas as pd
import numpy as np
import warnings


from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import TargetEncoder, StandardScaler
from scipy.optimize import minimize

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

# ============================================================
# 2. ADVANCED FEATURE ENGINEERING
# ============================================================
def add_features(df):
    df = df.copy()
    physical_tests = ['Sprint_40yd', 'Vertical_Jump', 'Bench_Press_Reps',
                      'Broad_Jump', 'Agility_3cone', 'Shuttle']

    #1. Performance at the Combine: Total number of tests missed
    df['Total_Missing_Tests'] = df[physical_tests].isnull().sum(axis=1)

    for col in physical_tests:
        df[f'{col}_is_missing'] = df[col].isnull().astype(int)

    #2. Vectorized physical formulas
    df['BMI'] = df['Weight'] / (df['Height']**2)
    df['SpeedScore'] = (df['Weight'] * 200) / (df['Sprint_40yd']**4)
    df['ExplosionScore'] = df['Vertical_Jump'] + df['Broad_Jump']
    df['StrengthScore'] = df['Bench_Press_Reps'] / df['Weight']
    df['AgilityScore'] = df['Agility_3cone'] + df['Shuttle']
    df['HeightWeight'] = df['Height'] * df['Weight']

    #3. Cross-interactions between mass and motion
    df['Speed_x_Weight'] = df['Sprint_40yd'] * df['Weight']
    df['BMI_x_Strength'] = df['BMI'] * df['StrengthScore']
    df['Explosion_per_Weight'] = df['ExplosionScore'] / df['Weight']

    # 4. Cruzado con el año histórico
    for col in physical_tests:
        df[f'{col}_x_Year'] = df[col] * df['Year']

    #5. Compound categorical interaction: Captures the “Position Factory” effect
    df['School_Position'] = df['School'].astype(str) + "_" + df['Position'].astype(str)

    return df

train_df = add_features(train_df)
test_df = add_features(test_df)

# ============================================================
# 3. ATRIBUTES CONFIGURATION
# ============================================================
categorical_features = ['School', 'Player_Type', 'Position_Type', 'Position', 'School_Position']

numeric_features = ['Age', 'Weight', 'Height', 'Year', 'BMI', 'Total_Missing_Tests',
                    'SpeedScore', 'ExplosionScore', 'StrengthScore',
                    'AgilityScore', 'HeightWeight',
                    'Sprint_40yd', 'Vertical_Jump', 'Bench_Press_Reps',
                    'Broad_Jump', 'Agility_3cone', 'Shuttle',
                    'Speed_x_Weight', 'BMI_x_Strength', 'Explosion_per_Weight']

numeric_features += [f'{col}_x_Year' for col in ['Sprint_40yd', 'Vertical_Jump',
                     'Bench_Press_Reps', 'Broad_Jump', 'Agility_3cone', 'Shuttle']]

physical_tests = ['Sprint_40yd', 'Vertical_Jump', 'Bench_Press_Reps',
                  'Broad_Jump', 'Agility_3cone', 'Shuttle']

skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

oof_lgb = np.zeros(len(train_df))
oof_cat = np.zeros(len(train_df))
oof_xgb = np.zeros(len(train_df))

test_lgb = np.zeros(len(test_df))
test_cat = np.zeros(len(test_df))
test_xgb = np.zeros(len(test_df))

# ============================================================
# 4. MAIN TRAINING LOOP
# ============================================================
for fold, (train_idx, val_idx) in enumerate(skf.split(train_df, y)):
    X_train = train_df.iloc[train_idx].copy()
    X_val = train_df.iloc[val_idx].copy()
    y_train = y.iloc[train_idx]
    y_val = y.iloc[val_idx]
    X_test_fold = test_df.copy()

    # 1. Robust Target Encoding
    te = TargetEncoder(smooth="auto", cv=5)
    te_cols = ['School', 'Position', 'Player_Type', 'School_Position']
    te_feat_names = [f"{c}_te" for c in te_cols]
    X_train[te_feat_names] = te.fit_transform(X_train[te_cols], y_train)
    X_val[te_feat_names] = te.transform(X_val[te_cols])
    X_test_fold[te_feat_names] = te.transform(X_test_fold[te_cols])

    # 2. Standard scaling while preserving NaNs
    scaler = StandardScaler()
    X_train[numeric_features] = scaler.fit_transform(X_train[numeric_features])
    X_val[numeric_features] = scaler.transform(X_val[numeric_features])
    X_test_fold[numeric_features] = scaler.transform(X_test_fold[numeric_features])

    # 3. Z-Score and Local Deviations by Position (Defensive vs. NaNs / Out-of-bounds)
    for col in physical_tests:
        means = X_train.groupby('Position_Type')[col].mean()
        stds = X_train.groupby('Position_Type')[col].std().replace(0, 1e-6)
        pos_med = X_train.groupby('Position_Type')[col].median()
        
        # Global fallbacks for orphaned categories during validation
        global_mean = X_train[col].mean()
        global_std = X_train[col].std() if X_train[col].std() != 0 else 1e-6
        global_med = X_train[col].median()
        
        X_train[f'{col}_z_pos'] = (X_train[col] - X_train['Position_Type'].map(means)) / X_train['Position_Type'].map(stds)
        X_val[f'{col}_z_pos']   = (X_val[col]   - X_val['Position_Type'].map(means).fillna(global_mean))   / X_val['Position_Type'].map(stds).fillna(global_std)
        X_test_fold[f'{col}_z_pos'] = (X_test_fold[col] - X_test_fold['Position_Type'].map(means).fillna(global_mean)) / X_test_fold['Position_Type'].map(stds).fillna(global_std)

        X_train[f'{col}_diff_pos'] = X_train[col] - X_train['Position_Type'].map(pos_med)
        X_val[f'{col}_diff_pos'] = X_val[col] - X_val['Position_Type'].map(pos_med).fillna(global_med)
        X_test_fold[f'{col}_diff_pos'] = X_test_fold[col] - X_test_fold['Position_Type'].map(pos_med).fillna(global_med)

    # 4. Frequency coding and strict typing free of categorical NaNs
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

    # Enforce identical column order to prevent silent array errors
    X_val = X_val[X_train.columns]
    X_test_fold = X_test_fold[X_train.columns]

    # ========================================================
    # A. LIGHTGBM 
    # ========================================================
    lgb_model = LGBMClassifier(
        boosting_type='gbdt',
        n_estimators=3000,
        learning_rate=0.01,
        num_leaves=45,
        max_depth=7,
        min_child_samples=25,
        subsample=0.8,
        colsample_bytree=0.7,
        reg_alpha=1.5,
        reg_lambda=2.5,
        class_weight='balanced',
        random_state=42,
        verbose=-1,
        n_jobs=-1
    )
    lgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[early_stopping(150, verbose=False)])

    # ========================================================
    # B. CATBOOST 
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
    # ENSEMBLE PREDICTIONS
    # ========================================================
    pred_lgb = lgb_model.predict_proba(X_val)[:, 1]
    pred_cat = cat_model.predict_proba(X_val)[:, 1]
    pred_xgb = xgb_model.predict_proba(X_val)[:, 1]

    oof_lgb[val_idx] = pred_lgb
    oof_cat[val_idx] = pred_cat
    oof_xgb[val_idx] = pred_xgb

    test_lgb += lgb_model.predict_proba(X_test_fold)[:, 1] / skf.n_splits
    test_cat += cat_model.predict_proba(X_test_fold)[:, 1] / skf.n_splits
    test_xgb += xgb_model.predict_proba(X_test_fold)[:, 1] / skf.n_splits

    print(f"--> FOLD {fold+1}/10 | LGBM: {roc_auc_score(y_val, pred_lgb):.5f} | CAT: {roc_auc_score(y_val, pred_cat):.5f} | XGB: {roc_auc_score(y_val, pred_xgb):.5f}")

# ============================================================
# 5. OPTIMIZATION METAMODEL ON THE AUC HYPERPLANE (POWELL + SOFTMAX)
# ============================================================
def auc_loss(weights_raw):
    # Convert arbitrary real numbers into a probability vector (sum 1, range [0,1])
    exp_w = np.exp(weights_raw - np.max(weights_raw)) # Estabilidad numérica contra desbordamientos
    w = exp_w / np.sum(exp_w)
    
    blend = w[0] * oof_lgb + w[1] * oof_cat + w[2] * oof_xgb
    return -roc_auc_score(y, blend)

# Powell does not require direct derivatives; it will skip the flat plateaus in the AUC range calculation
opt_res = minimize(auc_loss, x0=[0.0, 0.0, 0.0], method='Powell')

# Reconstruct the normalized weights from the returned optimal vector
exp_w_opt = np.exp(opt_res.x - np.max(opt_res.x))
w_lgb, w_cat, w_xgb = exp_w_opt / np.sum(exp_w_opt)

final_oof = w_lgb * oof_lgb + w_cat * oof_cat + w_xgb * oof_xgb
final_score = roc_auc_score(y, final_oof)

print('\n==================================================')
print(f'PESOS ÓPTIMOS ENCONTRADOS -> LGBM: {w_lgb:.4f} | CAT: {w_cat:.4f} | XGB: {w_xgb:.4f}')
print(f'NUEVO ROC AUC GLOBAL (10-Fold CV Ensamble): {final_score:.5f}')
print('==================================================')

# ============================================================
# 6. EXPORT SUBMISSION
# ============================================================
final_test_preds = np.clip(w_lgb * test_lgb + w_cat * test_cat + w_xgb * test_xgb, 0.0, 1.0)

submission = pd.DataFrame({
    'Id': test_ids,
    'Drafted': final_test_preds
})

OUTPUT_PATH = 'data/submission_advanced_ensemble.csv'
submission.to_csv(OUTPUT_PATH, index=False)
print(f'\nCompleted. File saved in:\n{OUTPUT_PATH}')