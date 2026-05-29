# ============================================================
# NFL DRAFT PREDICTION — RIGOROUS ENSEMBLE
# ============================================================

import pandas as pd
import numpy as np
import warnings

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import TargetEncoder, StandardScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from scipy.optimize import minimize

from lightgbm import LGBMClassifier, early_stopping
from catboost import CatBoostClassifier
from xgboost import XGBClassifier

warnings.filterwarnings('ignore')

# ============================================================
# 1. LOAD DATA
# ============================================================
TRAIN_PATH = 'data/train.csv'
TEST_PATH  = 'data/test.csv'

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

y = train_df['Drafted']
test_ids = test_df['Id']

train_df.drop(columns=['Id','Drafted'], inplace=True)
test_df.drop(columns=['Id'], inplace=True)

# ============================================================
# 2. FEATURE ENGINEERING (Vectorizada)
# ============================================================
def add_features(df):
    df = df.copy()
    
    physical_tests = ['Sprint_40yd', 'Vertical_Jump', 'Bench_Press_Reps', 
                      'Broad_Jump', 'Agility_3cone', 'Shuttle']
    
    # 1. Indicadores de nulos
    for col in physical_tests:
        df[f'{col}_is_missing'] = df[col].isnull().astype(int)

    # 2. Fórmulas Físicas
    df['BMI'] = df['Weight'] / (df['Height']**2)
    df['SpeedScore'] = (df['Weight'] * 200) / (df['Sprint_40yd']**4)
    df['ExplosionScore'] = df['Vertical_Jump'] + df['Broad_Jump']
    df['StrengthScore'] = df['Bench_Press_Reps'] / df['Weight']
    df['AgilityScore'] = df['Agility_3cone'] + df['Shuttle']
    df['HeightWeight'] = df['Height'] * df['Weight']

    return df

train_df = add_features(train_df)
test_df = add_features(test_df)

# ============================================================
# 3. CONFIG
# ============================================================
categorical_features = ['School', 'Player_Type', 'Position_Type', 'Position']

numeric_features = ['Age', 'Weight', 'Height', 'BMI', 'SpeedScore', 'ExplosionScore', 
                    'StrengthScore', 'AgilityScore', 'HeightWeight',
                    'Sprint_40yd', 'Vertical_Jump', 'Bench_Press_Reps', 
                    'Broad_Jump', 'Agility_3cone', 'Shuttle']

physical_tests = ['Sprint_40yd', 'Vertical_Jump', 'Bench_Press_Reps', 
                  'Broad_Jump', 'Agility_3cone', 'Shuttle']

# ============================================================
# 4. CROSS VALIDATION
# ============================================================
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

oof_lgb = np.zeros(len(train_df))
oof_cat = np.zeros(len(train_df))
oof_xgb = np.zeros(len(train_df))

test_lgb = np.zeros(len(test_df))
test_cat = np.zeros(len(test_df))
test_xgb = np.zeros(len(test_df))

# ============================================================
# 5. TRAIN LOOP (Riguroso)
# ============================================================
for fold, (train_idx, val_idx) in enumerate(skf.split(train_df, y)):
    print(f'\n========== FOLD {fold+1} ==========')

    X_train = train_df.iloc[train_idx].copy()
    X_val = train_df.iloc[val_idx].copy()
    y_train = y.iloc[train_idx]
    y_val = y.iloc[val_idx]
    X_test_fold = test_df.copy()

    # --------------------------------------------------------
    # 1. TARGET ENCODING
    # --------------------------------------------------------
    te = TargetEncoder(smooth="auto")
    te_cols = ['School', 'Position', 'Player_Type']
    te_feat_names = [f"{c}_te" for c in te_cols]
    
    X_train[te_feat_names] = te.fit_transform(X_train[te_cols], y_train)
    X_val[te_feat_names] = te.transform(X_val[te_cols])
    X_test_fold[te_feat_names] = te.transform(X_test_fold[te_cols])

    # --------------------------------------------------------
    # 2. ESCALADO VECTORIZADO Y MULTI-IMPUTACIÓN
    # --------------------------------------------------------
    scaler = StandardScaler()
    X_train[numeric_features] = scaler.fit_transform(X_train[numeric_features])
    X_val[numeric_features] = scaler.transform(X_val[numeric_features])
    X_test_fold[numeric_features] = scaler.transform(X_test_fold[numeric_features])

    imputer = IterativeImputer(max_iter=10, random_state=42)
    X_train[numeric_features] = imputer.fit_transform(X_train[numeric_features])
    X_val[numeric_features] = imputer.transform(X_val[numeric_features])
    X_test_fold[numeric_features] = imputer.transform(X_test_fold[numeric_features])

    # --------------------------------------------------------
    # 3. POSITION DIFFERENCE FEATURES (Post-Estandarización)
    # --------------------------------------------------------
    for col in physical_tests:
        pos_med = X_train.groupby('Position_Type')[col].median()
        X_train[f'{col}_diff_pos'] = X_train[col] - X_train['Position_Type'].map(pos_med)
        X_val[f'{col}_diff_pos'] = X_val[col] - X_val['Position_Type'].map(pos_med)
        X_test_fold[f'{col}_diff_pos'] = X_test_fold[col] - X_test_fold['Position_Type'].map(pos_med)

    # --------------------------------------------------------
    # 4. FREQ ENCODING Y TIPOS
    # --------------------------------------------------------
    for col in categorical_features:
        freq = X_train[col].value_counts()
        X_train[f'{col}_freq'] = X_train[col].map(freq)
        X_val[f'{col}_freq'] = X_val[col].map(freq)
        X_test_fold[f'{col}_freq'] = X_test_fold[col].map(freq)
        
        X_train[col] = X_train[col].astype('category')
        X_val[col] = X_val[col].astype('category')
        X_test_fold[col] = X_test_fold[col].astype('category')

    # ========================================================
    # A. LIGHTGBM 
    # ========================================================
    lgb_model = LGBMClassifier(
        n_estimators=1500, learning_rate=0.02, num_leaves=31, 
        max_depth=6, min_child_samples=20, subsample=0.8,
        colsample_bytree=0.8, reg_alpha=1.0, reg_lambda=1.0,
        class_weight='balanced', random_state=42, verbose=-1
    )
    lgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[early_stopping(100, verbose=False)])

    # ========================================================
    # B. CATBOOST
    # ========================================================
    cat_model = CatBoostClassifier(
        iterations=1500, learning_rate=0.03, depth=5, l2_leaf_reg=3,
        auto_class_weights='Balanced', eval_metric='AUC',
        early_stopping_rounds=100, random_seed=42, verbose=False
    )
    cat_model.fit(X_train, y_train, eval_set=(X_val, y_val), cat_features=categorical_features, use_best_model=True)

    # ========================================================
    # C. XGBOOST (NUEVO)
    # ========================================================
    xgb_model = XGBClassifier(
        n_estimators=1500, learning_rate=0.02, max_depth=5,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=1.0, reg_lambda=3.0, # Mayor penalización L2
        scale_pos_weight=(len(y_train) - sum(y_train)) / sum(y_train), # Balanceo interno
        enable_categorical=True, early_stopping_rounds=100,
        eval_metric='auc', random_state=42
    )
    xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    # ========================================================
    # PREDICTIONS
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
    
    print(f"AUC -> LGBM: {roc_auc_score(y_val, pred_lgb):.5f} | CAT: {roc_auc_score(y_val, pred_cat):.5f} | XGB: {roc_auc_score(y_val, pred_xgb):.5f}")

# ============================================================
# 6. OPTIMIZACIÓN NUMÉRICA RESTRINGIDA (SLSQP)
# ============================================================
def auc_loss(weights):
    # La pérdida es el AUC negativo (buscamos minimizarlo)
    blend = weights[0] * oof_lgb + weights[1] * oof_cat + weights[2] * oof_xgb
    return -roc_auc_score(y, blend)

# Restricción: La suma de todos los pesos debe ser exactamente 1
cons = ({'type': 'eq', 'fun': lambda w: 1 - sum(w)})
bounds = [(0, 1), (0, 1), (0, 1)]

# Empezamos asumiendo una distribución equitativa (1/3 a cada uno)
opt_res = minimize(auc_loss, x0=[1/3, 1/3, 1/3], bounds=bounds, constraints=cons, method='SLSQP')
w_lgb, w_cat, w_xgb = opt_res.x

final_oof = w_lgb * oof_lgb + w_cat * oof_cat + w_xgb * oof_xgb
final_score = roc_auc_score(y, final_oof)

print('\n===============================')
print(f'PESOS ÓPTIMOS (Restringidos) -> LGBM: {w_lgb:.3f} | CAT: {w_cat:.3f} | XGB: {w_xgb:.3f}')
print(f'FINAL ROC AUC (CV): {final_score:.5f}')
print('===============================')

# ============================================================
# 7. EXPORT SUBMISSION
# ============================================================
final_test_preds = w_lgb * test_lgb + w_cat * test_cat + w_xgb * test_xgb

submission = pd.DataFrame({
    'Id': test_ids,
    'Drafted': final_test_preds
})

OUTPUT_PATH = 'data/submission_rigorous_ensemble.csv'
submission.to_csv(OUTPUT_PATH, index=False)
print(f'\nSubmission guardada en:\n{OUTPUT_PATH}')