# ============================================================
# NFL DRAFT PREDICTION — FINAL ENSEMBLE (v3.0)
# ============================================================

import pandas as pd
import numpy as np
import warnings

from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import TargetEncoder, StandardScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import LogisticRegression

from lightgbm import LGBMClassifier, early_stopping
from catboost import CatBoostClassifier
from xgboost import XGBClassifier

warnings.filterwarnings('ignore')

# ============================================================
# 1. CARGA DE DATOS
# ============================================================
TRAIN_PATH = 'data/train.csv'
TEST_PATH  = 'data/test.csv'

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

y = train_df['Drafted']
test_ids = test_df['Id']

# Conservamos 'Year' como variable
train_df.drop(columns=['Id', 'Drafted'], inplace=True)
test_df.drop(columns=['Id'], inplace=True)

# ============================================================
# 2. INGENIERÍA DE CARACTERÍSTICAS (vectorizada)
# ============================================================
def add_features(df):
    df = df.copy()
    
    physical_tests = ['Sprint_40yd', 'Vertical_Jump', 'Bench_Press_Reps',
                      'Broad_Jump', 'Agility_3cone', 'Shuttle']
    
    # 1. Indicadores de valores nulos
    for col in physical_tests:
        df[f'{col}_is_missing'] = df[col].isnull().astype(int)

    # 2. Fórmulas físicas básicas
    df['BMI'] = df['Weight'] / (df['Height']**2)
    df['SpeedScore'] = (df['Weight'] * 200) / (df['Sprint_40yd']**4)
    df['ExplosionScore'] = df['Vertical_Jump'] + df['Broad_Jump']
    df['StrengthScore'] = df['Bench_Press_Reps'] / df['Weight']
    df['AgilityScore'] = df['Agility_3cone'] + df['Shuttle']
    df['HeightWeight'] = df['Height'] * df['Weight']

    # 3. Interacciones relevantes
    df['Speed_x_Weight'] = df['Sprint_40yd'] * df['Weight']
    df['BMI_x_Strength'] = df['BMI'] * df['StrengthScore']
    df['Explosion_per_Weight'] = df['ExplosionScore'] / df['Weight']

    return df

train_df = add_features(train_df)
test_df = add_features(test_df)

# ============================================================
# 3. CONFIGURACIÓN
# ============================================================
categorical_features = ['School', 'Player_Type', 'Position_Type', 'Position']

numeric_features = ['Age', 'Weight', 'Height', 'Year', 'BMI',
                    'SpeedScore', 'ExplosionScore', 'StrengthScore',
                    'AgilityScore', 'HeightWeight',
                    'Sprint_40yd', 'Vertical_Jump', 'Bench_Press_Reps',
                    'Broad_Jump', 'Agility_3cone', 'Shuttle',
                    'Speed_x_Weight', 'BMI_x_Strength', 'Explosion_per_Weight']

physical_tests = ['Sprint_40yd', 'Vertical_Jump', 'Bench_Press_Reps',
                  'Broad_Jump', 'Agility_3cone', 'Shuttle']

# ============================================================
# 4. VALIDACIÓN CRUZADA (10 folds)
# ============================================================
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# Solo usaremos los tres modelos fuertes
oof_lgb = np.zeros(len(train_df))
oof_cat = np.zeros(len(train_df))
oof_xgb = np.zeros(len(train_df))

test_lgb = np.zeros(len(test_df))
test_cat = np.zeros(len(test_df))
test_xgb = np.zeros(len(test_df))

# ============================================================
# 5. BUCLE DE ENTRENAMIENTO (tuning más intensivo)
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
    # 2. ESCALADO E IMPUTACIÓN
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
    # 3. Z‑SCORE DENTRO DE POSITION_TYPE (post-imputación)
    # --------------------------------------------------------
    for col in physical_tests:
        means = X_train.groupby('Position_Type')[col].mean()
        stds = X_train.groupby('Position_Type')[col].std().replace(0, 1e-6)
        
        X_train[f'{col}_z_pos'] = (X_train[col] - X_train['Position_Type'].map(means)) / X_train['Position_Type'].map(stds)
        X_val[f'{col}_z_pos']   = (X_val[col]   - X_val['Position_Type'].map(means))   / X_val['Position_Type'].map(stds)
        X_test_fold[f'{col}_z_pos'] = (X_test_fold[col] - X_test_fold['Position_Type'].map(means)) / X_test_fold['Position_Type'].map(stds)
        
        X_val[f'{col}_z_pos'].fillna(0, inplace=True)
        X_test_fold[f'{col}_z_pos'].fillna(0, inplace=True)

    # Diferencia a la mediana (complementaria)
    for col in physical_tests:
        pos_med = X_train.groupby('Position_Type')[col].median()
        X_train[f'{col}_diff_pos'] = X_train[col] - X_train['Position_Type'].map(pos_med)
        X_val[f'{col}_diff_pos'] = X_val[col] - X_val['Position_Type'].map(pos_med)
        X_test_fold[f'{col}_diff_pos'] = X_test_fold[col] - X_test_fold['Position_Type'].map(pos_med)
        
        X_val[f'{col}_diff_pos'].fillna(0, inplace=True)
        X_test_fold[f'{col}_diff_pos'].fillna(0, inplace=True)

    # --------------------------------------------------------
    # 4. FREQ ENCODING Y CONVERSIÓN A CATEGORÍA
    # --------------------------------------------------------
    for col in categorical_features:
        freq = X_train[col].value_counts()
        X_train[f'{col}_freq'] = X_train[col].map(freq)
        X_val[f'{col}_freq'] = X_val[col].map(freq)
        X_test_fold[f'{col}_freq'] = X_test_fold[col].map(freq)
        
        X_val[f'{col}_freq'].fillna(1, inplace=True)
        X_test_fold[f'{col}_freq'].fillna(1, inplace=True)
        
        X_train[col] = X_train[col].astype('category')
        X_val[col] = X_val[col].astype('category')
        X_test_fold[col] = X_test_fold[col].astype('category')

    # ========================================================
    # A. LIGHTGBM (búsqueda más exhaustiva: 20 iteraciones)
    # ========================================================
    lgb_param_dist = {
        'n_estimators': [2000],
        'learning_rate': [0.01, 0.02],
        'num_leaves': [20, 31, 40, 50],
        'max_depth': [4, 6, 8],
        'min_child_samples': [20, 30, 50],
        'subsample': [0.7, 0.8, 0.9],
        'colsample_bytree': [0.7, 0.8, 0.9],
        'reg_alpha': [0.1, 1.0, 2.0],
        'reg_lambda': [0.1, 1.0, 2.0]
    }
    lgb_base = LGBMClassifier(
        class_weight='balanced',
        random_state=42,
        verbose=-1,
        early_stopping_round=100,
        force_col_wise=True
    )
    lgb_rs = RandomizedSearchCV(
        lgb_base, lgb_param_dist, n_iter=20, scoring='roc_auc',
        cv=3, random_state=42, n_jobs=-1
    )
    lgb_rs.fit(X_train, y_train, eval_set=[(X_val, y_val)])
    lgb_model = lgb_rs.best_estimator_

    # ========================================================
    # B. CATBOOST (mantenemos parámetros afinados)
    # ========================================================
    cat_model = CatBoostClassifier(
        iterations=2000,
        learning_rate=0.02,
        depth=4,
        l2_leaf_reg=5,
        auto_class_weights='Balanced',
        eval_metric='AUC',
        early_stopping_rounds=100,
        random_seed=42,
        verbose=False
    )
    cat_model.fit(
        X_train, y_train,
        eval_set=(X_val, y_val),
        cat_features=categorical_features,
        use_best_model=True
    )

    # ========================================================
    # C. XGBOOST (búsqueda más exhaustiva: 20 iteraciones)
    # ========================================================
    scale_pos_weight = (len(y_train) - sum(y_train)) / sum(y_train)
    xgb_param_dist = {
        'n_estimators': [2000],
        'learning_rate': [0.01, 0.02, 0.03],
        'max_depth': [4, 5, 6],
        'subsample': [0.7, 0.8, 0.9],
        'colsample_bytree': [0.7, 0.8, 0.9],
        'reg_alpha': [0.1, 1.0, 2.0],
        'reg_lambda': [1.0, 3.0, 5.0],
    }
    xgb_base = XGBClassifier(
        scale_pos_weight=scale_pos_weight,
        enable_categorical=True,
        eval_metric='auc',
        early_stopping_rounds=100,
        random_state=42
    )
    xgb_rs = RandomizedSearchCV(
        xgb_base, xgb_param_dist, n_iter=20, scoring='roc_auc',
        cv=3, random_state=42, n_jobs=-1
    )
    xgb_rs.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    xgb_model = xgb_rs.best_estimator_

    # ========================================================
    # PREDICCIONES
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

    auc_lgb = roc_auc_score(y_val, pred_lgb)
    auc_cat = roc_auc_score(y_val, pred_cat)
    auc_xgb = roc_auc_score(y_val, pred_xgb)
    print(f"AUC -> LGBM: {auc_lgb:.5f} | CAT: {auc_cat:.5f} | XGB: {auc_xgb:.5f}")

# ============================================================
# 6. STACKING CON REGRESIÓN LOGÍSTICA (meta‑modelo nativo)
# ============================================================
oof_matrix = np.column_stack([oof_lgb, oof_cat, oof_xgb])
test_matrix = np.column_stack([test_lgb, test_cat, test_xgb])

# La regresión logística produce probabilidades y mantiene todo en [0,1]
meta = LogisticRegression(penalty=None, class_weight='balanced', random_state=42)
meta.fit(oof_matrix, y)

# Pesos normalizados para interpretación (solo informativo)
weights = np.abs(meta.coef_[0])
weights /= weights.sum()
w_lgb, w_cat, w_xgb = weights

final_oof = meta.predict_proba(oof_matrix)[:, 1]
final_score = roc_auc_score(y, final_oof)

print('\n===============================')
print(f'PESOS (LogReg) -> LGBM: {w_lgb:.3f} | CAT: {w_cat:.3f} | XGB: {w_xgb:.3f}')
print(f'FINAL ROC AUC (CV): {final_score:.5f}')
print('===============================')

# ============================================================
# 7. EXPORTAR SUBMISSION (con recorte explícito)
# ============================================================
final_test_preds = meta.predict_proba(test_matrix)[:, 1]
# Aseguramos valores en [0,1] por si hubiera pequeños errores numéricos
final_test_preds = np.clip(final_test_preds, 0.0, 1.0)

submission = pd.DataFrame({
    'Id': test_ids,
    'Drafted': final_test_preds
})

OUTPUT_PATH = 'data/submission_final_ensemble.csv'
submission.to_csv(OUTPUT_PATH, index=False)
print(f'\nSubmission guardada en:\n{OUTPUT_PATH}')