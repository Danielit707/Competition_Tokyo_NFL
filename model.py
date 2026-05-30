# ============================================================
# NFL DRAFT PREDICTION — GPU‑DIVERSE ENSEMBLE (v5.0)
# ============================================================

import pandas as pd
import numpy as np
import warnings

from sklearn.model_selection import StratifiedKFold, GridSearchCV
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

train_df.drop(columns=['Id', 'Drafted'], inplace=True)
test_df.drop(columns=['Id'], inplace=True)

# ============================================================
# 2. INGENIERÍA DE CARACTERÍSTICAS
# ============================================================
def add_features(df):
    df = df.copy()
    
    physical_tests = ['Sprint_40yd', 'Vertical_Jump', 'Bench_Press_Reps',
                      'Broad_Jump', 'Agility_3cone', 'Shuttle']
    
    for col in physical_tests:
        df[f'{col}_is_missing'] = df[col].isnull().astype(int)

    df['BMI'] = df['Weight'] / (df['Height']**2)
    df['SpeedScore'] = (df['Weight'] * 200) / (df['Sprint_40yd']**4)
    df['ExplosionScore'] = df['Vertical_Jump'] + df['Broad_Jump']
    df['StrengthScore'] = df['Bench_Press_Reps'] / df['Weight']
    df['AgilityScore'] = df['Agility_3cone'] + df['Shuttle']
    df['HeightWeight'] = df['Height'] * df['Weight']

    df['Speed_x_Weight'] = df['Sprint_40yd'] * df['Weight']
    df['BMI_x_Strength'] = df['BMI'] * df['StrengthScore']
    df['Explosion_per_Weight'] = df['ExplosionScore'] / df['Weight']

    for col in physical_tests:
        df[f'{col}_x_Year'] = df[col] * df['Year']

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

numeric_features += [f'{col}_x_Year' for col in ['Sprint_40yd', 'Vertical_Jump',
                     'Bench_Press_Reps', 'Broad_Jump', 'Agility_3cone', 'Shuttle']]

physical_tests = ['Sprint_40yd', 'Vertical_Jump', 'Bench_Press_Reps',
                  'Broad_Jump', 'Agility_3cone', 'Shuttle']

# ============================================================
# 4. MODELOS BASE DIVERSOS (5 configuraciones)
# ============================================================
# Definimos variantes que se entrenarán en cada fold
base_models = {
    'LGB_gbdt': LGBMClassifier(
        boosting_type='gbdt', n_estimators=1500, learning_rate=0.015,
        num_leaves=50, max_depth=6, min_child_samples=30,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.5, reg_lambda=1.5,
        class_weight='balanced', random_state=42, verbose=-1,
        force_col_wise=True
    ),
    'LGB_rf': LGBMClassifier(
        boosting_type='rf', n_estimators=1500, learning_rate=0.015,
        num_leaves=70, max_depth=8, min_child_samples=40,
        subsample=0.8, colsample_bytree=0.7,
        reg_alpha=0.1, reg_lambda=0.1,
        class_weight='balanced', random_state=42, verbose=-1,
        extra_trees=True, force_col_wise=True
    ),
    'CAT_depth4': CatBoostClassifier(
        iterations=1500, learning_rate=0.02, depth=4, l2_leaf_reg=4,
        auto_class_weights='Balanced', eval_metric='AUC',
        early_stopping_rounds=100, random_seed=42, verbose=False,
        task_type='GPU', devices='0'   # Usa tu RTX 2060
    ),
    'CAT_depth6': CatBoostClassifier(
        iterations=1500, learning_rate=0.02, depth=6, l2_leaf_reg=3,
        auto_class_weights='Balanced', eval_metric='AUC',
        early_stopping_rounds=100, random_seed=42, verbose=False,
        task_type='GPU', devices='0'
    ),
    'XGB_gbtree': XGBClassifier(
        n_estimators=1500, learning_rate=0.015, max_depth=5,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=1.0, reg_lambda=3.0,
        scale_pos_weight=1.0,              # se calculará por fold
        tree_method='hist',            # GPU
        enable_categorical=True,
        early_stopping_rounds=100,
        eval_metric='auc', random_state=42
    ),
    'XGB_dart': XGBClassifier(
        n_estimators=1500, learning_rate=0.015, max_depth=5,
        subsample=0.8, colsample_bytree=0.7,
        reg_alpha=0.5, reg_lambda=2.0,
        booster='dart', rate_drop=0.1,
        tree_method='hist',
        enable_categorical=True,
        early_stopping_rounds=100,
        eval_metric='auc', random_state=42
    )
}

# Inicializamos arrays para las predicciones OOF y test de cada modelo
oof_preds = {name: np.zeros(len(train_df)) for name in base_models}
test_preds = {name: np.zeros(len(test_df)) for name in base_models}

# ============================================================
# 5. VALIDACIÓN CRUZADA 10‑FOLD (GPU permite más folds)
# ============================================================
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

for fold, (train_idx, val_idx) in enumerate(skf.split(train_df, y)):
    print(f'\n========== FOLD {fold+1} ==========')

    X_train = train_df.iloc[train_idx].copy()
    X_val = train_df.iloc[val_idx].copy()
    y_train = y.iloc[train_idx]
    y_val = y.iloc[val_idx]
    X_test_fold = test_df.copy()

    # 1. Target Encoding
    te = TargetEncoder(smooth="auto")
    te_cols = ['School', 'Position', 'Player_Type']
    te_feat_names = [f"{c}_te" for c in te_cols]
    X_train[te_feat_names] = te.fit_transform(X_train[te_cols], y_train)
    X_val[te_feat_names] = te.transform(X_val[te_cols])
    X_test_fold[te_feat_names] = te.transform(X_test_fold[te_cols])

    # 2. Escalado e Imputación
    scaler = StandardScaler()
    X_train[numeric_features] = scaler.fit_transform(X_train[numeric_features])
    X_val[numeric_features] = scaler.transform(X_val[numeric_features])
    X_test_fold[numeric_features] = scaler.transform(X_test_fold[numeric_features])

    imputer = IterativeImputer(max_iter=10, random_state=42)
    X_train[numeric_features] = imputer.fit_transform(X_train[numeric_features])
    X_val[numeric_features] = imputer.transform(X_val[numeric_features])
    X_test_fold[numeric_features] = imputer.transform(X_test_fold[numeric_features])

    # 3. Features posicionales
    for col in physical_tests:
        means = X_train.groupby('Position_Type')[col].mean()
        stds = X_train.groupby('Position_Type')[col].std().replace(0, 1e-6)
        pos_med = X_train.groupby('Position_Type')[col].median()
        
        X_train[f'{col}_z_pos'] = (X_train[col] - X_train['Position_Type'].map(means)) / X_train['Position_Type'].map(stds)
        X_val[f'{col}_z_pos']   = (X_val[col]   - X_val['Position_Type'].map(means))   / X_val['Position_Type'].map(stds)
        X_test_fold[f'{col}_z_pos'] = (X_test_fold[col] - X_test_fold['Position_Type'].map(means)) / X_test_fold['Position_Type'].map(stds)
        
        X_train[f'{col}_diff_pos'] = X_train[col] - X_train['Position_Type'].map(pos_med)
        X_val[f'{col}_diff_pos'] = X_val[col] - X_val['Position_Type'].map(pos_med)
        X_test_fold[f'{col}_diff_pos'] = X_test_fold[col] - X_test_fold['Position_Type'].map(pos_med)

        for df_temp in [X_val, X_test_fold]:
            df_temp[f'{col}_z_pos'].fillna(0, inplace=True)
            df_temp[f'{col}_diff_pos'].fillna(0, inplace=True)

    # 4. Frecuencias y categorías
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
    # ENTRENAMIENTO DE CADA VARIANTE
    # ========================================================
    # Calcular scale_pos_weight para XGB (balanceo interno)
    sw = (len(y_train) - sum(y_train)) / sum(y_train)

    for name, model in base_models.items():
        # Clonar el modelo para no sobrescribir sus parámetros originales
        fold_model = model.__class__(**model.get_params())

        # Ajustes específicos por modelo
        if name.startswith('LGB'):
            fold_model.fit(X_train, y_train,
                           eval_set=[(X_val, y_val)],
                           callbacks=[early_stopping(100, verbose=False)])
        elif name.startswith('CAT'):
            fold_model.fit(X_train, y_train,
                           eval_set=(X_val, y_val),
                           cat_features=categorical_features,
                           use_best_model=True)
        elif name.startswith('XGB'):
            fold_model.set_params(scale_pos_weight=sw)
            fold_model.fit(X_train, y_train,
                           eval_set=[(X_val, y_val)], verbose=False)

        # Predicciones
        pred_val = fold_model.predict_proba(X_val)[:, 1]
        pred_test = fold_model.predict_proba(X_test_fold)[:, 1]

        oof_preds[name][val_idx] = pred_val
        test_preds[name] += pred_test / skf.n_splits

        print(f"   {name}: AUC={roc_auc_score(y_val, pred_val):.4f}")

# ============================================================
# 6. META‑MODELO (LogisticRegression con búsqueda de C)
# ============================================================
oof_matrix = np.column_stack([oof_preds[n] for n in base_models])
test_matrix = np.column_stack([test_preds[n] for n in base_models])

# Búsqueda del mejor C para evitar sobreajuste en el stacking
param_grid = {'C': [0.01, 0.1, 1.0, 10.0, 100.0]}
meta_cv = GridSearchCV(
    LogisticRegression(penalty='l2', solver='liblinear', random_state=42),
    param_grid, cv=5, scoring='roc_auc'
)
meta_cv.fit(oof_matrix, y)
meta = meta_cv.best_estimator_

final_oof = meta.predict_proba(oof_matrix)[:, 1]
final_score = roc_auc_score(y, final_oof)

# Pesos interpretables (coeficientes absolutos normalizados)
weights = np.abs(meta.coef_[0])
weights /= weights.sum()

print('\n===============================')
print('PESOS FINALES:')
for name, w in zip(base_models.keys(), weights):
    print(f'   {name}: {w:.3f}')
print(f'FINAL ROC AUC (CV): {final_score:.5f}')
print('===============================')

# ============================================================
# 7. EXPORTAR SUBMISSION
# ============================================================
final_test_preds = meta.predict_proba(test_matrix)[:, 1]
final_test_preds = np.clip(final_test_preds, 0.0, 1.0)

submission = pd.DataFrame({
    'Id': test_ids,
    'Drafted': final_test_preds
})

OUTPUT_PATH = 'data/submission_gpu_diverse_ensemble.csv'
submission.to_csv(OUTPUT_PATH, index=False)
print(f'\nSubmission guardada en:\n{OUTPUT_PATH}')