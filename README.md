# NFL Draft Prediction — High-Accuracy GPU Ensemble 

This repository contains the advanced machine learning pipeline developed for the GCI World Cup 2026 In-Class Competition hosted by the University of Tokyo.

## 🚀 Architecture Overview
The solution implements a highly robust, stabilized ensemble designed to handle tabular noise and prevent leaderboard overfitting:

1. **Advanced Feature Engineering:** Custom mass-to-motion formulas (`SpeedScore`, `ExplosionScore`, `StrengthScore`) along with positional normalization via Z-Score and historical cross-interactions (`{col}_x_Year`).
2. **Robust Categorical Encoding:** 5-fold internal Target Encoding paired with native frequency mapping.
3. **Validation Strategy:** Strict 10-Fold Stratified Cross-Validation to ensure local stability.
4. **Base Models:** Optimized instances of LightGBM, CatBoost (GPU), and XGBoost (GPU/CUDA).
5. **Within-Fold Rank Scaling:** Converts raw probabilities to normalized ranks partition by partition to neutralize calibration scale mismatches.
6. **Hyperplane Optimization:** A SciPy `minimize` block using Powell's method to dynamically search for the optimal soft-maxed combination weights based on global Out-of-Fold (OOF) ROC AUC.

## 📁 Repository Structure
```
├── data_structure.py         # Initial data structure and baseline experimentation
├── model.py                  # Main architecture: High-Accuracy GPU Ensemble (v5.6 Pro)
├── .gitignore                # Cleans repository from local logs, IDE configurations, and datasets
└── README.md                 # Project documentation
```

**🛠️ Requirements & Environment**

The code is optimized to run on a cloud environment with a CUDA-capable GPU (such as Google Colab).

The main dependencies are:

- pandas

- numpy

- scikit-learn

- scipy

- lightgbm

- catboost

- xgboost

**🏃 How to Run**

Ensure the dataset (train.csv and test.csv) is available in the relative path data/ or mounted via Google Drive at /content/drive/MyDrive/tokyoGCI_Competition/.

Run the main model script:
```
Bash
python model.py
The script will automatically execute the 10-fold cross-validation loop, search for the best rank-blending weights using Powell's optimization, and export the optimized predictions to submission_advanced_ensemble.csv.
