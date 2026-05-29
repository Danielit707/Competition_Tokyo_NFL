import pandas as pd

# Cargar los datasets
train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')

print("=== ESTRUCTURA DE TRAIN ===")
print(f"Dimensiones: {train.shape}")
print("\nColumnas y tipos de datos:")
print(train.info())

print("\n=== VALORES NULOS EN TRAIN ===")
print(train.isnull().sum()[train.isnull().sum() > 0])

print("\n=== PRIMERAS FILAS DE TRAIN ===")
print(train.head())