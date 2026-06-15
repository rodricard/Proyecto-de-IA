!pip install -U transformers accelerate sentencepiece -q

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc
)

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

"""Generar dataset CSV sin normalizar"""

np.random.seed(42)

n = 1000

datos = pd.DataFrame({
    "edad": np.random.randint(18, 70, n),
    "ingresos_mensuales": np.random.randint(1500000, 12000000, n),
    "horas_redes_sociales": np.random.uniform(0, 8, n).round(2),
    "visitas_web": np.random.randint(0, 30, n),
    "genero": np.random.choice(["Masculino", "Femenino", "Otro"], n),
    "ciudad": np.random.choice(["Asunción", "San Lorenzo", "Luque", "Capiatá", "Fernando de la Mora"], n),
    "tipo_cliente": np.random.choice(["Nuevo", "Frecuente", "Premium"], n),
    "dispositivo": np.random.choice(["Celular", "Computadora", "Tablet"], n)
})

probabilidad_compra = (
    (datos["ingresos_mensuales"] > 5000000).astype(int) +
    (datos["visitas_web"] > 10).astype(int) +
    (datos["horas_redes_sociales"] > 3).astype(int) +
    (datos["tipo_cliente"] == "Premium").astype(int)
)

datos["compro"] = np.where(probabilidad_compra >= 2, 1, 0)

for columna in ["edad", "ingresos_mensuales", "horas_redes_sociales", "genero", "ciudad", "tipo_cliente"]:
    indices_nulos = np.random.choice(datos.index, size=40, replace=False)
    datos.loc[indices_nulos, columna] = np.nan

datos.to_csv("clientes_sin_normalizar.csv", index=False)

datos.head()

"""Verificar datos faltantes"""

df = pd.read_csv("clientes_sin_normalizar.csv")

print("Tamaño del dataset:")
print(df.shape)

print("\nPrimeras filas:")
display(df.head())

print("\nValores nulos por columna:")
print(df.isnull().sum())

"""AGENTE 1: Normalizador"""

class AgenteNormalizador:
    def __init__(self, target):
        self.target = target
        self.preprocesador = None

    def procesar(self, df):
        X = df.drop(columns=[self.target])
        y = df[self.target]

        columnas_numericas = X.select_dtypes(include=["int64", "float64"]).columns
        columnas_categoricas = X.select_dtypes(include=["object"]).columns

        transformador_numerico = Pipeline(steps=[
            ("imputador", SimpleImputer(strategy="mean")),
            ("escalador", StandardScaler())
        ])

        transformador_categorico = Pipeline(steps=[
            ("imputador", SimpleImputer(strategy="most_frequent")),
            ("codificador", OneHotEncoder(handle_unknown="ignore"))
        ])

        self.preprocesador = ColumnTransformer(
            transformers=[
                ("numericas", transformador_numerico, columnas_numericas),
                ("categoricas", transformador_categorico, columnas_categoricas)
            ]
        )

        X_limpio = self.preprocesador.fit_transform(X)

        return X_limpio, y, columnas_numericas, columnas_categoricas

"""Ejecutar AGENTE 1"""

normalizador = AgenteNormalizador(target="compro")

X_limpio, y, columnas_numericas, columnas_categoricas = normalizador.procesar(df)

print("AGENTE 1 FINALIZADO")
print("Columnas numéricas:", list(columnas_numericas))
print("Columnas categóricas:", list(columnas_categoricas))
print("Tamaño del dataset limpio:", X_limpio.shape)

"""Guardar el Dataset limpio"""

nombres_columnas = normalizador.preprocesador.get_feature_names_out()

df_limpio = pd.DataFrame(
    X_limpio.toarray() if hasattr(X_limpio, "toarray") else X_limpio,
    columns=nombres_columnas
)

df_limpio["compro"] = y.values

df_limpio.to_csv("dataset_limpio_normalizado.csv", index=False)

print("Dataset limpio guardado correctamente.")
display(df_limpio.head())
