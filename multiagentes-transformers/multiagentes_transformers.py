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

"""AGENTE 2: Entrenador"""

class AgenteEntrenador:
    def __init__(self):
        self.modelos = {
            "Regresión Logística": LogisticRegression(max_iter=1000),
            "Árbol de Decisión": DecisionTreeClassifier(random_state=42),
            "Random Forest": RandomForestClassifier(random_state=42)
        }

        self.resultados = {}
        self.mejor_modelo = None
        self.nombre_mejor_modelo = None
        self.X_test = None
        self.y_test = None

    def entrenar(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.30,
            random_state=42,
            stratify=y
        )

        self.X_test = X_test
        self.y_test = y_test

        for nombre, modelo in self.modelos.items():
            modelo.fit(X_train, y_train)

            predicciones = modelo.predict(X_test)

            self.resultados[nombre] = {
                "predicciones": predicciones
            }

        self.nombre_mejor_modelo = list(self.modelos.keys())[0]
        self.mejor_modelo = self.modelos[self.nombre_mejor_modelo]

        return self.resultados, self.nombre_mejor_modelo

"""Ejecutar AGENTE 2"""

entrenador = AgenteEntrenador()

resultados, mejor_modelo = entrenador.entrenar(X_limpio, y)

print("AGENTE 2 FINALIZADO")
print("Modelos entrenados:", list(resultados.keys()))

"""AGENTE 2: Entrenador con métricas"""

class AgenteEntrenador:
    def __init__(self):
        self.modelos = {
            "Regresión Logística": LogisticRegression(max_iter=1000),
            "Árbol de Decisión": DecisionTreeClassifier(random_state=42),
            "Random Forest": RandomForestClassifier(random_state=42)
        }

        self.resultados = {}
        self.mejor_modelo = None
        self.nombre_mejor_modelo = None
        self.X_test = None
        self.y_test = None

    def entrenar(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.30,
            random_state=42,
            stratify=y
        )

        self.X_test = X_test
        self.y_test = y_test

        for nombre, modelo in self.modelos.items():
            modelo.fit(X_train, y_train)
            predicciones = modelo.predict(X_test)

            if hasattr(modelo, "predict_proba"):
                probabilidades = modelo.predict_proba(X_test)[:, 1]
            else:
                probabilidades = predicciones

            self.resultados[nombre] = {
                "accuracy": accuracy_score(y_test, predicciones),
                "precision": precision_score(y_test, predicciones),
                "recall": recall_score(y_test, predicciones),
                "f1_score": f1_score(y_test, predicciones),
                "predicciones": predicciones,
                "probabilidades": probabilidades
            }

        self.nombre_mejor_modelo = max(
            self.resultados,
            key=lambda modelo: self.resultados[modelo]["f1_score"]
        )

        self.mejor_modelo = self.modelos[self.nombre_mejor_modelo]

        return self.resultados, self.nombre_mejor_modelo


"""Ejecutar AGENTE 2"""

entrenador = AgenteEntrenador()

resultados, mejor_modelo = entrenador.entrenar(X_limpio, y)

print("AGENTE 2 FINALIZADO")
print("Mejor modelo seleccionado:", mejor_modelo)

tabla_resultados = pd.DataFrame(resultados).T

display(tabla_resultados[["accuracy", "precision", "recall", "f1_score"]])

"""Gráfico de comparación de modelos"""

metricas = tabla_resultados[["accuracy", "precision", "recall", "f1_score"]]

metricas.plot(kind="bar", figsize=(10, 6))

plt.title("Comparación de modelos de Machine Learning")
plt.xlabel("Modelos")
plt.ylabel("Valor de la métrica")
plt.ylim(0, 1)
plt.grid(axis="y")
plt.show()


"""Curva ROC de los modelos"""

plt.figure(figsize=(8, 6))

for nombre, datos_modelo in resultados.items():
    fpr, tpr, _ = roc_curve(
        entrenador.y_test,
        datos_modelo["probabilidades"]
    )

    roc_auc = auc(fpr, tpr)

    plt.plot(
        fpr,
        tpr,
        label=f"{nombre} - AUC: {roc_auc:.3f}"
    )

plt.plot([0, 1], [0, 1], linestyle="--")

plt.title("Curva ROC de los modelos")
plt.xlabel("Tasa de falsos positivos")
plt.ylabel("Tasa de verdaderos positivos")
plt.legend()
plt.grid()
plt.show()


"""Matriz de confusión y reporte del mejor modelo"""

predicciones_mejor_modelo = resultados[mejor_modelo]["predicciones"]

matriz = confusion_matrix(entrenador.y_test, predicciones_mejor_modelo)
reporte = classification_report(entrenador.y_test, predicciones_mejor_modelo)

resultados[mejor_modelo]["matriz_confusion"] = matriz
resultados[mejor_modelo]["reporte"] = reporte

print("Mejor modelo:", mejor_modelo)

print("\nMatriz de confusión:")
print(matriz)

print("\nReporte de clasificación:")
print(reporte)

"""AGENTE 3: Comunicador interactivo con Transformers"""

class AgenteComunicadorTransformer:
    def __init__(self):
        modelo = "google/flan-t5-small"
        self.tokenizer = AutoTokenizer.from_pretrained(modelo)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(modelo)

    def crear_contexto(self, df, resultados, mejor_modelo, columnas_numericas, columnas_categoricas):
        mejor = resultados[mejor_modelo]

        contexto = f"""
Proyecto de Machine Learning multiagente.

Dataset:
- Nombre: clientes_sin_normalizar.csv
- Cantidad de registros: {df.shape[0]}
- Cantidad de columnas: {df.shape[1]}
- Variable objetivo: compro
- Columnas numéricas: {list(columnas_numericas)}
- Columnas categóricas: {list(columnas_categoricas)}
- Valores nulos totales antes de normalizar: {int(df.isnull().sum().sum())}

Agente 1:
- Limpia el dataset.
- Imputa valores nulos.
- Escala variables numéricas.
- Codifica variables categóricas con One-Hot Encoding.
- Genera dataset_limpio_normalizado.csv.

Agente 2:
- Entrena Regresión Logística, Árbol de Decisión y Random Forest.
- Divide los datos en entrenamiento y prueba.
- Evalúa los modelos con accuracy, precision, recall y F1 Score.
- Selecciona el mejor modelo según F1 Score.

Resultados:
- Mejor modelo: {mejor_modelo}
- Accuracy: {mejor["accuracy"]:.4f}
- Precision: {mejor["precision"]:.4f}
- Recall: {mejor["recall"]:.4f}
- F1 Score: {mejor["f1_score"]:.4f}
"""
        return contexto

    def responder_pregunta(self, pregunta, df, resultados, mejor_modelo, columnas_numericas, columnas_categoricas):
        contexto = self.crear_contexto(
            df,
            resultados,
            mejor_modelo,
            columnas_numericas,
            columnas_categoricas
        )

        prompt = f"""
Responde en español de forma clara y académica usando solamente la información del contexto.

Contexto:
{contexto}

Pregunta del usuario:
{pregunta}

Respuesta:
"""

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=180,
            temperature=0.7,
            do_sample=True
        )

        respuesta = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        return respuesta
    
    """Ejecutar AGENTE 3"""

comunicador = AgenteComunicadorTransformer()

pregunta_usuario = input("Ingrese una pregunta sobre el proyecto o el dataset: ")

respuesta = comunicador.responder_pregunta(
    pregunta=pregunta_usuario,
    df=df,
    resultados=resultados,
    mejor_modelo=mejor_modelo,
    columnas_numericas=columnas_numericas,
    columnas_categoricas=columnas_categoricas
)

print("\nRESPUESTA DEL AGENTE 3")
print("=" * 60)
print(respuesta)

"""Guardar métricas y reporte final"""

tabla_metricas = tabla_resultados[["accuracy", "precision", "recall", "f1_score"]]

tabla_metricas.to_csv("metricas_modelos.csv")

with open("respuesta_agente_comunicador.txt", "w", encoding="utf-8") as archivo:
    archivo.write(respuesta)

print("Métricas guardadas como metricas_modelos.csv")
print("Respuesta guardada como respuesta_agente_comunicador.txt")