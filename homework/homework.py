# flake8: noqa: E501
#
# En este dataset se desea pronosticar el default (pago) del cliente el próximo
# mes a partir de 23 variables explicativas.
#
#   LIMIT_BAL: Monto del credito otorgado. Incluye el credito individual y el
#              credito familiar (suplementario).
#         SEX: Genero (1=male; 2=female).
#   EDUCATION: Educacion (0=N/A; 1=graduate school; 2=university; 3=high school; 4=others).
#    MARRIAGE: Estado civil (0=N/A; 1=married; 2=single; 3=others).
#         AGE: Edad (years).
#       PAY_0: Historia de pagos pasados. Estado del pago en septiembre, 2005.
#       PAY_2: Historia de pagos pasados. Estado del pago en agosto, 2005.
#       PAY_3: Historia de pagos pasados. Estado del pago en julio, 2005.
#       PAY_4: Historia de pagos pasados. Estado del pago en junio, 2005.
#       PAY_5: Historia de pagos pasados. Estado del pago en mayo, 2005.
#       PAY_6: Historia de pagos pasados. Estado del pago en abril, 2005.
#   BILL_AMT1: Historia de pagos pasados. Monto a pagar en septiembre, 2005.
#   BILL_AMT2: Historia de pagos pasados. Monto a pagar en agosto, 2005.
#   BILL_AMT3: Historia de pagos pasados. Monto a pagar en julio, 2005.
#   BILL_AMT4: Historia de pagos pasados. Monto a pagar en junio, 2005.
#   BILL_AMT5: Historia de pagos pasados. Monto a pagar en mayo, 2005.
#   BILL_AMT6: Historia de pagos pasados. Monto a pagar en abril, 2005.
#    PAY_AMT1: Historia de pagos pasados. Monto pagado en septiembre, 2005.
#    PAY_AMT2: Historia de pagos pasados. Monto pagado en agosto, 2005.
#    PAY_AMT3: Historia de pagos pasados. Monto pagado en julio, 2005.
#    PAY_AMT4: Historia de pagos pasados. Monto pagado en junio, 2005.
#    PAY_AMT5: Historia de pagos pasados. Monto pagado en mayo, 2005.
#    PAY_AMT6: Historia de pagos pasados. Monto pagado en abril, 2005.
#
# La variable "default payment next month" corresponde a la variable objetivo.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# clasificación están descritos a continuación.
#
#
# Paso 1.
# Realice la limpieza de los datasets:
# - Renombre la columna "default payment next month" a "default".
# - Remueva la columna "ID".
# - Elimine los registros con informacion no disponible.
# - Para la columna EDUCATION, valores > 4 indican niveles superiores
#   de educación, agrupe estos valores en la categoría "others".
# - Renombre la columna "default payment next month" a "default"
# - Remueva la columna "ID".
#
#
# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.
#
#
# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Descompone la matriz de entrada usando PCA. El PCA usa todas las componentes.
# - Estandariza la matriz de entrada.
# - Selecciona las K columnas mas relevantes de la matrix de entrada.
# - Ajusta una maquina de vectores de soporte (svm).
#
#
# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use la función de precision
# balanceada para medir la precisión del modelo.
#
#
# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
#
#
# Paso 6.
# Calcule las metricas de precision, precision balanceada, recall,
# y f1-score para los conjuntos de entrenamiento y prueba.
# Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# Este diccionario tiene un campo para indicar si es el conjunto
# de entrenamiento o prueba. Por ejemplo:
#
# {'dataset': 'train', 'precision': 0.8, 'balanced_accuracy': 0.7, 'recall': 0.9, 'f1_score': 0.85}
# {'dataset': 'test', 'precision': 0.7, 'balanced_accuracy': 0.6, 'recall': 0.8, 'f1_score': 0.75}
#
#
# Paso 7.
# Calcule las matrices de confusion para los conjuntos de entrenamiento y
# prueba. Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'cm_matrix', 'dataset': 'train', 'true_0': {"predicted_0": 15562, "predicte_1": 666}, 'true_1': {"predicted_0": 3333, "predicted_1": 1444}}
# {'type': 'cm_matrix', 'dataset': 'test', 'true_0': {"predicted_0": 15562, "predicte_1": 650}, 'true_1': {"predicted_0": 2490, "predicted_1": 1420}}
#


import os
import pandas as pd
import gzip
import json
import pickle
import numpy as np


from sklearn.preprocessing import OneHotEncoder, StandardScaler 
from sklearn.compose import ColumnTransformer
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, balanced_accuracy_score, precision_score, recall_score, f1_score
from sklearn.feature_selection import SelectKBest, f_classif, f_regression 

"Paso 1: cargar y preprocesar datos"
def cargar_preprocesar_datos():
    train_dataset = pd.read_csv("files/input/train_data.csv.zip", index_col=False)
    test_dataset = pd.read_csv("files/input/test_data.csv.zip", index_col=False)

    train_dataset.rename(columns={"default payment next month": "default"}, inplace=True)
    test_dataset.rename(columns={"default payment next month": "default"}, inplace=True)

    train_dataset.drop(columns="ID", inplace=True)
    test_dataset.drop(columns="ID", inplace=True)

    train_dataset = train_dataset[train_dataset["EDUCATION"] != 0]
    test_dataset = test_dataset[test_dataset["EDUCATION"] != 0]

    train_dataset = train_dataset[train_dataset["MARRIAGE"] != 0]
    test_dataset = test_dataset[test_dataset["MARRIAGE"] != 0]

    train_dataset["EDUCATION"] = train_dataset["EDUCATION"].apply(lambda x: 4 if x > 4 else x)
    test_dataset["EDUCATION"] = test_dataset["EDUCATION"].apply(lambda x: 4 if x > 4 else x)

    return train_dataset, test_dataset

"Paso 2: División de los datos en conjuntos de entrenamiento y prueba"
def make_train_test_split(train_dataset, test_dataset):
    X_train = train_dataset.drop(columns="default")
    y_train = train_dataset["default"]

    X_test = test_dataset.drop(columns="default")
    y_test = test_dataset["default"]

    return X_train, y_train, X_test, y_test

"Paso 3: Cree un pipeline para el modelo de clasificación."
def make_pipeline(X_train):
    categorical_features = ["EDUCATION", "MARRIAGE", "SEX"]
    numerical_features = [col for col in X_train.columns if col not in categorical_features]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(), categorical_features),
            ('num',StandardScaler(), numerical_features),
        ],
        remainder='passthrough'
    )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ('pca', PCA()),
        ("selectKBest", SelectKBest(score_func=f_classif)),
        ('svc', SVC(random_state=42))
    ])

    return pipeline

"Paso 4: Optimización de los hiperparámetros"
def make_grid_search(pipeline, X_train, y_train):
    param_grid = {
    'pca__n_components':[20, 21],   # Cantidad de componentes principales a conservar en el paso PCA del pipeline.
    'selectKBest__k':[12],          # Características a mantener en el SelectKBest (selección de características).
    'svc__kernel': ['rbf'],         # (Radial Basis Function), kernel gaussiano más usado.
    'svc__gamma': [0.1],            # Gamma controla cuánto influye un ejemplo de entrenamiento sobre el modelo.
    
    }

    model = GridSearchCV(
    pipeline,
    param_grid,
    cv = 10,
    scoring="balanced_accuracy",
    n_jobs=-1,
    refit=True
    )
    model.fit(X_train, y_train)

    return model

"Paso 5: Guardar Modelo"
def save_estimator(estimator):
    models_path = "files/models"
    os.makedirs(models_path, exist_ok=True)
    print("Guardando modelo en:", models_path)  # 👈 agrega esto
    model_file = os.path.join(models_path, "model.pkl.gz")

    with gzip.open(model_file, "wb") as file:
        pickle.dump(estimator, file)   

"Paso 6 Y 7: Metricas, matriz de confusión y guardarlas en formato JSON"

def calc_metrics(model, X_train, y_train, X_test, y_test):

    # Cálculo de Predicciones
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    #Cálculo de Matriz de Confusión
    cm_train = confusion_matrix(y_train, y_train_pred)
    cm_test = confusion_matrix(y_test, y_test_pred)

    metricas =[
        {   # Train
            'type': 'metrics',
            'dataset': 'train',
            'precision': precision_score(y_train, y_train_pred, zero_division=0),
            'balanced_accuracy': balanced_accuracy_score(y_train, y_train_pred),
            'recall': recall_score(y_train, y_train_pred, zero_division=0),
            'f1_score': f1_score(y_train, y_train_pred, zero_division=0)
        },
        {   # Test
            'type': 'metrics',
            'dataset': 'test',
            'precision': precision_score(y_test, y_test_pred, zero_division=0),
            'balanced_accuracy': balanced_accuracy_score(y_test, y_test_pred),
            'recall': recall_score(y_test, y_test_pred, zero_division=0),
            'f1_score': f1_score(y_test, y_test_pred, zero_division=0)
        },
        {   # Matriz de Confusión Train
            'type': 'cm_matrix',
            'dataset': 'train',
            'true_0': {'predicted_0': int(cm_train[0, 0]), 'predicted_1': int(cm_train[0, 1])},
            'true_1': {'predicted_0': int(cm_train[1, 0]), 'predicted_1': int(cm_train[1, 1])}
        },
        {   # Matriz de Confusión Test
            'type': 'cm_matrix',
            'dataset': 'test',
            'true_0': {'predicted_0': int(cm_test[0, 0]), 'predicted_1': int(cm_test[0, 1])},
            'true_1': {'predicted_0': int(cm_test[1, 0]), 'predicted_1': int(cm_test[1, 1])}
        }

    ]

    return metricas

def save_metrics(metricas):
    output_path="files/output"
    os.makedirs(output_path, exist_ok=True)
    metrics_file = os.path.join(output_path, "metrics.json")
   
    # El test espera un archivo JSONL (una métrica por línea)
    with open(metrics_file, "w", encoding="utf-8") as f:
        for metric in metricas:
            json.dump(metric, f)
            f.write("\n")

    print("Métricas guardadas en:", metrics_file)


def main():
    try:
        train_dataset, test_dataset = cargar_preprocesar_datos()
        X_train, y_train, X_test, y_test = make_train_test_split(train_dataset, test_dataset)
        pipeline = make_pipeline(X_train)
        model = make_grid_search(pipeline, X_train, y_train)
        save_estimator(model)
        metricas = calc_metrics(model, X_train, y_train, X_test, y_test)
        save_metrics(metricas)
        print(model.best_estimator_)
        print(model.best_params_)
    except Exception as e:
        print("ERROR:", e)
    

if __name__ == "__main__":
    main()