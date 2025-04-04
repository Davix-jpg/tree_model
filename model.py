import pandas as pd
import numpy as np
import seaborn as sns
import pickle
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_curve, auc
from sklearn.preprocessing import LabelEncoder


df = pd.read_csv('datos_atletas_nan_outliers.csv')
print(df.head())
print(df.info())

#? Ahora detectamos y tratamos los datos atípicos, creando un histograma de estos para visualizar su distribución
# df.hist(figsize=(12, 8), bins=30)
# plt.tight_layout()
# plt.show()

#? Seleccionamos las variables que vamos a normalizar y creamos una lista
variables = ['Volumen Sistolico', 'Peso', 'VO2 Max']

#?Normalizamos las variables con normalización MinMax
scaler = MinMaxScaler()
df[variables] = scaler.fit_transform(df[variables])

#?Creamos histogramas de las variables normalizadas para comprobar que hemos eliminado los outliers
df[variables].hist(figsize=(12, 6), bins=30)
plt.tight_layout()
plt.show()

#? Creamos un diagrama boxplot para visualizar los outliers
# sns.boxplot(data=df[variables])
# plt.title("Boxplot de las variables seleccionadas")
# plt.show()

#? Filtramos los outliers:
df_limpio = df.copy()

for var in variables:
    Q1 = df[var].quantile(0.25)
    Q3 = df[var].quantile(0.75)
    IQR = Q3 - Q1
    limite_inferior = Q1 - 1.5 * IQR
    limite_superior = Q3 + 1.5 * IQR
    df_limpio = df_limpio[(df_limpio[var] >= limite_inferior) & (df_limpio[var] <= limite_superior)]
print(df_limpio.shape) 

#? Ahora volvemos a crear el diagrama boxplot para ver si los outliers se han eliminado
# sns.boxplot(data=df_limpio[variables])
# plt.title("Boxplot de las variables sin outliers")
# plt.show()

#? Declaramos las variables:
X = df_limpio[variables]
y = df_limpio['Tipo Atleta']  

#? Convertir la variable 'Tipo de atleta' a valores binarios (0 y 1)
le = LabelEncoder()
df_limpio['Tipo Atleta'] = le.fit_transform(df_limpio['Tipo Atleta'])

#? Definimos X e y
X = df_limpio[variables]
y = df_limpio['Tipo Atleta']  

#? Dividimos los datos en train, 80% y test 20%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

#? Entrenamos el modelo
modelo = RandomForestClassifier(n_estimators=100, random_state=42)
modelo.fit(X_train, y_train)

#? Realizamos las predicciones, tanto de la curva ROC-AUC, como las del modelo
y_pred = modelo.predict(X_test)
y_pred_proba = modelo.predict_proba(X_test)[:, 1] 

#? Realizamos la evaluación del modelo y pedimos las métricas de evaluación
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)  
roc_auc = auc(fpr, tpr)

#? Representamos la Curva ROC
plt.figure(figsize=(8,6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'Curva ROC (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Tasa de Falsos Positivos (FPR)')
plt.ylabel('Tasa de Verdaderos Positivos (TPR)')
plt.title('Curva ROC')
plt.legend(loc='lower right')
plt.show()

#? Ahora imprimimos la matriz de confusión y las métricas de evaluación:
print("Matriz de Confusión:")
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicción")
plt.ylabel("Real")
plt.title("Matriz de Confusión")
plt.show()

print("Reporte de Clasificación:")
print(classification_report(y_test, y_pred))

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

#? Guardamos el modelo y el scaler
with open('modelo.pkl', 'wb') as f:
    pickle.dump((modelo, scaler, le), f)

print("Modelo entrenado y guardado como 'modelo.pkl'")