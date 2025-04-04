import streamlit as st
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, auc
from sklearn.tree import plot_tree

# Configuración de la app
st.set_page_config(page_title='Clasificación de Atletas', layout='wide')

# Cargar modelo y datos
@st.cache_resource
def cargar_modelo():
    with open('modelo.pkl', 'rb') as f:
        modelo, scaler, label_encoder = pickle.load(f)
    return modelo, scaler, label_encoder

modelo, scaler, label_encoder = cargar_modelo()

@st.cache_data
def cargar_datos():
    df = pd.read_csv('datos_atletas_nan_outliers.csv')
    variables = ['Volumen Sistolico', 'Peso', 'VO2 Max']
    df[variables] = scaler.transform(df[variables])
    return df, variables

def eliminar_outliers(df, variables):
    df_limpio = df.copy()
    for var in variables:
        Q1 = df[var].quantile(0.25)
        Q3 = df[var].quantile(0.75)
        IQR = Q3 - Q1
        limite_inferior = Q1 - 1.5 * IQR
        limite_superior = Q3 + 1.5 * IQR
        df_limpio = df_limpio[(df_limpio[var] >= limite_inferior) & (df_limpio[var] <= limite_superior)]
    return df_limpio

# Main y subpáginas
pagina = st.sidebar.selectbox("Selecciona una página:", ["Predicción", "Preprocesamiento y Métricas"])

df, variables = cargar_datos()
df_limpio = eliminar_outliers(df, variables)
X = df_limpio[variables]
y = label_encoder.transform(df_limpio['Tipo Atleta'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

if pagina == "Predicción":
    st.sidebar.title('Ajustes del Modelo')

    volumen_sistolico = st.sidebar.slider('Volumen Sistolico', 40, 150, 75)
    peso = st.sidebar.slider('Peso', 40, 120, 70)
    vo2_max = st.sidebar.slider('VO2 Max', 30, 80, 50)
    criterion = st.sidebar.radio('Criterio', ['gini', 'entropy'])
    max_depth = st.sidebar.slider('Profundidad del Árbol', 2, 4, 3)

    entrada = np.array([[volumen_sistolico, peso, vo2_max]])
    entrada = scaler.transform(entrada)

    modelo.set_params(criterion=criterion, max_depth=max_depth)
    modelo.fit(X_train, y_train)

    prediccion = modelo.predict(entrada)[0]
    prediccion_label = label_encoder.inverse_transform([prediccion])[0]

    st.title('Clasificación de Atletas')
    st.write('Esta aplicación predice el tipo de atleta según sus características fisiológicas.')

    st.subheader('Predicción:')
    st.write(f'El atleta pertenece a la categoría: **{prediccion_label}**')

    st.subheader('Visualización del Árbol de Decisión')
    fig, ax = plt.subplots(figsize=(10, 5))
    plot_tree(modelo.estimators_[0], feature_names=variables, class_names=label_encoder.classes_, filled=True, rounded=True, fontsize=8)
    st.pyplot(fig)

elif pagina == "Preprocesamiento y Métricas":
    st.title('Análisis de Preprocesamiento y Evaluación del Modelo')

    st.subheader('Datos Preprocesados')
    st.write('Visualización de los datos después del preprocesamiento:')
    st.dataframe(df_limpio)

    st.subheader('Histogramas de Distribución')
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for i, var in enumerate(variables):
        sns.histplot(df_limpio[var], kde=True, ax=ax[i])
        ax[i].set_title(f'Distribución de {var}')
    st.pyplot(fig)

    st.subheader('Diagramas Boxplot (Sin Outliers)')
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    sns.boxplot(data=df_limpio[variables], ax=ax2)
    st.pyplot(fig2)

    st.subheader('Balance de Clases')
    fig4, ax4 = plt.subplots()
    clase_counts = pd.Series(y).value_counts().sort_index()
    ax4.bar(label_encoder.classes_, clase_counts, color='skyblue')
    ax4.set_title("Distribución de Clases")
    ax4.set_ylabel("Número de Instancias")
    st.pyplot(fig4)

    st.subheader('Evaluación del Modelo')
    y_pred = modelo.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    st.write(f"**Accuracy:** {acc:.4f}")
    st.json(report)

    st.subheader('Curva ROC-AUC')
    y_pred_proba = modelo.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    fig3, ax3 = plt.subplots()
    ax3.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    ax3.plot([0, 1], [0, 1], color='gray', linestyle='--')
    ax3.set_xlabel('Tasa de Falsos Positivos')
    ax3.set_ylabel('Tasa de Verdaderos Positivos')
    ax3.set_title('Curva ROC-AUC')
    ax3.legend(loc='lower right')
    st.pyplot(fig3)
