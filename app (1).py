import pickle
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(page_title="Modelo Holt-Winters", layout="wide")

# ----------- PESTAÑAS PRINCIPALES -----------
tab1, tab2, tab3, tab4 = st.tabs(["Inicio", "Exploración", "Predicciones", "Validación del Modelo"])

# ----------- CARGA DEL MODELO Y LA SERIE -----------

# Cargar el modelo Holt-Winters entrenado
with open("modelo_holt.pkl", "rb") as f:
    modelo = pickle.load(f)

# Cargar la serie temporal transformada con índice TRIMESTRE
serie = pd.read_csv("serie_holt.csv", index_col='TRIMESTRE')
serie.index = pd.to_datetime(serie.index)  # Asegurar formato datetime

# Separar datos en entrenamiento y prueba
test_size = int(len(serie) * 0.2)
y_train = serie[:-test_size]['Global_Sales']
y_test = serie[-test_size:]['Global_Sales']

# ----------- PESTAÑA 4: Validación del Modelo -----------
with tab4:
    st.header("📉 Validación del Modelo Holt-Winters")
    st.write("En esta sección se muestra cómo se ajusta el modelo Holt-Winters a los datos históricos por trimestre.")

    # Crear DataFrame con valores reales y ajustados
    df_comparacion = pd.DataFrame({
        "Ventas Reales": y_train,
        "Ajuste del Modelo": modelo.fittedvalues
    }).dropna()

    # Mostrar tabla comparativa
    st.subheader("Comparación de ventas reales vs. ajustadas")
    st.dataframe(df_comparacion)

    # Gráfico de comparación
    st.subheader("Gráfico de Ajuste del Modelo")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df_comparacion.index, df_comparacion["Ventas Reales"], label="Ventas Reales", marker="o")
    ax.plot(df_comparacion.index, df_comparacion["Ajuste del Modelo"], label="Ajuste del Modelo", linestyle="--")
    ax.set_title("Comparación de Ventas Reales vs. Modelo Holt-Winters")
    ax.set_xlabel("Trimestre")
    ax.set_ylabel("Ventas Totales (Global_Sales)")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

    # Cálculo del MAE
    error_absoluto = abs(df_comparacion["Ventas Reales"] - df_comparacion["Ajuste del Modelo"])
    mae = error_absoluto.mean()
    st.write(f"📌 **Error absoluto medio (MAE):** {mae:.3f} millones de unidades")
