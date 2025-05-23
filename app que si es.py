import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.cluster import KMeans

st.set_page_config(page_title="Análisis y Predicción de Ventas de Videojuegos", layout="wide")

# ----------- PESTAÑAS PRINCIPALES -----------
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Inicio", "Exploración", "Predicciones", "Validación del Modelo", "Modelos No Supervisados"])

# ----------- CARGA DE DATOS Y MODELOS -----------
with open("modelo_holt.pkl", "rb") as f:
    modelo_holt = pickle.load(f)

with open("modelo_kmeans.pkl", "rb") as f:
    kmeans = pickle.load(f)

with open("modelo_kproto.pkl", "rb") as f:
    kproto = pickle.load(f)

serie = pd.read_csv("serie_holt.csv", index_col='TRIMESTRE')
serie.index = pd.to_datetime(serie.index)

df_pca_final = pd.read_csv("df_pca_final_kmeans.csv")
df_pca = pd.read_csv("df_pca_final_kproto.csv")
df_1 = pd.read_csv("C:/Users/ju444/OneDrive/Documents/Proyecto Analitica 2025-1/Video_Games_Sales_as_at_22_Dec_2016 (1).csv")

# Preparar entrenamiento/prueba para Holt-Winters
test_size = int(len(serie) * 0.2)
y_train = serie[:-test_size]['Global_Sales']
y_test = serie[-test_size:]['Global_Sales']

# ----------- TAB 1: INICIO -----------
with tab1:
    st.title("🎮 Análisis y Predicción de Ventas de Videojuegos")
    st.markdown("""
        Bienvenido al dashboard interactivo para el análisis de ventas de videojuegos a nivel global.
        Aquí podrás explorar los datos, visualizar patrones, realizar predicciones y aplicar modelos de segmentación.
        
        **Contenido:**
        - 📊 Exploración de datos históricos.
        - 🔮 Predicción de ventas mediante Holt-Winters.
        - ✅ Validación del modelo con métricas.
        - 🧠 Clustering con KMeans y KPrototypes.
    """)

# ----------- TAB 2: EXPLORACIÓN -----------
with tab2:
    st.header("📈 Exploración de Datos")

    st.subheader("Vista general del dataset")
    st.dataframe(df_1.head())

    st.subheader("Distribución de Ventas Globales")
    fig_hist, ax_hist = plt.subplots()
    df_1['Global_Sales'].hist(bins=30, ax=ax_hist, color='skyblue', edgecolor='black')
    ax_hist.set_title("Distribución de Ventas Globales")
    ax_hist.set_xlabel("Ventas (millones)")
    ax_hist.set_ylabel("Frecuencia")
    st.pyplot(fig_hist)

    st.subheader("Ventas promedio por plataforma")
    ventas_plataforma = df_1.groupby('Platform')['Global_Sales'].mean().sort_values(ascending=False)
    st.bar_chart(ventas_plataforma)

# ----------- TAB 3: PREDICCIONES -----------
with tab3:
    st.header("🔮 Predicciones con Holt-Winters")
    st.write("Predicción de ventas trimestrales a futuro usando el modelo Holt-Winters.")

    # Predicción para los próximos trimestres
    steps = st.slider("Selecciona cuántos trimestres futuros quieres predecir:", 1, 12, 4)
    pred_hw = modelo_holt.forecast(steps)

    fig_pred, ax_pred = plt.subplots(figsize=(10, 5))
    ax_pred.plot(serie.index, serie['Global_Sales'], label="Histórico")
    future_index = pd.date_range(start=serie.index[-1], periods=steps+1, freq='Q')[1:]
    ax_pred.plot(future_index, pred_hw, label="Predicción", linestyle='--')
    ax_pred.set_title("Predicción de Ventas con Holt-Winters")
    ax_pred.set_xlabel("Trimestre")
    ax_pred.set_ylabel("Ventas Globales")
    ax_pred.legend()
    st.pyplot(fig_pred)

# ----------- TAB 4: VALIDACIÓN DEL MODELO -----------
with tab4:
    st.header("📉 Validación del Modelo Holt-Winters")
    st.subheader("Comparación de ventas reales vs. ajustadas")

    df_comparacion = pd.DataFrame({
        "Ventas Reales": y_train,
        "Ajuste del Modelo": modelo_holt.fittedvalues
    }).dropna()

    st.dataframe(df_comparacion)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df_comparacion.index, df_comparacion["Ventas Reales"], label="Ventas Reales", marker="o")
    ax.plot(df_comparacion.index, df_comparacion["Ajuste del Modelo"], label="Ajuste del Modelo", linestyle="--")
    ax.set_title("Comparación de Ventas Reales vs. Modelo Holt-Winters")
    ax.set_xlabel("Trimestre")
    ax.set_ylabel("Ventas Totales")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

    error_absoluto = abs(df_comparacion["Ventas Reales"] - df_comparacion["Ajuste del Modelo"])
    mae = error_absoluto.mean()
    ventas_reales = df_comparacion["Ventas Reales"].replace(0, 1e-10)
    mape = (error_absoluto / abs(ventas_reales)).mean() * 100

    st.write(f"📌 **MAE:** {mae:.3f} millones de unidades")
    st.write(f"📌 **MAPE:** {mape:.2f}%")

# ----------- TAB 5: MODELOS NO SUPERVISADOS -----------
with tab5:
    st.header("🧠 Modelos No Supervisados")

    st.subheader("Clustering con KMeans")
    fig1, ax1 = plt.subplots(figsize=(8,6))
    scatter1 = ax1.scatter(df_pca_final["PC1"], df_pca_final["PC2"], c=kmeans.labels_, cmap="viridis", alpha=0.6)
    ax1.set_xlabel("Componente Principal 1")
    ax1.set_ylabel("Componente Principal 2")
    ax1.set_title("Clustering KMeans")
    plt.colorbar(scatter1, ax=ax1, label="Cluster")
    st.pyplot(fig1)

    st.subheader("Clustering con KPrototypes")
    fig2, ax2 = plt.subplots(figsize=(8,6))
    scatter2 = ax2.scatter(df_pca["PC1"], df_pca["PC2"], c=df_pca["Cluster"], cmap="viridis", alpha=0.6)
    ax2.set_xlabel("Componente Principal 1")
    ax2.set_ylabel("Componente Principal 2")
    ax2.set_title("Clustering KPrototypes")
    plt.colorbar(scatter2, ax=ax2, label="Cluster")
    st.pyplot(fig2)

    st.subheader("Matriz de Correlación")
    num_cols = df_1.select_dtypes(include=['float64', 'int64'])
    corr_matrix = num_cols.corr()

    fig_corr, ax_corr = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', linewidths=0.5, ax=ax_corr)
    ax_corr.set_title("Matriz de Correlación entre Variables Numéricas")
    st.pyplot(fig_corr)
