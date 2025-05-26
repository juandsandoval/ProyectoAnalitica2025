import pickle
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import seaborn as sns
from sklearn.cluster import KMeans


st.set_page_config(page_title="Análisis y Predicción de Ventas", layout="wide")

# ... resto del código, incluyendo importaciones, estilos, etc.


# --- CSS para Glassmorphism y estilo moderno ---
st.markdown("""
<style>
/* Fondo general con blur y transparencia */
[data-testid="stAppViewContainer"] {
    background: rgba(255, 255, 255, 0.15);
    backdrop-filter: blur(15px);
    -webkit-backdrop-filter: blur(15px);
    border-radius: 20px;
    margin: 20px;
    padding: 30px;
    box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
}

/* Contenedores tipo tarjeta con glass effect */
.glass-card {
    background: rgba(255, 255, 255, 0.25);
    border-radius: 20px;
    padding: 20px;
    box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.25);
    border: 1px solid rgba(255, 255, 255, 0.18);
    margin-bottom: 30px;
}

/* Botones con transición y colores */
.stButton > button {
    background: linear-gradient(135deg, #6e8efb, #a777e3);
    border: none;
    color: white;
    padding: 10px 24px;
    font-weight: 600;
    border-radius: 12px;
    transition: all 0.3s ease;
    box-shadow: 0 8px 15px rgba(110, 142, 251, 0.3);
    cursor: pointer;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #a777e3, #6e8efb);
    box-shadow: 0 15px 20px rgba(167, 119, 227, 0.5);
    transform: translateY(-3px);
}

/* Headers estilizados */
h1, h2, h3, h4 {
    color: #4b4276;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    text-shadow: 0 1px 2px rgba(255,255,255,0.7);
}
</style>
""", unsafe_allow_html=True)


# Inicializar estado de navegación entre pestañas
if 'active_tab' not in st.session_state:
    st.session_state.active_tab = 'Inicio'

# Función para cambiar de pestaña
def go_to_tab(tab_name):
    st.session_state.active_tab = tab_name

# Carga de datos y modelos compartidos
@st.cache_data
def cargar_datos():
    serie = pd.read_csv("serie_holt.csv", index_col='TRIMESTRE')
    serie.index = pd.to_datetime(serie.index)
    test_size = int(len(serie) * 0.2)
    y_train = serie[:-test_size]['Global_Sales']
    y_test = serie[-test_size:]['Global_Sales']
    return serie, y_train, y_test

@st.cache_resource
def cargar_modelos():
    with open("modelo_holt.pkl", "rb") as f:
        modelo_holt = pickle.load(f)
    with open("modelo_kmeans.pkl", "rb") as f:
        kmeans = pickle.load(f)
    with open("modelo_kproto.pkl", "rb") as f:
        kproto = pickle.load(f)
    return modelo_holt, kmeans, kproto

@st.cache_data
def cargar_datos_cluster():
    df_pca_final = pd.read_csv("df_pca_final_kmeans.csv")
    df_pca = pd.read_csv("df_pca_final_kproto.csv")
    df_original = pd.read_csv("Video_Games_Sales_as_at_22_Dec_2016 (1).csv")
    return df_pca_final, df_pca, df_original

# Cargar datos y modelos una vez
serie, y_train, y_test = cargar_datos()
modelo_holt, kmeans, kproto = cargar_modelos()
df_pca_final, df_pca, df_1 = cargar_datos_cluster()

# ----------- Pestañas personalizadas -----------

if st.session_state.active_tab == 'Inicio':
    with st.container():
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.title("🎮 Análisis y Predicción de Ventas de Videojuegos")
        st.markdown("Bienvenido a la aplicación de analítica de ventas. Usa los botones para navegar a cada sección:")
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("📊 Exploración"):
                go_to_tab("Exploración")
            if st.button("📈 Predicciones"):
                go_to_tab("Predicciones")
        with col2:
            if st.button("📉 Validación del Modelo"):
                go_to_tab("Validación")
            if st.button("🔍 Modelos No Supervisados"):
                go_to_tab("No Supervisado")
        st.markdown('</div>', unsafe_allow_html=True)

elif st.session_state.active_tab == 'Exploración':
    with st.container():
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.header("📈 Exploración de Datos")

        st.subheader("Vista general del dataset")
        st.dataframe(df_1.head())

        st.subheader("Segmentación Personalizada")

        columnas_segmento = ['Platform', 'Genre', 'Publisher', 'Rating', 'Developer', 'Year_of_Release']
        columna_segmento = st.selectbox("Selecciona una columna para segmentar:", columnas_segmento)

        columnas_metricas = ['Global_Sales', 'NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales',
                            'Critic_Score', 'User_Score']
        columna_metrica = st.selectbox("Selecciona la métrica a visualizar:", columnas_metricas)

        # Convertir User_Score a numérico si se selecciona
        df_filtrado = df_1[[columna_segmento, columna_metrica]].copy()
        if columna_metrica == 'User_Score':
            df_filtrado[columna_metrica] = pd.to_numeric(df_filtrado[columna_metrica], errors='coerce')

        # Eliminar filas con NaN en la métrica seleccionada
        df_filtrado = df_filtrado.dropna()

        # Agrupación y visualización
        top_categorias = df_filtrado.groupby(columna_segmento)[columna_metrica].mean().sort_values(ascending=False).head(15)
        st.bar_chart(top_categorias)

        if st.button("Volver al Inicio"):
            go_to_tab("Inicio")
        st.markdown('</div>', unsafe_allow_html=True)

elif st.session_state.active_tab == 'Predicciones':
    with st.container():
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.header("🔮 Predicciones con Holt-Winters")
        st.write("Predicción de ventas trimestrales a futuro usando el modelo Holt-Winters.")

        steps = st.slider("Selecciona cuántos trimestres futuros quieres predecir:", 1, 70, 4)
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

        if st.button("Volver al Inicio"):
            go_to_tab("Inicio")
        st.markdown('</div>', unsafe_allow_html=True)

elif st.session_state.active_tab == 'Validación':
    with st.container():
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
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

        if st.button("Volver al Inicio"):
            go_to_tab("Inicio")
        st.markdown('</div>', unsafe_allow_html=True)

elif st.session_state.active_tab == 'No Supervisado':
    with st.container():
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.header("🧠 Modelos No Supervisados")

        st.subheader("Clustering con KMeans")
        fig1, ax1 = plt.subplots(figsize=(8,6))
        scatter1 = ax1.scatter(df_pca_final["PC1"], df_pca_final["PC2"], c=kmeans.labels_, cmap="viridis", alpha=0.6)
        ax1.set_xlabel("PC1")
        ax1.set_ylabel("PC2")
        ax1.set_title("Clusters KMeans")
        st.pyplot(fig1)

        st.subheader("Clustering con KPrototypes")
        fig2, ax2 = plt.subplots(figsize=(8,6))
        scatter2 = ax2.scatter(df_pca["PC1"], df_pca["PC2"], c=kproto.labels_, cmap="plasma", alpha=0.6)
        ax2.set_xlabel("PC1")
        ax2.set_ylabel("PC2")
        ax2.set_title("Clusters KPrototypes")
        st.pyplot(fig2)

        if st.button("Volver al Inicio"):
            go_to_tab("Inicio")
        st.markdown('</div>', unsafe_allow_html=True)