import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from kmodes.kmodes import KModes

# Configuración de la página
st.set_page_config(page_title="App de Análisis y Predicción de Ventas", layout="wide")

# Cargar modelo de predicción Holt-Winters
fit = joblib.load("modelo_holt.pkl")

# Cargar datos originales
df = pd.read_csv(r"C:\Users\ju444\OneDrive\Documents\Proyecto Analitica 2025-1\Video_Games_Sales_as_at_22_Dec_2016 (1).csv")


# Preprocesamiento: generar columna de fecha desde el año de lanzamiento
df = df[df["Year_of_Release"].notna()].copy()
df["Year_of_Release"] = df["Year_of_Release"].astype(int)
df["Fecha"] = pd.to_datetime(df["Year_of_Release"].astype(str) + "-01-01", errors="coerce")

# Crear pestañas
tab1, tab2, tab3, tab4, tab5 = st.tabs(["🔮 Predicciones", "📊 Estadísticas Descriptivas", "🌎 Ventas por Región", "Validación del Modelo y Visualización del Error","Análisis de Clustering"])

# ----- PESTAÑA 1: Predicciones -----
# ----- PESTAÑA 1: Predicciones + Cargar nuevos datos -----
with tab1:
    st.header("🔮 Predicción de ventas por trimestres")

    trimestres = st.number_input("¿Cuántos trimestres deseas predecir?", min_value=1, max_value=200, value=4)

    if st.button("Predecir"):
        ultima_fecha = pd.to_datetime("2020-04-01")  # Ajusta según tu serie entrenada
        fechas_futuras = pd.date_range(start=ultima_fecha + pd.offsets.QuarterEnd(1), periods=trimestres, freq="Q")
        pronostico = fit.forecast(trimestres)

        resultado = pd.DataFrame({
            "Fecha": fechas_futuras,
            "Pronóstico": pronostico
        })

        st.write("### Resultados del pronóstico:")
        st.dataframe(resultado)

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(resultado["Fecha"], resultado["Pronóstico"], marker="o", linestyle="-", color="green")
        ax.set_title("Pronóstico de Ventas por Trimestre")
        ax.set_xlabel("Fecha")
        ax.set_ylabel("Ventas (en millones)")
        ax.set_ylim(resultado["Pronóstico"].min() * 0.95, resultado["Pronóstico"].max() * 1.05)
        ax.grid(True)
        st.pyplot(fig)

    st.markdown("---")
    st.header("📥 Agregar nuevos datos históricos")

    uploaded_file = st.file_uploader("Sube un nuevo archivo CSV con datos históricos", type=["csv"])

    if uploaded_file is not None:
        nuevos_datos = pd.read_csv(uploaded_file)
        st.write("Vista previa de los nuevos datos:")
        st.dataframe(nuevos_datos.head())

        if st.button("Agregar y actualizar dataset"):
            # Asegurarse de que la columna 'Year_of_Release' existe
            if "Year_of_Release" not in nuevos_datos.columns:
                st.error("El archivo debe contener una columna 'Year_of_Release'")
            else:
                nuevos_datos = nuevos_datos[nuevos_datos["Year_of_Release"].notna()].copy()
                nuevos_datos["Year_of_Release"] = nuevos_datos["Year_of_Release"].astype(int)
                nuevos_datos["Fecha"] = pd.to_datetime(nuevos_datos["Year_of_Release"].astype(str) + "-01-01", errors="coerce")

                # Leer el archivo actual
                df_actual = pd.read_csv(r"C:\Users\ju444\OneDrive\Documents\Proyecto Analitica 2025-1\Video_Games_Sales_as_at_22_Dec_2016 (1).csv")
                df_actual["Year_of_Release"] = df_actual["Year_of_Release"].fillna(0).astype(int)
                df_actual["Fecha"] = pd.to_datetime(df_actual["Year_of_Release"].astype(str) + "-01-01", errors="coerce")

                # Concatenar y eliminar duplicados
                df_actualizado = pd.concat([df_actual, nuevos_datos], ignore_index=True)
                df_actualizado.drop_duplicates(inplace=True)

                # Guardar el nuevo dataset
                df_actualizado.to_csv(r"C:\Users\ju444\OneDrive\Documents\Proyecto Analitica 2025-1\Video_Games_Sales_as_at_22_Dec_2016 (1).csv", index=False)
                st.success("✅ Datos añadidos correctamente. Por favor, reinicia la app para aplicar los cambios.")

                # Opción para reentrenar el modelo (solo si quieres automatizarlo más)
                # Podrías integrar aquí el reentrenamiento de Holt-Winters si lo deseas.


# ----- PESTAÑA 2: Estadísticas Descriptivas -----
with tab2:
    st.header("📊 Estadísticas Descriptivas del Dataset")

    st.subheader("Vista previa de los datos")
    st.dataframe(df.head())

    st.subheader("Descripción estadística")
    st.write(df.describe())
    st.write("Esta tabla muestra un resumen numérico de las variables del conjunto de datos, como el promedio, los valores máximos y mínimos, y los percentiles. Es útil para conocer la distribución de los datos.")

    st.subheader("Ventas por región")
    regiones = ["NA_Sales", "EU_Sales", "JP_Sales", "Other_Sales", "Global_Sales"]
    fig, ax = plt.subplots(figsize=(10, 5))
    df[regiones].sum().plot(kind="bar", ax=ax)
    ax.set_ylabel("Ventas en millones")
    ax.set_title("Ventas totales por región")
    st.pyplot(fig)
    st.write("Este gráfico compara las ventas totales de videojuegos en diferentes regiones: Norteamérica, Europa, Japón, Otros y Ventas Globales. De acuerdo con los datos, Norteamérica es la región con las mayores ventas, destacándose significativamente sobre las demás con una participación dominante en el mercado global. Esto refleja la gran demanda de videojuegos en esta región y su influencia en la industria. En contraste, Japón y la categoría de Otros presentan las ventas más bajas, lo que podría estar relacionado con diferentes factores como las preferencias culturales o la oferta limitada de ciertos géneros en estas regiones. Aunque Japón sigue siendo un mercado importante en términos de desarrollo de videojuegos, su volumen de ventas globales es menor comparado con Norteamérica y Europa.")

    st.subheader("Géneros más vendidos")
    top_generos = df.groupby("Genre")["Global_Sales"].sum().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(10, 5))
    top_generos.plot(kind="bar", ax=ax)
    ax.set_ylabel("Ventas globales")
    ax.set_title("Ventas globales por género")
    st.pyplot(fig)
    st.write("Este gráfico proporciona una visión clara sobre los géneros de videojuegos que han generado mayores ventas a nivel global, lo cual es esencial para entender las tendencias y preferencias del mercado. El género de acción lidera las ventas con un impresionante total superior a los 1500 millones de unidades, lo que refleja su popularidad y la continua demanda de títulos intensos y dinámicos. En segundo lugar, se encuentran los géneros de deportes y disparos, que también muestran un rendimiento fuerte, con ventas considerables en ambas categorías, señalando la preferencia por experiencias competitivas y de acción. Por otro lado, el género de estrategia se encuentra entre los menos vendidos, lo que podría sugerir que, aunque tiene una base de jugadores leales, no alcanza el nivel de popularidad de otros géneros. Este análisis permite identificar áreas de oportunidad para el desarrollo de nuevos juegos y ajustar estrategias de marketing de acuerdo con los intereses actuales de los jugadores.")

# ----- PESTAÑA 3: Clustering de ventas por región -----
with tab3:
    st.header("🌍 Agrupación de Países por Ventas (KMeans)")

    # Crear tabla resumida con ventas por región
    regiones = ["NA_Sales", "EU_Sales", "JP_Sales", "Other_Sales"]
    df_region = pd.DataFrame(df[regiones].sum()).reset_index()
    df_region.columns = ["Region", "Ventas"]

    # Aplicar clustering
    k = st.slider("Selecciona el número de clusters", 2, 4, 2)
    kmeans = KMeans(n_clusters=k, random_state=42)
    df_region["Cluster"] = kmeans.fit_predict(df_region[["Ventas"]])

    st.subheader("Resultados del Clustering")
    st.dataframe(df_region)

    st.subheader("Ventas por Región (Clusterizado)")
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(data=df_region, x="Region", y="Ventas", hue="Cluster", dodge=False, ax=ax)
    plt.title("Clustering de regiones según ventas")
    st.pyplot(fig)
    import streamlit as st
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans
    from kmodes.kprototypes import KPrototypes
    import matplotlib.pyplot as plt

    @st.cache_data
    def preprocess_pca_kmeans(df, num_cols, k=2):
        df_clean = df[num_cols].dropna()
        scaler = StandardScaler()
        df_scaled = scaler.fit_transform(df_clean)

        pca = PCA(n_components=2)
        df_pca = pca.fit_transform(df_scaled)
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(df_pca)

        df_plot = pd.DataFrame(df_pca, columns=["PC1", "PC2"])
        df_plot["Cluster"] = clusters
        return df_plot

    def preprocess_kprototypes(df, cat_cols, num_cols, k=2):
        # Check for missing values and handle them
        df_kp = df[cat_cols + num_cols].dropna().copy()
    
        # Encode categorical columns using LabelEncoder
        for col in cat_cols:
            df_kp[col] = LabelEncoder().fit_transform(df_kp[col].astype(str))
    
        # Standardize numerical columns
        df_kp[num_cols] = StandardScaler().fit_transform(df_kp[num_cols])
    
        # Prepare data for KPrototypes
        matrix = df_kp[num_cols + cat_cols].values
    
        # Fit the KPrototypes model
        kproto = KPrototypes(n_clusters=k, init='Huang', random_state=42, n_jobs=-1)
        clusters = kproto.fit_predict(matrix, categorical=[len(num_cols) + i for i in range(len(cat_cols))])
    
        # Perform PCA for visualization
        pca = PCA(n_components=2)
        df_pca = pca.fit_transform(df_kp[num_cols])
    
        # Prepare the dataframe for visualization
        df_plot = pd.DataFrame(df_pca, columns=["PC1", "PC2"])
        df_plot["Cluster"] = clusters
    
        return df_plot
          



with tab4:
    st.header("Validación del Modelo y Visualización del Error")
    from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
    import numpy as np

    # Supón que entrenaste con esta columna:
    ventas_historicas = df.groupby("Fecha")["Global_Sales"].sum().sort_index()
    pred = fit.fittedvalues

    st.subheader("Evaluación del modelo Holt-Winters")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(ventas_historicas.index, ventas_historicas.values, label="Ventas reales")
    ax.plot(pred.index, pred.values, label="Predicciones Holt-Winters", linestyle="--")
    ax.set_title("Ajuste del modelo en datos históricos")
    ax.legend()
    st.pyplot(fig)

    # Ensure both arrays have the same length (as previously explained)
    if len(ventas_historicas) != len(pred):
        min_length = min(len(ventas_historicas), len(pred))
        ventas_historicas = ventas_historicas[:min_length]
        pred = pred[:min_length]

    # Optionally, remove NaN values if any
    mask = ~np.isnan(ventas_historicas) & ~np.isnan(pred)
    ventas_historicas = ventas_historicas[mask]
    pred = pred[mask]

    # Calculate MAPE and RMSE
    mape = mean_absolute_percentage_error(ventas_historicas, pred)
    rmse = np.sqrt(mean_squared_error(ventas_historicas, pred))
    st.markdown(f"- **MAPE**: {mape:.2%}")
    st.markdown(f"- **RMSE**: {rmse:.2f} millones")

with tab5:
    st.header("Análisis de Clustering")
    cluster_method = st.selectbox("Selecciona el tipo de clustering:", ["K-Means (PCA numéricas)", "K-Prototypes (Mixto)"])

    if cluster_method == "K-Means (PCA numéricas)":
        num_cols = ['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales', 'Global_Sales',
                    'Critic_Score', 'Critic_Count', 'User_Score', 'User_Count']
        df_plot = preprocess_pca_kmeans(df, num_cols)
        st.write("Visualización con PCA y clustering K-Means")
        fig, ax = plt.subplots()
        scatter = ax.scatter(df_plot["PC1"], df_plot["PC2"], c=df_plot["Cluster"], cmap="viridis", alpha=0.6)
        plt.colorbar(scatter, ax=ax, label="Cluster")
        st.pyplot(fig)

    elif cluster_method == "K-Prototypes (Mixto)":
        cat_cols = ["Platform", "Genre", "Publisher"]
        num_cols = ["Year_of_Release", "NA_Sales", "EU_Sales", "JP_Sales", "Other_Sales"]
        df_plot = preprocess_kprototypes(df, cat_cols, num_cols)
        st.write("Visualización con PCA (numéricas) y clustering K-Prototypes")
        fig, ax = plt.subplots()
        scatter = ax.scatter(df_plot["PC1"], df_plot["PC2"], c=df_plot["Cluster"], cmap="viridis", alpha=0.6)
        plt.colorbar(scatter, ax=ax, label="Cluster")
        st.pyplot(fig)