import streamlit as st
import pandas as pd
import pickle
import re
import unicodedata

def normalizar_nombre(nombre):
    # Eliminar acentos y caracteres especiales
    nombre = unicodedata.normalize('NFKD', nombre).encode('ascii', 'ignore').decode('utf-8')
    nombre = re.sub(r'[^\w\s]', '', nombre)  # Elimina puntuación
    nombre = nombre.strip().lower().replace(" ", "_")  # Opcional: reemplazar espacios
    return nombre

@st.cache_resource
def cargar_modelo_datos():
    with open("best_model.pkl", "rb") as file:
        data = pickle.load(file)
        modelo = data["model"]
        diccionario_inverso = data["label_encoder_mapping"]
        df = data["dataframe_codificado"]
        return modelo, diccionario_inverso, df

modelo, diccionario_inverso, df_codificado = cargar_modelo_datos()

# Crear mapa de nombres normalizados
col_originales = df_codificado.drop(columns=["Estado Aprendiz"], errors="ignore").columns
col_normalizadas = [normalizar_nombre(col) for col in col_originales]
mapa_col = dict(zip(col_normalizadas, col_originales))

st.title("🔍 Predicción del Estado del Aprendiz")

edad = st.slider("Edad", 18, 100, 25)
cantidad_quejas = st.selectbox("Cantidad de quejas", list(range(0, 11)))
estrato = st.selectbox("Estrato socioeconómico", [1, 2, 3, 4, 5, 6])

if st.button("Realizar predicción"):
    try:
        muestra = df_codificado.drop(columns=["Estado Aprendiz"], errors="ignore").iloc[[0]].copy()

        # Reemplazar columnas numéricas modificadas
        for col in muestra.columns:
            nombre_normalizado = normalizar_nombre(col)
            if "edad" in nombre_normalizado:
                muestra[col] = edad
            elif "cantidad_de_quejas" in nombre_normalizado:
                muestra[col] = cantidad_quejas
            elif "estrato" in nombre_normalizado:
                muestra[col] = estrato

        pred = modelo.predict(muestra)[0]
        resultado = diccionario_inverso.get(pred, f"Desconocido ({pred})")

        st.subheader("📈 Resultado de la predicción:")
        st.success(f"Estado del aprendiz predicho: **{resultado}**")

        st.subheader("📌 Datos ingresados:")
        st.write({
            "Edad": edad,
            "Cantidad de quejas": cantidad_quejas,
            "Estrato socioeconómico": estrato
        })

    except Exception as e:
        st.error("❌ Error al hacer la predicción:")
        st.code(str(e))
