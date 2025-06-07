import streamlit as st
import pandas as pd
import pickle

# ----------------------------
# Cargar el modelo, diccionario y dataframe codificado
# ----------------------------
@st.cache_resource
def cargar_modelo_datos():
    with open("best_model.pkl", "rb") as file:
        data = pickle.load(file)
        modelo = data["model"]
        diccionario_inverso = data["label_encoder_mapping"]
        df = data["dataframe_codificado"]
        return modelo, diccionario_inverso, df

modelo, diccionario_inverso, df_codificado = cargar_modelo_datos()

# ----------------------------
# Configuración Streamlit
# ----------------------------
st.title("🔍 Predicción del Estado del Aprendiz")
st.write("Ingrese los datos para realizar una predicción basada en el modelo entrenado.")

# Entradas del usuario
edad = st.slider("Edad", 18, 100, 25)
cantidad_quejas = st.selectbox("Cantidad de quejas", list(range(0, 11)))
estrato = st.selectbox("Estrato socioeconómico", [1, 2, 3, 4, 5, 6])

# ----------------------------
# Botón para predecir
# ----------------------------
if st.button("Realizar predicción"):
    try:
        # Usar una fila del dataframe ya codificado como plantilla
        fila_base = df_codificado.drop(columns=["Estado Aprendiz"], errors="ignore").mean().to_frame().T

        # Reemplazar valores conocidos si existen
        for col in fila_base.columns:
            if "Edad" in col:
                fila_base[col] = edad
            elif "Cantidad de quejas" in col:
                fila_base[col] = cantidad_quejas
            elif "Estrato" in col:
                fila_base[col] = estrato

        # Hacer predicción
        pred = modelo.predict(fila_base)[0]
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
