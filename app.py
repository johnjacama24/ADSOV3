import streamlit as st
import pandas as pd
import pickle

@st.cache_resource
def cargar_modelo_datos():
    with open("best_model.pkl", "rb") as file:
        data = pickle.load(file)
        modelo = data["model"]
        diccionario_inverso = data["label_encoder_mapping"]
        df = data["dataframe_codificado"]
        return modelo, diccionario_inverso, df

modelo, diccionario_inverso, df_codificado = cargar_modelo_datos()

# Configuración
st.title("🔍 Predicción del Estado del Aprendiz")
st.write("Ingrese los datos para realizar una predicción basada en el modelo entrenado.")

# Entradas del usuario
edad = st.slider("Edad", 18, 100, 25)
cantidad_quejas = st.selectbox("Cantidad de quejas", list(range(0, 11)))
estrato = st.selectbox("Estrato socioeconómico", [1, 2, 3, 4, 5, 6])

# Botón
if st.button("Realizar predicción"):
    try:
        # Seleccionar una fila promedio (ya codificada correctamente)
        muestra = df_codificado.drop(columns=["Estado Aprendiz"]).mean(numeric_only=True).to_frame().T

        # Reemplazar valores
        for col in muestra.columns:
            if "Edad" in col:
                muestra[col] = edad
            if "Cantidad de quejas" in col:
                muestra[col] = cantidad_quejas
            if "Estrato" in col:
                muestra[col] = estrato

        # Predicción
        pred = modelo.predict(muestra)[0]
        resultado = diccionario_inverso.get(pred, f"Desconocido ({pred})")

        # Mostrar resultado
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
