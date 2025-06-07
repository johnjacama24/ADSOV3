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

modelo, diccionario_inverso, df = cargar_modelo_datos()

st.title("🔍 Predicción del Estado del Aprendiz")

edad = st.slider("Edad", 18, 100, 25)
cantidad_quejas = st.selectbox("Cantidad de quejas", list(range(0, 11)))
estrato = st.selectbox("Estrato socioeconómico", [1, 2, 3, 4, 5, 6])

if st.button("Realizar predicción"):
    try:
        # ✅ Usar una fila real del dataframe codificado original
        fila_original = df.drop(columns=["Estado Aprendiz"]).iloc[0].copy()

        # ✅ Convertir a DataFrame
        muestra = pd.DataFrame([fila_original])

        # ✅ Modificar solo variables que se deben personalizar
        for col in muestra.columns:
            if "Edad" in col:
                muestra[col] = edad
            elif "Cantidad de quejas" in col:
                muestra[col] = cantidad_quejas
            elif "Estrato" in col:
                muestra[col] = estrato

        # ✅ Predecir
        pred = modelo.predict(muestra)[0]
        resultado = diccionario_inverso.get(pred, f"Desconocido ({pred})")

        st.success(f"📈 Estado del aprendiz predicho: **{resultado}**")

    except Exception as e:
        st.error("❌ Error al hacer la predicción:")
        st.code(str(e))
