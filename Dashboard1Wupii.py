import streamlit as st
import plotly.express as px
import pandas as pd 

# Carga de datos
@st.cache_resource
def load_data():
    df = pd.read_csv("Usuarios.csv")
    df["Usuario"] = df["Usuario"].str.lower()
    Lista = ["mini juego", "color presionado", "dificultad", "Juego"]  
    return df, Lista

df, Lista = load_data()

st.sidebar.title("Usuarios Wuupi")
View = st.sidebar.selectbox(label="Tipo de Análisis", options=["Extracción de Características", "Regresión Lineal"])


if View == "Extracción de Características":
    Variable_Cat = st.sidebar.selectbox(label="Variable Categórica", options=Lista)
    
    st.title("Comparación de Frecuencias por Usuario")

    tabla_usuario = df.groupby(["Usuario", Variable_Cat]).size().reset_index(name="frecuencia")
    pivot_tabla = tabla_usuario.pivot(index="Usuario", columns=Variable_Cat, values="frecuencia").fillna(0)
    pivot_prop = pivot_tabla.div(pivot_tabla.sum(axis=1), axis=0).round(3)

    # -------- GRÁFICAS --------
    # Gráfico de Barras Agrupadas
    st.subheader("Gráfico de Barras Agrupadas")
    fig1 = px.bar(tabla_usuario, x=Variable_Cat, y="frecuencia", color="Usuario", barmode="group")
    fig1.update_layout(height=400)
    st.plotly_chart(fig1, use_container_width=True)

    # Gráfico de Burbujas
    st.subheader("Gráfico de Burbujas")
    fig2 = px.scatter(tabla_usuario, x=Variable_Cat, y="Usuario", size="frecuencia", color="Usuario", size_max=40)
    fig2.update_layout(height=400)
    st.plotly_chart(fig2, use_container_width=True)

    # Gráfico de Barras Apiladas
    st.subheader("Gráfico de Barras Apiladas")
    fig3 = px.bar(tabla_usuario, x="Usuario", y="frecuencia", color=Variable_Cat, barmode="stack")
    fig3.update_layout(height=400)
    st.plotly_chart(fig3, use_container_width=True)

    # Tabla
    st.subheader("Tabla de Frecuencias (Conteo)")
    st.dataframe(pivot_tabla.style.background_gradient(cmap="YlOrRd"))

    # Heatmap
    st.subheader("Heatmap de Proporciones (Por Usuario)")
    fig5 = px.imshow(pivot_prop, text_auto=True, color_continuous_scale="Viridis", aspect="auto")
    st.plotly_chart(fig5, use_container_width=True)

    # Boxplot 
    if st.checkbox("Mostrar Boxplot"):
        cols_num = df.select_dtypes(include="number").columns.tolist()
        if cols_num:
            var_num = st.selectbox("Variable numérica", cols_num)
            fig_box = px.box(df.dropna(subset=[Variable_Cat, var_num]), 
                            x=Variable_Cat, y=var_num, color="Usuario", points="all")
            st.plotly_chart(fig_box, use_container_width=True)
        else:
            st.warning("No hay variables numéricas disponibles.")

elif View == "Regresión Lineal":
    st.title("Análisis de Regresión Lineal")

    numeric_df = df.select_dtypes(include=["number"])  # Filtrar solo columnas numéricas
    Lista_num = numeric_df.columns.tolist()            # Lista de nombres de columnas numéricas

    if len(Lista_num) < 2:
        st.warning("Se necesitan al menos dos variables numéricas para aplicar regresión.")
    else:
        Variable_y = st.sidebar.selectbox("Variable objetivo (Y)", options=Lista_num)
        posibles_x = [col for col in Lista_num if col != Variable_y]

        if posibles_x:
            Variable_x = st.sidebar.selectbox("Variable independiente (X)", options=posibles_x)

            from sklearn.linear_model import LinearRegression
            import numpy as np
# ------- REGRESIÓN SIMPLE -------
            X_simple = df[[Variable_x]].dropna()
            y_simple = df[Variable_y].loc[X_simple.index]

            model_simple = LinearRegression()
            model_simple.fit(X_simple, y_simple)
            y_pred_simple = model_simple.predict(X_simple)

            coef_deter_simple = model_simple.score(X_simple, y_simple)
            coef_correl_simple = np.sqrt(coef_deter_simple)

            st.subheader("Correlación Lineal Simple")
            st.write(f"Coeficiente de Correlación: **{coef_correl_simple:.3f}**")

            fig_simple = px.scatter(x=X_simple[Variable_x], y=y_simple,
                                    trendline="ols",
                                    labels={"x": Variable_x, "y": Variable_y},
                                    title="Modelo Lineal Simple")
            fig_simple.add_scatter(x=X_simple[Variable_x], y=y_pred_simple, mode='lines', name='Línea de regresión')
            st.plotly_chart(fig_simple, use_container_width=True)

        # ------- REGRESIÓN MÚLTIPLE -------
        st.subheader("Correlación Lineal Múltiple")
        Variables_x = st.sidebar.multiselect("Variables independientes (X)", options=posibles_x)

        if Variables_x:
            X_mult = df[Variables_x].dropna()
            y_mult = df[Variable_y].loc[X_mult.index]

            model_mult = LinearRegression()
            model_mult.fit(X_mult, y_mult)
            y_pred_mult = model_mult.predict(X_mult)

            coef_deter_mult = model_mult.score(X_mult, y_mult)
            coef_correl_mult = np.sqrt(coef_deter_mult)

            st.write(f"Coeficiente de Correlación Múltiple: **{coef_correl_mult:.3f}**")

            fig_mult = px.scatter(x=y_pred_mult, y=y_mult,
                                  labels={"x": "Valores predichos", "y": Variable_y},
                                  title="Modelo Lineal Múltiple")
            st.plotly_chart(fig_mult, use_container_width=True)
        else:
            st.info("Selecciona al menos una variable para la regresión múltiple.")


