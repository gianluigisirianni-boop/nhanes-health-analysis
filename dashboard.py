import streamlit as st
import pandas as pd
import sqlite3

st.set_page_config(page_title="NHANES Health Dashboard", page_icon="🩺", layout="wide")

#connessione al databse
conn=sqlite3.connect("data/dati.db")
df= pd.read_sql("SELECT * FROM nhanes", conn)
conn.close()

#Titolo applicazione
st.title("Nhanes health-Dashboard")

#Colonne per le metriche
col1, col2, col3, col4=st.columns(4)

col1.metric("Età media (anni)",round(df["Age in years at screening"].mean(),2))
col2.metric("Sedentarietà media (min/giorno)", round(df["Minutes sedentary activity"].mean(),2))
col3.metric("% fumatori ( 100+ sigarette)", round(df["Smoked at least 100 cigarettes in life"].mean()*100,2))
col4.metric("% perdiabete", round(df["Ever told you have prediabetes"].mean()*100,2))


eta_min_df = int(df["Age in years at screening"].min())
eta_max_df = int(df["Age in years at screening"].max())

scelta = st.sidebar.slider("Seleziona un intervallo di età:", eta_min_df, eta_max_df, (eta_min_df, eta_max_df))
eta_min, eta_max = scelta
df_filtrato = df[(df["Age in years at screening"] >= eta_min) & (df["Age in years at screening"] <= eta_max)]

st.subheader("Età filtrate")
conteggi = df_filtrato["Age in years at screening"].value_counts().sort_index()
st.bar_chart(conteggi)

st.caption(f"Persone nell'intervallo selezionato: {len(df_filtrato)}")


