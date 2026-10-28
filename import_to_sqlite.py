"""
NHANES 2021-2023 - Import dati puliti in SQLite.

Replica ESATTAMENTE la pipeline di caricamento/pulizia di generate_report.py
(stessa sorgente di verita') e salva il risultato in data/dati.db,
tabella "nhanes", per alimentare dashboard.py.

Uso:  python import_to_sqlite.py
"""

import sqlite3
from pathlib import Path

import kagglehub
from kagglehub import KaggleDatasetAdapter
import numpy as np
import pandas as pd

HANDLE = "rileyzurrin/nhanes-2021-2023-public-data-and-codebooks"

print("1/3  Caricamento dati e dizionario (kagglehub)...")
diz = kagglehub.dataset_load(KaggleDatasetAdapter.PANDAS, HANDLE, "DATA_DICTIONARY.csv")
part = kagglehub.dataset_load(KaggleDatasetAdapter.PANDAS, HANDLE, "data/participants.csv")

diz_part = diz[diz["output_file"] == "data/participants.csv"]
mappa = dict(zip(diz_part["output_column"], diz_part["label"]))
df_clean = part.rename(columns=mappa)

COLONNE = [
    "Doctor told you have diabetes", "Ever told you had high blood pressure",
    "Body Mass Index (kg/m**2)", "Age in years at screening", "Gender",
    "Smoked at least 100 cigarettes in life", "Minutes sedentary activity",
    "Ever told you have prediabetes", "Age when first told you had diabetes",
    "Told had high blood pressure - 2+ times", "Taking high blood pressure medication",
]
df_sub = df_clean[COLONNE].copy()

print("2/3  Ricodifica codici NHANES e pulizia (come generate_report.py)...")
mappa_binaria = {1: 1, 2: 0}
for col in ["Doctor told you have diabetes", "Ever told you had high blood pressure",
            "Smoked at least 100 cigarettes in life", "Ever told you have prediabetes"]:
    df_sub[col] = df_sub[col].map(mappa_binaria)
df_sub = df_sub.dropna(subset=["Doctor told you have diabetes",
                               "Ever told you had high blood pressure",
                               "Body Mass Index (kg/m**2)"])
df_sub["Minutes sedentary activity"] = df_sub["Minutes sedentary activity"].replace(9999, np.nan)

print("3/3  Scrittura in SQLite...")
BASE_DIR = Path(__file__).parent
(BASE_DIR / "data").mkdir(exist_ok=True)

conn = sqlite3.connect(BASE_DIR / "data" / "dati.db")
df_sub.to_sql("nhanes", conn, if_exists="replace", index=False)
check = pd.read_sql("SELECT COUNT(*) AS righe FROM nhanes", conn)
conn.close()

print(check)
print("OK - data/dati.db aggiornato, tabella 'nhanes'")
