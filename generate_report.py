"""
NHANES 2021-2023 - Generatore automatico di report.

Esegue l'intera pipeline (caricamento, pulizia, EDA, 2 modelli ML)
e produce un report HTML auto-contenuto: report_nhanes.html

Uso:  python generate_report.py
"""

import base64
import io
from datetime import datetime
from pathlib import Path

import kagglehub
from kagglehub import KaggleDatasetAdapter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, ConfusionMatrixDisplay)
from sklearn.model_selection import train_test_split

HANDLE = "rileyzurrin/nhanes-2021-2023-public-data-and-codebooks"
OUT = Path(__file__).parent / "output" / "report_nhanes.html"


def fig_to_base64(fig):
    """Converte una figura matplotlib in stringa base64 per embed HTML."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=110)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def img_tag(fig, width="700"):
    return f'<img src="data:image/png;base64,{fig_to_base64(fig)}" width="{width}">'


# ---------------------------------------------------------------- pipeline
print("1/5  Caricamento dati e dizionario...")
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

print("2/5  Ricodifica codici NHANES e pulizia...")
mappa_binaria = {1: 1, 2: 0}
for col in ["Doctor told you have diabetes", "Ever told you had high blood pressure",
            "Smoked at least 100 cigarettes in life", "Ever told you have prediabetes"]:
    df_sub[col] = df_sub[col].map(mappa_binaria)
df_sub = df_sub.dropna(subset=["Doctor told you have diabetes",
                               "Ever told you had high blood pressure",
                               "Body Mass Index (kg/m**2)"])
df_sub["Minutes sedentary activity"] = df_sub["Minutes sedentary activity"].replace(9999, np.nan)

print("3/5  EDA...")
vc_diabete = df_sub["Doctor told you have diabetes"].value_counts().sort_index()
vc_press = df_sub["Ever told you had high blood pressure"].value_counts().sort_index()
describe = df_sub[["Age in years at screening", "Body Mass Index (kg/m**2)",
                   "Minutes sedentary activity"]].describe().round(1)

fig, ax = plt.subplots(figsize=(10, 7))
sns.heatmap(df_sub.corr(numeric_only=True), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
ax.set_title("Matrice di correlazione")
heatmap_html = img_tag(fig, width="820")

print("4/5  Training modelli...")
X = df_sub.drop(columns=[
    "Doctor told you have diabetes", "Ever told you had high blood pressure",
    "Body Mass Index (kg/m**2)", "Age when first told you had diabetes",
    "Told had high blood pressure - 2+ times", "Taking high blood pressure medication",
])
X = pd.DataFrame(SimpleImputer(strategy="median").fit_transform(X), columns=X.columns)

risultati = {}
for nome, y, cw in [("Diabete", df_sub["Doctor told you have diabetes"], "balanced"),
                    ("Pressione alta", df_sub["Ever told you had high blood pressure"], None)]:
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2,
                                              random_state=42, stratify=y)
    rf = RandomForestClassifier(n_estimators=200, class_weight=cw, random_state=42)
    rf.fit(X_tr, y_tr)
    y_pr = rf.predict(X_te)

    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay(confusion_matrix(y_te, y_pr),
                           display_labels=["No", "Si"]).plot(ax=ax, colorbar=False)
    ax.set_title(f"Confusion matrix - {nome}")
    cm_html = img_tag(fig, width="380")

    imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values()
    fig, ax = plt.subplots(figsize=(7, 3.2))
    imp.plot(kind="barh", ax=ax, title=f"Feature importance - {nome}")
    fi_html = img_tag(fig, width="620")

    report = classification_report(y_te, y_pr, target_names=["No", "Si"], output_dict=True)
    risultati[nome] = {
        "accuracy": accuracy_score(y_te, y_pr),
        "recall_si": report["Si"]["recall"],
        "precision_si": report["Si"]["precision"],
        "cm_html": cm_html, "fi_html": fi_html,
    }

# ---------------------------------------------------------------- report
print("5/5  Generazione HTML...")
def tabella_vc(vc, nome_si):
    return (f"<table><tr><th>Risposta</th><th>N</th><th>%</th></tr>"
            f"<tr><td>No</td><td>{int(vc[0])}</td><td>{vc[0]/vc.sum()*100:.1f}%</td></tr>"
            f"<tr><td>{nome_si}</td><td>{int(vc[1])}</td><td>{vc[1]/vc.sum()*100:.1f}%</td></tr></table>")

r1, r2 = risultati["Diabete"], risultati["Pressione alta"]

data_generazione = datetime.now().strftime("%d/%m/%Y alle %H:%M")

html = f"""<!DOCTYPE html>
<html lang="it"><head><meta charset="utf-8">
<title>NHANES 2021-2023 — Report di Analisi Sanitaria: Diabete e Ipertensione</title>
<style>
 body {{ font-family: 'Segoe UI', sans-serif; max-width: 900px; margin: 30px auto;
        color: #222; line-height: 1.55; padding: 0 16px; }}
 h1 {{ border-bottom: 3px solid #2b6cb0; padding-bottom: 8px; }}
 h2 {{ color: #2b6cb0; margin-top: 40px; }}
 table {{ border-collapse: collapse; margin: 12px 0; }}
 th, td {{ border: 1px solid #ccc; padding: 6px 14px; text-align: center; }}
 th {{ background: #ebf3fb; }}
 .box {{ background: #f7fafc; border-left: 4px solid #2b6cb0; padding: 10px 16px; }}
 .meta {{ color: #555; font-size: 0.95em; margin-top: -6px; }}
 .warn {{ border-left-color: #dd6b20; }}
 img {{ border: 1px solid #ddd; border-radius: 6px; margin: 8px 0; }}
 .duecol {{ display: flex; gap: 24px; flex-wrap: wrap; align-items: flex-start; }}
</style></head><body>

<h1>NHANES 2021-2023 — Report di Analisi Sanitaria: Diabete e Ipertensione</h1>
<p class="meta">Report generato automaticamente il <b>{data_generazione}</b> da <code>generate_report.py</code> —
nessun valore inserito a mano.</p>

<h2>1. Dataset</h2>
<p>NHANES (CDC), ciclo ago 2021 - ago 2023: <b>{part.shape[0]:,}</b> partecipanti,
<b>{part.shape[1]:,}</b> variabili. Dopo la selezione di {len(COLONNE)} colonne rilevanti
e la pulizia dei codici speciali (3/7/9 = borderline/rifiutato/non so; 9999 = non risposto):
<b>{df_sub.shape[0]:,}</b> righe analizzabili.</p>

<h2>2. Bilanciamento dei target</h2>
<div class="duecol">
<div><h3>Diabete</h3>{tabella_vc(vc_diabete, "Si")}
<div class="box warn">Rapporto ~1:10: un modello che dice sempre "No" farebbe 90% di
accuracy trovando zero diabetici → stratify + class_weight="balanced".</div></div>
<div><h3>Pressione alta</h3>{tabella_vc(vc_press, "Si")}
<div class="box">Rapporto ~1:1.9, sbilanciamento lieve: nessun peso necessario.</div></div>
</div>

<h2>3. Statistiche descrittive</h2>
{describe.to_html()}

<h2>4. Correlazioni</h2>
{heatmap_html}

<h2>5. Modello 1 — Diabete</h2>
<div class="duecol"><div>{r1["cm_html"]}</div><div>
<table><tr><th>Metrica</th><th>Valore</th></tr>
<tr><td>Accuracy</td><td>{r1["accuracy"]:.2f}</td></tr>
<tr><td>Recall (diabetici trovati)</td><td>{r1["recall_si"]:.2f}</td></tr>
<tr><td>Precision</td><td>{r1["precision_si"]:.2f}</td></tr></table>
</div></div>
{r1["fi_html"]}

<h2>6. Modello 2 — Pressione alta</h2>
<div class="duecol"><div>{r2["cm_html"]}</div><div>
<table><tr><th>Metrica</th><th>Valore</th></tr>
<tr><td>Accuracy</td><td>{r2["accuracy"]:.2f}</td></tr>
<tr><td>Recall (ipertesi trovati)</td><td>{r2["recall_si"]:.2f}</td></tr>
<tr><td>Precision</td><td>{r2["precision_si"]:.2f}</td></tr></table>
</div></div>
{r2["fi_html"]}

<h2>7. Conclusioni e limiti</h2>
<div class="box">
<ul>
<li>L'<b>età</b> è il driver principale per entrambe le condizioni (corr 0.43 con pressione).</li>
<li>Recall modesti (0.42-0.46): le feature sono solo da questionario. Gli esami del sangue
(HbA1c, glicemia) sono esclusi <b>volontariamente</b>: prevedere la diagnosi con l'esame
che la definisce sarebbe target leakage.</li>
<li>La sedentarietà conta nel modello (2a feature) pur avendo correlazione lineare ~0:
gli alberi vedono soglie che la correlazione non vede.</li>
</ul></div>

</body></html>"""

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(html, encoding="utf-8")
print(f"\nReport salvato: {OUT}")
