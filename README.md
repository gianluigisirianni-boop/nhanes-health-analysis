# NHANES 2021-2023 — Health Data Analysis

Analisi del dataset NHANES (CDC, ciclo agosto 2021 - agosto 2023):
11.933 partecipanti, 1.112 variabili sanitarie, demografiche e di stile di vita.

## Dataset

- Fonte: Kaggle — `rileyzurrin/nhanes-2021-2023-public-data-and-codebooks`
- `data/participants.csv` = dati dei partecipanti
- `DATA_DICTIONARY.csv` = dizionario delle colonne (3.664 righe di metadati:
  ogni codice criptico tipo `DIQ010` ha la sua etichetta umana)

## Domande di ricerca

1. **Prevedere il diabete** (`DIQ010` — "il dottore ti ha detto che hai il diabete?")
2. **Prevedere la pressione alta** (`BPQ020` — "ti è mai stata detta pressione alta?")
3. **Driver del BMI** (`BMXBMI` — Body Mass Index kg/m²) — analizzato in EDA,
   non modellato: con queste feature il modello sarebbe circolare (BMI ≈ peso/altezza)

## Il ragionamento, passo per passo

### 1. Mai toccare i dati alla cieca
Prima si legge il dizionario, poi i dati. Su 1.112 colonne, caricare tutto e
"vedere cosa succede" non è una strategia: si cercano le colonne per concetto
("diabetes", "blood pressure", "smoke"...) con filtri sulle etichette.

### 2. dropna() sull'intero dataset = disastro
`df.dropna()` su NHANES restituisce **0 righe**: su 1.112 colonne, ogni
partecipante ha almeno un valore mancante. I missing si gestiscono solo sulle
colonne rilevanti (`dropna(subset=[...])`), dopo averle scelte.

### 3. Gli skip pattern non sono errori
Colonne come "Age when first told you had diabetes" hanno il 91% di NaN perché
la domanda viene fatta **solo a chi ha il diabete**. Quei NaN significano
"domanda non applicabile", non "dato perso". Criterio adottato: escluse le
colonne con meno di ~3.000 risposte, salvo motivi specifici.

### 4. I codici speciali NHANES si travestono da numeri
- Domande Sì/No: `1=Yes, 2=No`, ma anche `3=Borderline`, `7=Refused`, `9=Don't know`
- Misure numeriche: `9999` = non risposto (un "max sedentarietà" di 9999 minuti
  = 6,9 giorni seduti: impossibile, è un codice!)
- Soluzione: `map({1:1, 2:0})` trasforma automaticamente 3/7/9 in NaN;
  `replace(9999, np.nan)` per le misure.

### 5. Target leakage: le feature che "barano"
Escluse da X le colonne che esistono **solo in conseguenza della diagnosi**
("Age when first told", "Taking blood pressure medication"): usarle sarebbe
come copiare la risposta dal retro del compito. Tenuta invece "prediabetes":
è una diagnosi *precedente*, informazione legittima al momento della previsione.

### 6. Class imbalance e il paradosso dell'accuratezza
Diabete: ~1.100 Yes vs ~10.300 No (1:10). Un modello che dice sempre "No"
avrebbe accuracy 90%... trovando **zero** diabetici. Per questo:
- `stratify=y` nello split (stessa proporzione in train e test)
- `class_weight="balanced"` nel RandomForest
- si valuta con **recall e confusion matrix**, non solo accuracy

### 7. Correlazione lineare ≠ importanza nel modello
"Minutes sedentary" aveva correlazione ~0 col diabete nella heatmap, ma è la
**seconda feature** per importanza nel RandomForest: la correlazione vede solo
relazioni lineari, gli alberi trovano anche soglie e pattern non lineari.

## Risultati

### Modello 1 — Diabete (RandomForest, class_weight="balanced")

| Metrica | No diabete | Diabete |
|---|---|---|
| Precision | 0.89 | 0.22 |
| Recall | 0.76 | **0.42** |
| Support | 1089 | 169 |

Accuracy 0.72. Trova 71 diabetici su 169. Feature dominante: **età** (~63%),
poi sedentarietà e prediabete.

### Modello 2 — Pressione alta (RandomForest)

| Metrica | No ipertensione | Ipertensione |
|---|---|---|
| Precision | 0.73 | 0.49 |
| Recall | 0.75 | 0.46 |
| Support | 828 | 430 |

Accuracy 0.65. Classi più bilanciate (34% positivi), nessun peso necessario.

## Limiti dichiarati

- Le feature sono solo da **questionario** (età, genere, fumo, sedentarietà,
  prediabete). Il diabete si diagnostica con esami del sangue (HbA1c, glicemia)
  presenti nel dataset ma **volutamente esclusi**: usarli per prevedere
  "il dottore ti ha detto che hai il diabete" sarebbe leakage — l'esame *è*
  la diagnosi. Un 95% ottenuto così sarebbe finto.
- Recall ~0.42-0.46: modelli di screening grezzi, non clinici. Il valore del
  progetto è nella metodologia (pulizia, codici speciali, leakage, imbalance),
  non nelle performance.

## Bug memorabile

`x_tain` (typo) invece di `x_train`: Python non si lamenta perché la variabile
vecchia del modello precedente è ancora in memoria → modello allenato su righe
X e etichette y **misallineate**, accuracy falsata (0.59 invece di 0.65).
Lezione: in Jupyter i nomi delle variabili non si riciclano tra modelli diversi.

## File

- `nhanes_analysis.ipynb` — notebook completo con tutto il ragionamento
