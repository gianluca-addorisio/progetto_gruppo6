# FIA Earthquake Damage Predictor

Progetto finale del corso di **Fondamenti di Intelligenza Artificiale**.

L'obiettivo è predire il livello di danno subito dagli edifici in seguito al terremoto del Nepal del 2015, usando il dataset della competizione DrivenData:

**Richter's Predictor: Modeling Earthquake Damage**

---

## Obiettivo del progetto

Il progetto consiste nello sviluppo di una pipeline di Machine Learning per classificare gli edifici in base al livello di danno subito.

La variabile target è:

- `damage_grade`

Le classi sono:

- `1`: danno basso
- `2`: danno medio
- `3`: danno elevato

La metrica principale di valutazione è:

- **micro-F1 score**

Metriche di supporto:

- macro-F1
- weighted-F1

Per l'analisi qualitativa degli errori potranno essere usati anche:

- classification report
- confusion matrix

---

## Dataset

I dati originali sono contenuti nella cartella `data/raw/`.

```text
data/raw/
├── train_values.csv
├── train_labels.csv
├── test_values.csv
└── submission_format.csv
```

Descrizione:

- `train_values.csv`: feature degli edifici del training set;
- `train_labels.csv`: target `damage_grade`;
- `test_values.csv`: feature degli edifici del test set;
- `submission_format.csv`: formato richiesto per eventuale submission finale.

I file originali non devono essere modificati direttamente.

---

## Struttura del repository

```text
.
├── data/
│   ├── raw/          # dati originali
│   ├── interim/      # dati intermedi
│   └── processed/    # dati finali o trasformati
├── docs/             # documentazione e decision log
├── models/           # eventuali modelli salvati
├── notebooks/        # notebook di analisi e modellazione
├── outputs/          # metriche, figure e submission
├── preprocessing/    # pipeline parallela/sperimentale
├── src/              # codice stabile e riutilizzabile
├── requirements.txt
├── README.md
└── LICENSE
```

La pipeline stabile usata per la modellazione si trova in `src/`.

La cartella `preprocessing/` contiene lavoro parallelo/sperimentale e non rappresenta, al momento, la pipeline finale ufficiale.

---

## Pipeline stabile

La pipeline stabile è centralizzata in:

```text
src/
├── config.py
├── data_loader.py
├── evaluation.py
├── features.py
├── feature_selection.py
├── models.py
├── pipeline_training_model.py
├── preprocessing.py
└── utils.py
```

File principali:

- `src/data_loader.py`: caricamento dati e split;
- `src/features.py`: feature engineering compatto;
- `src/preprocessing.py`: preprocessing finale e pipeline sklearn;
- `src/evaluation.py`: metriche e valutazione;
- `src/models.py`: definizione o supporto ai modelli;
- `src/pipeline_training_model.py`: funzioni per training pipeline.

---

## Feature set finale

È stata adottata una versione compatta e interpretabile della feature matrix.

La matrice preparata contiene 17 feature prima dell'encoding:

```text
geo_level_1_id
geo_level_2_id
geo_level_3_id
count_floors_pre_eq
age
land_surface_condition
foundation_type
roof_type
ground_floor_type
other_floor_type
position
count_families
total_superstructure_count
total_secondary_use_count
has_fragile_material
has_engineered_structure
building_volume_proxy
```

---

## Feature engineering

La pipeline crea le seguenti feature aggregate:

```text
building_volume_proxy
total_superstructure_count
total_secondary_use_count
has_fragile_material
has_engineered_structure
```

Decisioni principali:

- `building_volume_proxy` sostituisce `area_percentage` e `height_percentage`;
- `total_secondary_use_count` sostituisce le feature originali `has_secondary_use_*`;
- `total_superstructure_count`, `has_fragile_material` e `has_engineered_structure` sostituiscono le feature originali `has_superstructure_*`;
- `age` viene mantenuta nella forma originale;
- `age_clipped` e `age_group` non vengono mantenute nella pipeline finale;
- `count_floors_pre_eq` e `count_families` vengono mantenute nella forma originale.

---

## Feature rimosse

Sono escluse dalla feature matrix finale:

```text
building_id
damage_grade
area_percentage
height_percentage
age_clipped
age_group
family_count_group
floor_count_group
plan_configuration
legal_ownership_status
has_secondary_use
has_secondary_use_*
has_superstructure_*
```

Motivazione sintetica:

- `building_id` è un identificativo tecnico;
- `damage_grade` è il target;
- `area_percentage` e `height_percentage` sono sostituite da `building_volume_proxy`;
- le feature derivate da `age`, `count_families` e `count_floors_pre_eq` non hanno mostrato vantaggio sufficiente rispetto alle variabili grezze;
- `plan_configuration` e `legal_ownership_status` sono state rimosse per bassa informatività e distribuzione fortemente sbilanciata;
- le feature originali di uso secondario e superstruttura sono state compresse in aggregati interpretabili.

---

## Encoding geografico

Le variabili geografiche sono codificate come numeri, ma rappresentano identificativi geografici. Non vengono quindi trattate come variabili numeriche continue.

Strategia finale:

```text
geo_level_1_id → one-hot encoding
geo_level_2_id → frequency encoding
geo_level_3_id → frequency encoding
```

Motivazione:

- `geo_level_1_id` ha cardinalità gestibile;
- `geo_level_2_id` e `geo_level_3_id` hanno cardinalità elevata;
- il one-hot completo delle geo granulari produrrebbe una matrice troppo ampia e sparsa;
- il frequency encoding conserva un segnale geografico compatto;
- il test diagnostico ha mostrato che la strategia ibrida è la migliore tra quelle confrontate.

Il frequency encoding viene fittato solo sul training set, per evitare leakage tra train e validation.

---

## Preprocessing finale

Il preprocessing finale è implementato in `src/preprocessing.py`.

Operazioni principali:

1. rimozione di identificativi e target se presenti;
2. creazione delle feature aggregate tramite `src/features.py`;
3. rimozione delle feature escluse o compresse;
4. one-hot encoding delle categoriche a bassa cardinalità;
5. one-hot encoding di `geo_level_1_id`;
6. frequency encoding di `geo_level_2_id` e `geo_level_3_id`;
7. passthrough o scaling delle feature numeriche a seconda del modello.

---

## Smoke test

Dopo l'aggiornamento della pipeline è stato eseguito uno smoke test con:

- split train/validation stratificato;
- `DecisionTreeClassifier(max_depth=12)`;
- feature set compatto;
- encoding geografico ibrido.

Risultati osservati:

```text
Feature prima del preprocessing: 17
Feature dopo preprocessing: 65
micro-F1:    circa 0.70265
macro-F1:    circa 0.62362
weighted-F1: circa 0.68989
```

Questo test conferma che la pipeline funziona correttamente e che la feature matrix finale mantiene una dimensionalità contenuta.

---

## Esecuzione

Creazione ambiente virtuale:

```bash
python -m venv .venv
```

Attivazione ambiente virtuale su macOS/Linux:

```bash
source .venv/bin/activate
```

Installazione dipendenze:

```bash
pip install -r requirements.txt
```

Smoke test del preprocessing:

```bash
python3 -m src.preprocessing
```

Test minimo con modello:

```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score

from src.config import RANDOM_STATE
from src.preprocessing import preprocess, split_train_validation, make_model_pipeline

X, y = preprocess()

X_train, X_valid, y_train, y_valid = split_train_validation(
    X,
    y,
    test_size=0.20,
    random_state=RANDOM_STATE,
)

model = DecisionTreeClassifier(
    max_depth=12,
    random_state=RANDOM_STATE,
)

pipe = make_model_pipeline(
    model=model,
    X=X_train,
    scale_numeric=False,
)

pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_valid)

print("micro-F1:", f1_score(y_valid, y_pred, average="micro"))
print("macro-F1:", f1_score(y_valid, y_pred, average="macro"))
print("weighted-F1:", f1_score(y_valid, y_pred, average="weighted"))
```

---

## Notebook

Notebook principali:

```text
notebooks/
├── 01_analisi_dati.ipynb
├── 02_qualita_dati.ipynb
├── 03_feature_comprehension.ipynb
├── 05_baseline_modeling.ipynb
└── 06_feature_analysis_selection.ipynb
```

Ruolo dei notebook:

- `01_analisi_dati.ipynb`: analisi esplorativa iniziale;
- `02_qualita_dati.ipynb`: qualità dati e sintesi data quality;
- `03_feature_comprehension.ipynb`: comprensione semantica delle feature;
- `05_baseline_modeling.ipynb`: baseline preliminare;
- `06_feature_analysis_selection.ipynb`: feature selection, test diagnostici e confronto strategie.

---

## Documentazione

Documenti principali:

```text
docs/
└── decision_log.md
```

`docs/decision_log.md` contiene il razionale delle principali decisioni metodologiche:

- target e metrica principale;
- feature mantenute e rimosse;
- feature engineering;
- encoding geografico;
- preprocessing finale;
- stato del progetto e prossimi step.

Il README fornisce invece una panoramica sintetica del progetto e istruzioni operative.

---

## Stato attuale

Stato aggiornato:

- repository organizzato su `main`, `dev` e branch personali/di integrazione;
- branch remoti obsoleti rimossi;
- feature set compatto implementato;
- preprocessing finale implementato in `src/preprocessing.py`;
- encoding geografico ibrido implementato;
- smoke test eseguito con successo;
- branch `data_preprocessing` di Claudia mantenuto come lavoro parallelo/sperimentale;
- branch `merge_preprocessing` creato per coordinare l'integrazione tra il lavoro su `gianluca` e il lavoro parallelo di Claudia.

---

## Prossimi step

- eseguire model comparison sulla pipeline aggiornata;
- valutare eventuale PCA come esperimento secondario;
- procedere con tuning e final evaluation;
- coordinare l'integrazione tra il preprocessing finale sviluppato nel branch `gianluca` e il lavoro parallelo di Claudia nel branch `data_preprocessing`, facendo confluire entrambi nel branch dedicato `merge_preprocessing`.

---

## Team

- Gianluca
- Nicola
- Claudia
- Mattia