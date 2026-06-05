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

I dati originali sono contenuti nella cartella:

```text
data/raw/
```

Struttura attesa:

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

La pipeline ufficiale lavora a partire dai dati raw. Le cartelle `data/interim/` e `data/processed/` restano disponibili per eventuali dati intermedi o finali esportati, ma non sono necessarie per il flusso principale attuale.

---

## Struttura del repository

```text
.
├── data/
│   ├── raw/              # dati originali
│   ├── interim/          # eventuali dati intermedi
│   └── processed/        # eventuali dati finali o trasformati
├── docs/                 # documentazione e decision log
├── models/               # eventuali modelli salvati
├── notebooks/            # notebook di analisi e modellazione
├── outputs/              # metriche, figure e submission
├── preprocessing/        # lavoro parallelo/sperimentale
├── src/                  # codice stabile e riutilizzabile
├── requirements.txt
├── README.md
└── LICENSE
```

La pipeline ufficiale usata per la modellazione si trova in:

```text
src/preprocessing/
```

La cartella `preprocessing/` nella root contiene lavoro parallelo/sperimentale e non rappresenta la pipeline finale ufficiale.

Il file `src/preprocessing.py` resta presente come componente legacy/backward-compatible, ma la source of truth attuale è il package modulare `src/preprocessing/`.

---

## Codice principale

File e package principali:

```text
src/
├── config.py
├── data_loader.py
├── evaluation.py
├── features.py
├── feature_selection.py
├── featureselector.py
├── models.py
├── pipeline_training_model.py
├── preprocessing.py              # legacy/backward-compatible
├── preprocessing/                # pipeline modulare ufficiale
│   ├── __init__.py
│   ├── age_handler.py
│   ├── cleaner.py
│   ├── encoding.py
│   ├── outliers.py
│   ├── pipeline.py
│   └── scaling.py
└── utils.py
```

Ruolo dei file principali:

- `src/data_loader.py`: caricamento dati e split;
- `src/features.py`: feature engineering compatto;
- `src/preprocessing/`: preprocessing modulare ufficiale;
- `src/featureselector.py`: transformer sklearn-compatible per feature selection opzionale;
- `src/feature_selection.py`: metodi di scoring/ranking delle feature;
- `src/models.py`: definizione dei modelli;
- `src/pipeline_training_model.py`: training pipeline principale;
- `src/evaluation.py`: metriche e valutazione;
- `src/utils.py`: utility.

---

## Pipeline ufficiale

La pipeline ufficiale lavora sui dati raw e costruisce internamente il flusso:

```text
raw data
→ split train/validation
→ preprocessing
→ eventuale FeatureSelector
→ eventuale PCA
→ modello
→ metriche
```

Questa struttura serve a evitare data leakage: preprocessing, feature selection e PCA vengono fittati solo sul training set o sul fold di training durante la cross-validation.

La pipeline modulare è composta dai seguenti step:

1. `feature_engineering`: crea feature aggregate tramite `src/features.py`;
2. `DataCleaner`: rimuove identificativi, target accidentale, feature escluse e feature originali ormai compresse;
3. `AgeHandler`: gestisce il valore speciale `age = 995`;
4. `FrequencyEncoder`: applica frequency encoding a `geo_level_2_id` e `geo_level_3_id`;
5. `CategoricalEncoder`: applica one-hot encoding alle categoriche strutturali e a `geo_level_1_id`;
6. `NumericalScaler`: applicato solo quando richiesto, per esempio con PCA o modelli sensibili alla scala;
7. `FeatureSelector`: opzionale;
8. `PCA`: opzionale;
9. modello finale.

Funzioni principali:

- `get_preprocessing_pipeline()`: costruisce la pipeline di solo preprocessing;
- `make_complete_pipeline()`: costruisce preprocessing + eventuale feature selection + eventuale PCA + modello;
- `run_training_pipeline()`: esegue il flusso completo di training e valutazione.

---

## Feature set compatto

È stata adottata una versione compatta e interpretabile della feature matrix.

La rappresentazione compatta principale contiene le seguenti feature prima dell'encoding:

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

La pipeline modulare aggiornata crea inoltre:

```text
is_historic
```

`is_historic` indica i casi in cui `age = 995`.

---

## Feature engineering

La pipeline crea e mantiene le seguenti feature aggregate:

```text
building_volume_proxy
total_superstructure_count
total_secondary_use_count
has_fragile_material
has_engineered_structure
is_historic
```

Decisioni principali:

- `building_volume_proxy` sostituisce `area_percentage` e `height_percentage`;
- `total_secondary_use_count` sostituisce le feature originali `has_secondary_use_*`;
- `total_superstructure_count`, `has_fragile_material` e `has_engineered_structure` sintetizzano le feature originali `has_superstructure_*`;
- `age` viene mantenuta, ma il valore speciale `age = 995` viene gestito da `AgeHandler`;
- `is_historic` viene creata per conservare l'informazione associata ad `age = 995`;
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

## Feature Selection

La feature selection è integrata come step opzionale tramite:

```text
src/featureselector.py
```

Il selector viene inserito dopo preprocessing e prima del modello:

```text
preprocessing
→ FeatureSelector
→ model
```

Metodi supportati:

- `rf`: Random Forest importance;
- `xgb`: XGBoost importance;
- `ctb`: CatBoost importance;
- `corr_matrix`: correlazione con il target;
- `chi2`: Chi-square;
- `mu`: mutual information;
- `rlf`: ReliefF.

I metodi basati su XGBoost, CatBoost, ReliefF o TensorFlow sono da considerare opzionali/pesanti e dipendono dalla configurazione dell'ambiente.

Risultati osservati con RandomForest:

```text
Senza feature selection:
micro-F1    ≈ 0.6921
macro-F1    ≈ 0.6107
weighted-F1 ≈ 0.6722

Feature Selection RF, 30 feature:
micro-F1    ≈ 0.6859
macro-F1    ≈ 0.6020
weighted-F1 ≈ 0.6623

Feature Selection RF, 50 feature:
micro-F1    ≈ 0.6944
macro-F1    ≈ 0.6161
weighted-F1 ≈ 0.6764
```

Conclusione:

- la feature selection funziona ed è leak-safe se usata dentro la pipeline;
- 30 feature risultano troppo aggressive;
- 50 feature migliorano leggermente la baseline;
- al momento la feature selection non supera la configurazione PCA 40.

---

## PCA

La PCA è integrata come step opzionale nella pipeline.

Quando `use_pca=True`, la pipeline forza automaticamente lo scaling numerico:

```text
preprocessing
→ scaling
→ PCA
→ model
```

La PCA non viene applicata sui dati grezzi.

Risultati osservati con RandomForest:

```text
Baseline RF:
micro-F1    ≈ 0.6921
macro-F1    ≈ 0.6107
weighted-F1 ≈ 0.6722

RF + scaling only:
micro-F1    ≈ 0.6921
macro-F1    ≈ 0.6106
weighted-F1 ≈ 0.6722

RF + PCA 40:
micro-F1    ≈ 0.6983
macro-F1    ≈ 0.6223
weighted-F1 ≈ 0.6866

RF + FS 50 + PCA 40:
micro-F1    ≈ 0.6975
macro-F1    ≈ 0.6204
weighted-F1 ≈ 0.6856
```

Conclusione:

- il miglioramento non dipende dal solo scaling;
- PCA con 40 componenti è la migliore configurazione osservata finora con RandomForest;
- FS 50 + PCA 40 funziona, ma non supera PCA 40 senza feature selection.

---

## Sample weighting

`sample_weight` è disponibile come opzione nella training pipeline.

Decisione:

- non usarlo come default;
- mantenerlo come esperimento alternativo orientato alla macro-F1.

Risultati osservati con RandomForest:

```text
Senza sample_weight:
micro-F1    ≈ 0.6921
macro-F1    ≈ 0.6107
weighted-F1 ≈ 0.6722

Con sample_weight bilanciato:
micro-F1    ≈ 0.6478
macro-F1    ≈ 0.6280
weighted-F1 ≈ 0.6509
```

Conclusione:

- i pesi bilanciati migliorano la macro-F1;
- peggiorano sensibilmente la micro-F1;
- poiché la metrica principale è micro-F1, non sono usati come default.

---

## Modelli

La pipeline supporta:

- RandomForest;
- XGBoost;
- LightGBM.

RandomForest è il modello baseline stabile.

XGBoost e LightGBM sono gestiti come modelli opzionali: se le dipendenze native non sono disponibili, vengono saltati senza bloccare l'intera pipeline.

Nota per macOS:

- XGBoost e LightGBM possono richiedere `libomp`;
- se necessario, installare tramite Homebrew:

```bash
brew install libomp
```

---

## Tuning

Il tuning è previsto come step successivo.

Decisione metodologica:

- il tuning deve essere eseguito sopra la pipeline modulare aggiornata;
- non deve fare preprocessing, feature selection o PCA sull'intero dataset prima dello split;
- lo scoring principale dovrebbe essere coerente con la metrica primaria, quindi micro-F1;
- macro-F1 può restare metrica secondaria o obiettivo alternativo dichiarato.

Sequenza corretta:

```text
split/CV
→ fit preprocessing solo su train/fold
→ eventuale FeatureSelector
→ eventuale PCA
→ model
→ metriche
```

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

Esecuzione training pipeline baseline:

```bash
python3 -m src.pipeline_training_model
```

Esecuzione con PCA 40:

```bash
python3 - <<'PY'
from src.pipeline_training_model import run_training_pipeline

run_training_pipeline(
    use_pca=True,
    pca_n_components=40,
)
PY
```

Esecuzione con Feature Selection RF a 50 feature:

```bash
python3 - <<'PY'
from src.pipeline_training_model import run_training_pipeline

run_training_pipeline(
    feature_selection=True,
    fs_method="rf",
    fs_threshold=0.0,
    max_features_to_hold=50,
)
PY
```

Esecuzione con Feature Selection RF 50 + PCA 40:

```bash
python3 - <<'PY'
from src.pipeline_training_model import run_training_pipeline

run_training_pipeline(
    feature_selection=True,
    fs_method="rf",
    fs_threshold=0.0,
    max_features_to_hold=50,
    use_pca=True,
    pca_n_components=40,
)
PY
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

I notebook servono come supporto analitico e narrativo. La logica stabile finale deve stare in `src/`.

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
- feature selection;
- PCA;
- sample weighting;
- tuning;
- stato del progetto e prossimi step.

Il README fornisce invece una panoramica sintetica del progetto e istruzioni operative.

---

## Stato attuale

Stato aggiornato:

- feature set compatto implementato;
- pipeline ufficiale implementata nel package `src/preprocessing/`;
- `src/preprocessing.py` mantenuto come legacy/backward-compatible;
- encoding geografico ibrido implementato;
- `AgeHandler` integrato per gestire `age = 995`;
- `FeatureSelector` integrato come step opzionale e leak-safe;
- PCA integrata come step opzionale;
- configurazione RandomForest + PCA 40 identificata come miglior risultato osservato finora;
- `sample_weight` disponibile ma non usato come default;
- XGBoost e LightGBM gestiti come modelli opzionali;
- tuning ancora da riallineare alla pipeline aggiornata.

---

## Prossimi step

- riallineare il tuning alla pipeline modulare aggiornata;
- rieseguire model comparison finale su configurazioni comparabili;
- valutare se usare PCA 40 nella configurazione finale;
- salvare metriche finali in `outputs/metrics/`;
- produrre eventuale final evaluation/submission;
- aggiornare worklog condiviso;
- comunicare al gruppo lo stato aggiornato della pipeline.

---

## Team

- Gianluca
- Nicola
- Mattia
- Claudia
