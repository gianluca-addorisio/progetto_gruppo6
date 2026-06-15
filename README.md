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

Per l'analisi qualitativa degli errori possono essere usati anche:

- classification report
- confusion matrix

La scelta del modello finale viene guidata principalmente dalla micro-F1, coerentemente con la metrica della competizione.

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
├── preprocessing/        # lavoro parallelo/sperimentale o legacy
├── src/                  # codice stabile e riutilizzabile
├── requirements.txt
├── README.md
└── LICENSE
```

La pipeline ufficiale usata per la modellazione si trova in:

```text
src/preprocessing/
```

La cartella `preprocessing/` nella root contiene lavoro parallelo/sperimentale o legacy e non rappresenta la pipeline finale ufficiale.

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
├── hyperparameter_tuning.py
├── hyperparameter_tuning_feature_selection.py
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
- `src/models.py`: definizione dei modelli e degli ensemble;
- `src/pipeline_training_model.py`: training pipeline principale;
- `src/hyperparameter_tuning.py`: tuning dei modelli senza feature selection;
- `src/hyperparameter_tuning_feature_selection.py`: tuning con feature selection;
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
area_percentage
height_percentage
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
```

La pipeline modulare crea inoltre:

```text
is_historic
```

`is_historic` indica i casi in cui `age = 995`.

---

## Feature engineering

La pipeline crea e mantiene le seguenti feature aggregate:

```text
total_superstructure_count
total_secondary_use_count
has_fragile_material
has_engineered_structure
is_historic
```

Decisioni principali:

- `area_percentage` e `height_percentage` vengono mantenute come feature dimensionali originali;
- `building_volume_proxy` non viene mantenuta nella pipeline finale attuale perché ridondante rispetto ad `area_percentage` e `height_percentage`;
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
building_volume_proxy
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
- `building_volume_proxy` è ridondante rispetto ad `area_percentage` e `height_percentage`;
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

## Model comparison e risultati principali

La configurazione candidata finale usa:

```text
Modello: XGBoost
Feature selection: no
PCA: no
Tuning: no
Sample weighting: no
Feature dimensionali: original_dims
Split strategy: 2
```

Risultati della baseline avanzata senza feature selection, senza PCA e senza tuning:

```text
RandomForest      micro-F1 0.716160 | macro-F1 0.645689 | weighted-F1 0.704259
XGBoost           micro-F1 0.741889 | macro-F1 0.687747 | weighted-F1 0.735980
LightGBM          micro-F1 0.727615 | macro-F1 0.664767 | weighted-F1 0.719590
VotingEnsemble    micro-F1 0.735462 | macro-F1 0.673930 | weighted-F1 0.727158
StackingEnsemble  micro-F1 0.741851 | macro-F1 0.688311 | weighted-F1 0.736433
```

Decisione:

- `XGBoost` è il candidato finale attuale perché ottiene la micro-F1 più alta;
- `StackingEnsemble` è quasi equivalente e ottiene macro-F1 e weighted-F1 leggermente migliori, ma è più complesso;
- poiché la metrica principale è micro-F1, `XGBoost` è più difendibile come modello finale;
- la scelta resta modificabile solo se ulteriori test mostrano un miglioramento chiaro e stabile.

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

I metodi basati su XGBoost, CatBoost o ReliefF sono da considerare opzionali/pesanti e dipendono dalla configurazione dell'ambiente.

Risultati osservati con Feature Selection RF a 30 feature, senza PCA e senza tuning:

```text
RandomForest      micro-F1 0.713858
XGBoost           micro-F1 0.738109
LightGBM          micro-F1 0.721859
VotingEnsemble    micro-F1 0.731529
StackingEnsemble  micro-F1 0.737380
```

Conclusione:

- la feature selection funziona ed è leak-safe se usata dentro la pipeline;
- 30 feature risultano troppo aggressive;
- la configurazione testata peggiora rispetto alla baseline avanzata senza feature selection;
- la feature selection resta disponibile come strumento opzionale, ma non viene adottata nella pipeline finale attuale.

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

Risultati osservati con PCA a 40 componenti, senza feature selection e senza tuning:

```text
RandomForest      micro-F1 0.707949
XGBoost           micro-F1 0.706049
LightGBM          micro-F1 0.700620
VotingEnsemble    micro-F1 0.710194
StackingEnsemble  micro-F1 0.709944
```

Conclusione:

- PCA 40 peggiora sensibilmente rispetto alla baseline avanzata senza PCA;
- la PCA non viene adottata nella configurazione finale attuale;
- viene mantenuta come esperimento secondario utile per collegare il progetto agli argomenti del corso, ma non come scelta prestazionale.

---

## Sample weighting

`sample_weight` è disponibile come opzione nella training pipeline.

Decisione:

- non usarlo come default;
- mantenerlo come esperimento alternativo orientato alla macro-F1.

Motivazione:

- i pesi bilanciati possono aiutare le classi meno frequenti;
- tuttavia possono penalizzare la micro-F1;
- poiché la metrica principale è micro-F1, non sono usati come default.

---

## Modelli

La pipeline supporta:

- RandomForest;
- XGBoost;
- LightGBM;
- VotingEnsemble;
- StackingEnsemble.

Decisione:

- RandomForest resta una baseline stabile e interpretabile;
- XGBoost è il candidato finale attuale;
- LightGBM è competitivo ma inferiore a XGBoost nella configurazione corrente;
- VotingEnsemble e StackingEnsemble sono utili come confronto, ma non vengono preferiti automaticamente;
- `models_to_run` permette di eseguire solo un sottoinsieme di modelli, rendendo più rapidi i test mirati.

Nota per macOS:

- XGBoost e LightGBM possono richiedere `libomp`;
- su Mac ARM può essere necessario installare `libomp` tramite Homebrew:

```bash
brew install libomp
```

---

## Tuning

Il tuning è stato integrato e testato in più configurazioni.

La pipeline distingue correttamente due casi:

- se `feature_selection=True`, il tuning usa `FeatureSelectionTuner`;
- se `feature_selection=False`, il tuning usa `ModelTuner` e ottimizza solo parametri del modello con prefisso `model__`.

Questa distinzione è necessaria perché, quando la feature selection è disattivata, la pipeline non contiene lo step `feature_selector`.

### Tuning + feature selection RF

Configurazione:

```text
feature_selection = True
fs_method = "rf"
tuning_iter = 3
tuning_sample_size = 10000
max_features_to_hold = 30
```

Risultati:

```text
RandomForest      micro-F1 0.700313
XGBoost           micro-F1 0.709119
LightGBM          micro-F1 0.673721
VotingEnsemble    micro-F1 0.675735
StackingEnsemble  micro-F1 0.685616
```

### Tuning senza feature selection

Configurazione:

```text
feature_selection = False
tuning_iter = 3
tuning_sample_size = 10000
```

Risultati:

```text
RandomForest      micro-F1 0.723950
XGBoost           micro-F1 0.719748
LightGBM          micro-F1 0.677807
VotingEnsemble    micro-F1 0.718156
StackingEnsemble  micro-F1 0.730473
```

### Tuning solo XGBoost

Configurazione:

```text
feature_selection = False
do_tuning = True
tuning_iter = 10
tuning_sample_size = 30000
models_to_run = ["XGBoost"]
```

Risultato:

```text
XGBoost micro-F1 0.724813 | macro-F1 0.661019 | weighted-F1 0.716496
```

Conclusione:

- il tuning funziona tecnicamente;
- nessuna configurazione testata batte la baseline avanzata corrente;
- la pipeline candidata finale non usa tuning.

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

Su macOS, se XGBoost o LightGBM danno errore legato a librerie native, installare:

```bash
brew install libomp
```

### Esecuzione candidato finale XGBoost

```bash
python3 - <<'PY'
from src.pipeline_training_model import run_training_pipeline

results = run_training_pipeline(
    feature_selection=False,
    split_strategy=2,
    use_sample_weight=False,
    fs_method="rf",
    use_pca=False,
    do_tuning=False,
    models_to_run=["XGBoost"],
)

print(results)
PY
```

Output atteso:

```text
XGBoost  micro-F1 ≈ 0.741889
```

### Esecuzione confronto modelli avanzati

```bash
python3 - <<'PY'
from src.pipeline_training_model import run_training_pipeline

results = run_training_pipeline(
    feature_selection=False,
    split_strategy=2,
    use_sample_weight=False,
    fs_method="rf",
    use_pca=False,
    do_tuning=False,
)

print(results)
PY
```

### Esecuzione PCA 40 come esperimento secondario

```bash
python3 - <<'PY'
from src.pipeline_training_model import run_training_pipeline

run_training_pipeline(
    feature_selection=False,
    split_strategy=2,
    use_sample_weight=False,
    fs_method="rf",
    use_pca=True,
    pca_n_components=40,
    do_tuning=False,
)
PY
```

### Esecuzione Feature Selection RF a 30 feature

```bash
python3 - <<'PY'
from src.pipeline_training_model import run_training_pipeline

run_training_pipeline(
    feature_selection=True,
    split_strategy=2,
    use_sample_weight=False,
    fs_method="rf",
    max_features_to_hold=30,
    use_pca=False,
    do_tuning=False,
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
├── 04_preprocessing_feature_engineering.ipynb
├── 05_baseline_modeling.ipynb
├── 06_model_comparison_feature_selection.ipynb
└── 07_tuning_final_evaluation.ipynb
```

Ruolo dei notebook:

- `01_analisi_dati.ipynb`: analisi esplorativa iniziale;
- `02_qualita_dati.ipynb`: qualità dati e sintesi data quality;
- `03_feature_comprehension.ipynb`: comprensione semantica delle feature;
- `04_preprocessing_feature_engineering.ipynb`: preprocessing e feature engineering;
- `05_baseline_modeling.ipynb`: baseline preliminare;
- `06_model_comparison_feature_selection.ipynb`: confronto modelli, feature selection e PCA;
- `07_tuning_final_evaluation.ipynb`: tuning e valutazione finale, se usato.

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
- model comparison;
- feature selection;
- PCA;
- sample weighting;
- tuning;
- scelta candidata finale;
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
- PCA integrata come step opzionale, ma non adottata nella pipeline finale;
- `sample_weight` disponibile ma non usato come default;
- XGBoost è il candidato finale attuale;
- StackingEnsemble è l'alternativa quasi equivalente;
- tuning testato ma non adottato;
- `models_to_run` permette test selettivi sui modelli;
- la documentazione metodologica dettagliata è in `docs/decision_log.md`.

---

## Prossimi step

- creare o aggiornare una tabella esperimenti unica e coerente in `outputs/metrics/`;
- decidere se implementare `make_submission` come patch separata;
- eseguire test finale minimo della pipeline;
- pushare il branch consolidato;
- aprire PR verso `dev`;
- preparare materiale per report e presentazione finale.

---

## Team

- Gianluca
- Nicola
- Mattia
- Claudia
