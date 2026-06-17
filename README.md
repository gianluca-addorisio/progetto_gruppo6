# FIA Earthquake Damage Predictor

Progetto finale del corso di **Fondamenti di Intelligenza Artificiale**.

L'obiettivo del progetto è predire il livello di danno subito dagli edifici in seguito al terremoto del Nepal del 2015, utilizzando il dataset della competizione DrivenData:

**Richter's Predictor: Modeling Earthquake Damage**

---

## Obiettivo

Il progetto consiste nello sviluppo di una pipeline di Machine Learning per classificare gli edifici in base al livello di danno subito.

La variabile target è:

```text
damage_grade
```

Le classi sono:

```text
1 = danno basso
2 = danno medio
3 = danno elevato
```

La metrica principale di valutazione è:

```text
micro-F1 score
```

Metriche di supporto:

```text
macro-F1
weighted-F1
```

La scelta del modello finale è guidata principalmente dalla micro-F1, coerentemente con la metrica ufficiale della competizione.

---

## Dataset

I dati originali devono essere inseriti nella cartella:

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

Descrizione dei file:

* `train_values.csv`: feature degli edifici del training set;
* `train_labels.csv`: target `damage_grade`;
* `test_values.csv`: feature degli edifici del test set;
* `submission_format.csv`: formato richiesto per la submission finale.

I file originali non devono essere modificati direttamente.

La pipeline ufficiale lavora a partire dai dati raw e applica preprocessing, feature engineering, eventuale feature selection ed eventuale PCA direttamente tramite pipeline scikit-learn, senza richiedere dataset intermedi salvati su disco.

---

## Struttura del repository

```text
.
├── data/
│   └── raw/                  # dati originali
├── docs/                     # documentazione metodologica e decision log
├── notebooks/                # notebook esplorativi e sperimentali
├── outputs/
│   ├── figures/              # figure generate da analisi e feature selection
│   ├── metrics/              # risultati e confronti metrici
│   └── submissions/          # submission finali
├── src/                      # codice stabile e riutilizzabile
│   ├── preprocessing/        # pipeline modulare ufficiale
│   ├── config.py
│   ├── data_loader.py
│   ├── evaluation.py
│   ├── features.py
│   ├── feature_selection.py
│   ├── featureselector.py
│   ├── hyperparameter_tuning.py
│   ├── hyperparameter_tuning_feature_selection.py
│   ├── models.py
│   ├── pipeline_training_model.py
│   └── utils.py
├── main.py                   # entry point CLI
├── requirements.txt
├── README.md
└── LICENSE
```

La pipeline ufficiale di preprocessing si trova in:

```text
src/preprocessing/
```

La configurazione finale del progetto è centralizzata in:

```text
src/config.py
```

---

## File principali

```text
src/config.py
```

Contiene path, colonne principali, random seed e configurazione finale del progetto.

```text
src/data_loader.py
```

Gestisce caricamento dati e split train-validation.

```text
src/features.py
```

Contiene il feature engineering compatto.

```text
src/preprocessing/
```

Contiene la pipeline modulare ufficiale di preprocessing.

```text
src/featureselector.py
```

Definisce un transformer sklearn-compatible per feature selection opzionale.

```text
src/feature_selection.py
```

Contiene metodi di scoring e ranking delle feature.

```text
src/models.py
```

Definisce baseline, modelli avanzati ed ensemble.

```text
src/pipeline_training_model.py
```

Contiene la training pipeline principale e la generazione della submission finale.

```text
main.py
```

Espone i principali comandi operativi da terminale.

---

## Pipeline ufficiale

La pipeline ufficiale lavora sui dati raw e costruisce internamente il seguente flusso:

```text
raw data
→ split train/validation
→ preprocessing
→ eventuale FeatureSelector
→ eventuale PCA
→ modello
→ metriche
```

Questa struttura riduce il rischio di data leakage: preprocessing, feature selection e PCA vengono fittati solo sul training set o sul fold di training durante la cross-validation.

La pipeline modulare è composta dai seguenti step:

1. `feature_engineering`: crea feature aggregate tramite `src/features.py`;
2. `DataCleaner`: rimuove identificativi, target accidentale, feature escluse e feature originali compresse;
3. `AgeHandler`: gestisce il valore speciale `age = 995`;
4. `FrequencyEncoder`: applica frequency encoding a `geo_level_2_id` e `geo_level_3_id`;
5. `CategoricalEncoder`: applica one-hot encoding alle variabili categoriche e a `geo_level_1_id`;
6. `NumericalScaler`: applicato solo quando richiesto, per esempio con PCA;
7. `FeatureSelector`: opzionale;
8. `PCA`: opzionale;
9. modello finale.

Funzioni principali:

```text
get_preprocessing_steps()
get_preprocessing_pipeline()
make_complete_pipeline()
run_training_pipeline()
generate_final_submission()
```

---

## Configurazione finale

La configurazione finale attualmente adottata è:

```text
Modello: XGBoost
Split strategy: 2
Feature selection: no
PCA: no
Tuning: no
Sample weighting: no
```

Questa configurazione è definita in `src/config.py` tramite:

```text
FINAL_MODEL_NAME = "XGBoost"
FINAL_SPLIT_STRATEGY = 2
FINAL_FEATURE_SELECTION = False
FINAL_USE_SAMPLE_WEIGHT = False
FINAL_USE_PCA = False
FINAL_DO_TUNING = False
```

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

* `area_percentage` e `height_percentage` vengono mantenute come feature dimensionali originali;
* `building_volume_proxy` è stata testata ma non mantenuta nella pipeline finale, perché ridondante rispetto ad `area_percentage` e `height_percentage`;
* `total_secondary_use_count` sintetizza le feature originali `has_secondary_use_*`;
* `total_superstructure_count`, `has_fragile_material` e `has_engineered_structure` sintetizzano le feature originali `has_superstructure_*`;
* `age` viene mantenuta, ma il valore speciale `age = 995` viene gestito da `AgeHandler`;
* `is_historic` conserva l'informazione associata al valore speciale `age = 995`;
* `age_clipped` e `age_group` non vengono mantenute nella pipeline finale;
* `count_floors_pre_eq` e `count_families` vengono mantenute nella forma originale.

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

* `building_id` è un identificativo tecnico;
* `damage_grade` è il target;
* `building_volume_proxy` è ridondante rispetto ad `area_percentage` e `height_percentage`;
* le feature derivate da `age`, `count_families` e `count_floors_pre_eq` non hanno mostrato vantaggio sufficiente rispetto alle variabili grezze;
* `plan_configuration` e `legal_ownership_status` sono state rimosse per bassa informatività e distribuzione fortemente sbilanciata;
* le feature originali di uso secondario e superstruttura sono state compresse in aggregati interpretabili.

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

* `geo_level_1_id` ha cardinalità gestibile;
* `geo_level_2_id` e `geo_level_3_id` hanno cardinalità elevata;
* il one-hot completo delle variabili geografiche granulari produrrebbe una matrice troppo ampia e sparsa;
* il frequency encoding conserva un segnale geografico compatto;
* il frequency encoding viene fittato solo sul training set, evitando leakage tra train e validation.

---

## Modelli

La pipeline supporta:

```text
RandomForest
XGBoost
LightGBM
VotingEnsemble
StackingEnsemble
```

Sono inoltre disponibili baseline preliminari:

```text
DummyClassifier
LogisticRegression
DecisionTree
```

Decisioni principali:

* RandomForest resta una baseline avanzata stabile e interpretabile;
* XGBoost è il modello finale selezionato;
* LightGBM è competitivo ma inferiore a XGBoost nella configurazione corrente;
* VotingEnsemble e StackingEnsemble sono utili come confronto, ma non vengono preferiti automaticamente;
* `models_to_run` permette di eseguire solo un sottoinsieme di modelli.

---

## Risultati

### Baseline preliminari

Risultati indicativi delle baseline preliminari:

```text
DummyClassifier       micro-F1 ≈ 0.569
LogisticRegression    micro-F1 ≈ 0.592
DecisionTree          micro-F1 ≈ 0.643
```

Queste baseline servono come riferimento iniziale e non vanno confrontate direttamente con gli esperimenti avanzati se non come progressione metodologica.

### Confronto modelli avanzati

Risultati della configurazione avanzata senza feature selection, senza PCA e senza tuning:

```text
RandomForest      micro-F1 0.716160 | macro-F1 0.645689 | weighted-F1 0.704259
XGBoost           micro-F1 0.741889 | macro-F1 0.687747 | weighted-F1 0.735980
LightGBM          micro-F1 0.727615 | macro-F1 0.664767 | weighted-F1 0.719590
VotingEnsemble    micro-F1 0.735462 | macro-F1 0.673930 | weighted-F1 0.727158
StackingEnsemble  micro-F1 0.741851 | macro-F1 0.688311 | weighted-F1 0.736433
```

Decisione:

* `XGBoost` ottiene la micro-F1 interna più alta;
* `StackingEnsemble` è quasi equivalente e ottiene macro-F1 e weighted-F1 leggermente migliori;
* poiché la metrica principale è micro-F1, `XGBoost` è il modello finale più difendibile;
* la maggiore semplicità di XGBoost rispetto allo stacking rafforza la scelta finale.

### Submission finale

La submission finale è stata generata usando:

```text
modello: XGBoost
feature selection: no
PCA: no
tuning: no
training finale: tutto il training set disponibile
```

File prodotto:

```text
outputs/submissions/final_submission.csv
```

Risultati pubblici:

```text
XGBoost           public score = 0.7397
StackingEnsemble  public score = 0.7384
```

La scelta finale rimane quindi `XGBoost`, coerentemente con la validazione interna e con il risultato pubblico.

---

## Feature selection

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

```text
rf           Random Forest importance
xgb          XGBoost importance
ctb          CatBoost importance
corr_matrix  correlazione con il target
chi2         Chi-square
mu           mutual information
rlf          ReliefF
rfe          Recursive Feature Elimination
sfs          Sequential Feature Selection
```

I metodi basati su XGBoost, CatBoost, ReliefF, RFE o SFS possono essere più pesanti e dipendono dalla configurazione dell'ambiente.

Risultati osservati con Feature Selection RF a 30 feature, senza PCA e senza tuning:

```text
RandomForest      micro-F1 0.713858
XGBoost           micro-F1 0.738109
LightGBM          micro-F1 0.721859
VotingEnsemble    micro-F1 0.731529
StackingEnsemble  micro-F1 0.737380
```

Conclusione:

* la feature selection funziona ed è leak-safe se usata dentro la pipeline;
* 30 feature risultano troppo aggressive nella configurazione testata;
* la configurazione testata peggiora rispetto alla baseline avanzata senza feature selection;
* la feature selection resta disponibile come strumento opzionale, ma non viene adottata nella pipeline finale.

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

Risultati osservati con PCA a 40 componenti, senza feature selection e senza tuning:

```text
RandomForest      micro-F1 0.707949
XGBoost           micro-F1 0.706049
LightGBM          micro-F1 0.700620
VotingEnsemble    micro-F1 0.710194
StackingEnsemble  micro-F1 0.709944
```

Conclusione:

* PCA 40 peggiora sensibilmente rispetto alla baseline avanzata senza PCA;
* la PCA non viene adottata nella configurazione finale;
* viene mantenuta come esperimento secondario utile per collegare il progetto agli argomenti del corso.

---

## Sample weighting

`sample_weight` è disponibile come opzione nella training pipeline.

Decisione:

* non usarlo come default;
* mantenerlo come esperimento alternativo orientato alla macro-F1.

Motivazione:

* i pesi bilanciati possono aiutare le classi meno frequenti;
* tuttavia possono penalizzare la micro-F1;
* poiché la metrica principale è micro-F1, non sono usati come default.

---

## Tuning

Il tuning è stato integrato e testato in più configurazioni.

La pipeline distingue due casi:

* se `feature_selection=True`, il tuning usa `FeatureSelectionTuner`;
* se `feature_selection=False`, il tuning usa `ModelTuner` e ottimizza solo parametri del modello.

Risultati sintetici:

```text
Tuning + feature selection RF:
miglior risultato osservato inferiore alla baseline avanzata senza tuning.

Tuning senza feature selection:
miglior risultato osservato inferiore alla configurazione finale XGBoost.

Tuning solo XGBoost:
XGBoost micro-F1 0.724813 | macro-F1 0.661019 | weighted-F1 0.716496
```

Conclusione:

* il tuning funziona tecnicamente;
* nessuna configurazione testata batte la baseline avanzata corrente;
* la pipeline candidata finale non usa tuning.

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

Su macOS, se XGBoost o LightGBM danno errore legato a librerie native:

```bash
brew install libomp
```

### Help CLI

```bash
python main.py --help
```

Comandi disponibili:

```text
evaluate-final
compare-models
make-submission
```

### Valutazione candidato finale

```bash
python main.py evaluate-final
```

Output atteso:

```text
XGBoost micro-F1 ≈ 0.741889
```

### Confronto modelli avanzati

```bash
python main.py compare-models
```

Il comando salva i risultati in:

```text
outputs/metrics/results_comparison.csv
```

È possibile selezionare un sottoinsieme di modelli:

```bash
python main.py compare-models --models XGBoost,LightGBM,StackingEnsemble
```

### Generazione submission finale

```bash
python main.py make-submission
```

Il comando salva la submission in:

```text
outputs/submissions/final_submission.csv
```

È possibile specificare il modello:

```bash
python main.py make-submission --model XGBoost
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

* `01_analisi_dati.ipynb`: analisi esplorativa iniziale;
* `02_qualita_dati.ipynb`: qualità dati e data quality;
* `03_feature_comprehension.ipynb`: comprensione semantica delle feature;
* `04_preprocessing_feature_engineering.ipynb`: preprocessing e feature engineering;
* `05_baseline_modeling.ipynb`: baseline preliminare;
* `06_model_comparison_feature_selection.ipynb`: confronto modelli, feature selection e PCA;
* `07_tuning_final_evaluation.ipynb`: tuning e valutazione finale.

I notebook servono come supporto analitico e narrativo. La logica stabile finale si trova in `src/`.

---

## Documentazione

Documenti principali:

```text
docs/
└── decision_log.md
```

`docs/decision_log.md` contiene il razionale delle principali decisioni metodologiche:

* target e metrica principale;
* feature mantenute e rimosse;
* feature engineering;
* encoding geografico;
* preprocessing finale;
* model comparison;
* feature selection;
* PCA;
* sample weighting;
* tuning;
* scelta finale.

Il README fornisce una panoramica sintetica del progetto e istruzioni operative.

---

## Stato attuale

Stato aggiornato:

* feature set compatto implementato;
* pipeline ufficiale implementata nel package `src/preprocessing/`;
* configurazione finale centralizzata in `src/config.py`;
* entry point CLI disponibile in `main.py`;
* encoding geografico ibrido implementato;
* `AgeHandler` integrato per gestire `age = 995`;
* `FeatureSelector` disponibile come step opzionale e leak-safe;
* PCA integrata come step opzionale, ma non adottata nella pipeline finale;
* `sample_weight` disponibile ma non usato come default;
* XGBoost selezionato come modello finale;
* StackingEnsemble mantenuto come alternativa quasi equivalente;
* tuning testato ma non adottato;
* submission finale generata e valutata sulla leaderboard pubblica;
* documentazione metodologica dettagliata disponibile in `docs/decision_log.md`.

---

## Prossimi step

* eseguire un controllo finale della repository;
* verificare coerenza tra README, `docs/decision_log.md`, report e presentazione;
* preparare report finale;
* preparare presentazione;
* consolidare il branch `dev`;
* effettuare merge finale su `main`.

---

## Team

* Gianluca
* Nicola
* Mattia
* Claudia
