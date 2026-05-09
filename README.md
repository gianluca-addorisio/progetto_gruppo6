# FIA Earthquake Damage Predictor

Progetto finale del corso di *Fondamenti di Intelligenza Artificiale*.

L'obiettivo del progetto è predire il livello di danno subito dagli edifici in seguito al terremoto del Nepal del 2015, utilizzando i dati della competizione DrivenData:

**Richter's Predictor: Modeling Earthquake Damage**

---

## Obiettivo del progetto

Il progetto consiste nello sviluppo di una pipeline di Machine Learning per classificare gli edifici in base al livello di danno subito.

La variabile target è:

- `damage_grade`

Le classi previste sono:

- `1` → danno basso
- `2` → danno medio
- `3` → danno elevato

La metrica di valutazione principale è:

- **micro-F1 score**

---

## Dataset

I dati originali sono contenuti nella cartella `data/raw/`.

File principali:

```text
data/raw/
├── train_values.csv
├── train_labels.csv
├── test_values.csv
└── submission_format.csv
```

Descrizione dei file:

- `train_values.csv`: contiene le feature degli edifici del training set.
- `train_labels.csv`: contiene la variabile target `damage_grade`.
- `test_values.csv`: contiene le feature degli edifici del test set.
- `submission_format.csv`: contiene il formato richiesto per eventuale submission finale.

I file originali non devono essere modificati direttamente.

---

## Struttura del repository

```text
.
├── data/
│   ├── raw/          # dati originali
│   ├── interim/      # dati intermedi generati durante analisi/preprocessing
│   └── processed/    # dati finali pronti per la modellazione
├── docs/             # documentazione, note e materiali di supporto
├── models/           # modelli salvati
├── notebooks/        # notebook di analisi, preprocessing e modellazione
├── outputs/          # risultati, metriche e file di submission
├── src/              # codice Python riutilizzabile
├── requirements.txt  # dipendenze del progetto
├── README.md
└── LICENSE
```

Le cartelle `data/interim/`, `data/processed/`, `models/` e `outputs/` sono predisposte per contenere file generati durante lo sviluppo.

I file generati in queste cartelle sono ignorati da Git, mentre i file `.gitkeep` permettono di mantenere la struttura delle cartelle nel repository.

---

## Pipeline di sviluppo

La pipeline prevista del progetto è la seguente.

### 1. Data loading

Caricamento dei file originali da `data/raw/`:

- `train_values.csv`
- `train_labels.csv`
- `test_values.csv`
- `submission_format.csv`

In questa fase viene anche effettuato il merge tra feature e target tramite `building_id`.

### 2. Exploratory Data Analysis

Analisi esplorativa generale del dataset:

- dimensioni del dataset;
- distribuzione del target;
- statistiche descrittive;
- prime osservazioni sulle feature disponibili.

### 3. Data quality analysis

Controllo della qualità dei dati:

- missing values;
- duplicati;
- tipi delle variabili;
- valori anomali;
- consistenza generale del dataset.

### 4. Feature comprehension

Analisi semantica delle feature:

- classificazione delle variabili per gruppo;
- analisi di feature numeriche, categoriche, geografiche e binarie;
- studio preliminare della relazione tra feature e `damage_grade`;
- indicazioni operative per preprocessing e modellazione.

### 5. Preprocessing

Preparazione dei dati per la modellazione:

- rimozione di `building_id`;
- separazione tra feature e target;
- train/validation split stratificato;
- encoding delle variabili categoriche;
- gestione delle feature geografiche ad alta cardinalità;
- scaling dove necessario.

### 6. Feature engineering / feature extraction

Creazione di nuove feature interpretabili a partire dalle variabili disponibili.

Possibili esempi:

- numero totale di materiali strutturali presenti;
- numero totale di usi secondari dell'edificio;
- indicatori aggregati su materiali fragili o resistenti;
- proxy dimensionali dell'edificio;
- classi di età dell'edificio.

### 7. Feature selection

Selezione delle feature più informative tramite:

- analisi esplorativa;
- feature importance da modelli tree-based;
- permutation importance;
- confronto tra modelli con tutte le feature e modelli con feature selezionate.

### 8. Dimensionality reduction

Esperimento opzionale con tecniche di riduzione della dimensionalità, in particolare PCA.

La PCA verrà considerata solo dopo il preprocessing e sarà valutata confrontando le performance dei modelli con e senza componenti principali.

### 9. Baseline modeling

Costruzione dei primi modelli di riferimento:

- DummyClassifier;
- Logistic Regression;
- Decision Tree semplice.

Questa fase serve a definire una base minima di confronto.

### 10. Model comparison

Confronto tra modelli diversi, ad esempio:

- Logistic Regression;
- k-NN;
- Decision Tree;
- Random Forest;
- Gradient Boosting;
- eventuali modelli aggiuntivi se compatibili con tempi e setup.

### 11. Hyperparameter tuning

Ottimizzazione dei modelli migliori tramite:

- GridSearchCV;
- RandomizedSearchCV;
- validazione incrociata.

### 12. Final evaluation

Valutazione finale tramite:

- micro-F1 score;
- confusion matrix;
- analisi degli errori;
- confronto delle performance sulle diverse classi di `damage_grade`.

### 13. Submission, report e presentazione

Produzione degli output finali:

- eventuale file di submission;
- sintesi dei risultati;
- aggiornamento della documentazione;
- preparazione della presentazione finale.

---

## Notebook

Notebook attuali o previsti:

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

Stato indicativo:

- `01_analisi_dati.ipynb`: analisi esplorativa iniziale.
- `02_qualita_dati.ipynb`: analisi della qualità dei dati.
- `03_feature_comprehension.ipynb`: analisi semantica e informatività delle feature.
- `04_preprocessing_feature_engineering.ipynb`: preprocessing e costruzione di nuove feature.
- `05_baseline_modeling.ipynb`: primi modelli baseline.
- `06_model_comparison_feature_selection.ipynb`: confronto modelli e feature selection.
- `07_tuning_final_evaluation.ipynb`: tuning, valutazione finale e analisi errori.

---

## Codice sorgente

La cartella `src/` contiene codice Python riutilizzabile:

```text
src/
├── __init__.py
├── data_loader.py
├── preprocessing.py
└── utils.py
```

Obiettivo della cartella `src/`:

- evitare duplicazione di codice nei notebook;
- centralizzare funzioni di caricamento dati;
- definire funzioni comuni di preprocessing;
- gestire metriche e utility condivise.

---

## Setup ambiente

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

---

## Esecuzione del progetto

La pipeline completa è ancora in fase di sviluppo.

Al momento il flusso consigliato è:

1. eseguire i notebook in ordine;
2. spostare in `src/` il codice stabile e riutilizzabile;
3. salvare eventuali output intermedi in `data/interim/`;
4. salvare dataset pronti per la modellazione in `data/processed/`;
5. salvare metriche, risultati e submission in `outputs/`.

---

## Organizzazione del lavoro

Proposta di suddivisione delle responsabilità.

### Mattia

Responsabilità principale:

- analisi esplorativa;
- primi modelli baseline.

Task previsti:

- completare e revisionare `01_analisi_dati.ipynb`;
- implementare primi modelli semplici;
- confrontare DummyClassifier, Logistic Regression e Decision Tree;
- produrre una prima tabella con micro-F1.

### Claudia

Responsabilità principale:

- qualità dei dati;
- preprocessing.

Task previsti:

- completare `02_qualita_dati.ipynb`;
- analizzare missing values, duplicati, tipi e valori anomali;
- contribuire alle decisioni di preprocessing;
- collaborare alla costruzione del notebook `04_preprocessing_feature_engineering.ipynb`.

### Gianluca

Responsabilità principale:

- feature comprehension;
- feature engineering;
- feature selection;
- esperimenti di dimensionality reduction.

Task previsti:

- completare e mantenere `03_feature_comprehension.ipynb`;
- proporre feature engineering interpretabile;
- sperimentare feature selection;
- valutare eventuale uso di PCA dopo preprocessing.

### Nicola

Responsabilità principale:

- infrastruttura tecnica comune;
- funzioni riutilizzabili in `src/`;
- modellazione avanzata nelle fasi successive.

Task immediati:

- sistemare o estendere `src/data_loader.py`;
- sistemare o estendere `src/preprocessing.py`;
- preparare funzioni per caricamento dati, merge feature-target e separazione X/y;
- preparare train/validation split stratificato;
- preparare funzioni comuni di valutazione con micro-F1.

Task successivi:

- confronto modelli;
- tuning;
- valutazione finale;
- eventuale submission.

---

## Risultati

Questa sezione verrà aggiornata durante lo sviluppo.

Risultati da riportare:

- metriche dei modelli baseline;
- confronto tra modelli;
- risultati dopo feature engineering;
- risultati dopo feature selection;
- risultati dopo tuning;
- modello finale scelto.

---

## Presentazione finale

La presentazione finale dovrà includere:

- obiettivo del progetto;
- descrizione del dataset;
- pipeline seguita;
- principali scelte di preprocessing;
- feature engineering e feature selection;
- modelli testati;
- metriche ottenute;
- analisi degli errori;
- conclusioni.

---

## Team

- Gianluca
- Nicola
- Claudia
- Mattia

---

## Stato del progetto

Stato attuale:

- repository inizializzato;
- dati originali caricati in `data/raw/`;
- struttura `data/` aggiornata con `raw/`, `interim/` e `processed/`;
- pipeline generale definita;
- suddivisione preliminare dei task proposta;
- notebook iniziali in fase di sviluppo.

