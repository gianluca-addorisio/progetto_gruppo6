# Decision Log

Questo documento raccoglie le principali decisioni metodologiche prese durante lo sviluppo del progetto **Richter's Predictor: Modeling Earthquake Damage**.

La versione qui riportata corrisponde alla **versione finale di consegna** del progetto. Codice, pipeline, submission, README, decision log e presentazione sono stati completati e verificati.

L'obiettivo del documento non è sostituire i notebook esplorativi, ma fornire una traccia sintetica, ordinata e difendibile delle scelte operative che hanno guidato preprocessing, feature engineering, model comparison, feature selection, tuning, scelta finale del modello e generazione della submission.

---

## 1. Target e metrica principale

Il task è una classificazione multiclasse.

La variabile target è:

```text
damage_grade
```

Le classi sono:

| Classe | Significato |
| -----: | ----------- |
| 1 | danno basso |
| 2 | danno medio |
| 3 | danno elevato |

La metrica principale del progetto è la **micro-F1**, coerente con la metrica ufficiale della competizione DrivenData.

Decisione metodologica:

- usare la micro-F1 come criterio principale per confrontare modelli e configurazioni;
- mantenere macro-F1 e weighted-F1 come metriche di supporto;
- usare classification report e confusion matrix come strumenti diagnostici, non come criterio primario di selezione.

Motivazione:

- la micro-F1 aggrega globalmente gli errori sulle tre classi;
- la metrica è quella richiesta dalla competizione;
- macro-F1 e weighted-F1 aiutano a leggere il comportamento sulle classi, ma non sostituiscono la metrica ufficiale.

---

## 2. Separazione tra identificativi, target e feature

`building_id` è un identificativo tecnico dell'edificio.

Decisione:

- non usarlo come feature predittiva;
- mantenerlo solo per merge, tracciamento righe e submission;
- rimuoverlo prima della modellazione tramite `DataCleaner`.

Motivazione:

- non rappresenta una proprietà fisica, strutturale, geografica o d'uso dell'edificio;
- usarlo come predittore rischierebbe di introdurre rumore o pattern non generalizzabili.

`damage_grade` è la variabile target.

Decisione:

- rimuoverla dalla matrice delle feature se accidentalmente presente;
- usarla solo come variabile da predire.

Motivazione:

- mantenerla in `X` causerebbe target leakage;
- la separazione corretta del problema supervisionato è `X = feature dell'edificio` e `y = damage_grade`.

Nel workflow finale le label originali `1`, `2`, `3` vengono temporaneamente ricodificate in `0`, `1`, `2` durante il training per garantire compatibilità con modelli come XGBoost. Le predizioni vengono poi riportate nello spazio originale richiesto dalla competizione.

---

## 3. Architettura finale del codice

La struttura finale separa chiaramente sperimentazione, training finale e generazione della submission.

Decisione finale:

| File | Responsabilità |
| ---- | -------------- |
| `main.py` | CLI del progetto e orchestrazione dei comandi principali |
| `src/config.py` | configurazione centralizzata di path, seed e parametri finali |
| `src/pipeline_training_model.py` | esperimenti, validazione interna, model comparison, ablation study |
| `src/final_model.py` | training finale, salvataggio pipeline, generazione submission |
| `src/preprocessing/` | pipeline modulare di preprocessing |
| `src/features.py` | feature engineering compatto |
| `src/models.py` | costruttori di baseline, modelli avanzati ed ensemble |
| `src/featureselector.py` | transformer sklearn per feature selection leak-safe |
| `src/feature_selection.py` | metodi di ranking e selezione feature |
| `src/hyperparameter_tuning.py` | tuning degli iperparametri dei modelli |
| `src/hyperparameter_tuning_feature_selection.py` | tuning congiunto di feature selection e modello |
| `src/evaluation.py` | metriche, classification report e confusion matrix |

Motivazione:

- evitare di mescolare esperimenti e workflow finale;
- mantenere il training finale riproducibile e isolato;
- consentire la rigenerazione della submission da una pipeline salvata;
- rendere più chiara la discussione tecnica nel report e nella presentazione.

Decisione di integrazione finale:

- mantenere la struttura modulare del branch finale;
- integrare selettivamente le parti prestazionali migliori del lavoro parallelo;
- non fare merge automatico di branch che reintroducevano versioni obsolete o meno modulari della pipeline;
- separare `pipeline_training_model.py` da `final_model.py` per evitare side effect tra esperimenti e consegna.

---

## 4. Preprocessing finale

La pipeline di preprocessing finale parte dai dati raw e applica trasformazioni ordinate dentro una pipeline sklearn.

Ordine logico degli step:

1. feature engineering da variabili raw;
2. pulizia e rimozione di identificativi, target e feature ridondanti;
3. gestione del valore speciale dell'età;
4. encoding delle variabili geografiche ad alta cardinalità;
5. one-hot encoding delle categoriche a bassa cardinalità;
6. scaling opzionale solo quando necessario;
7. feature selection opzionale;
8. PCA opzionale;
9. modello finale.

Decisione:

- mantenere preprocessing, feature selection, PCA e modello nello stesso oggetto `Pipeline`;
- evitare trasformazioni manuali separate tra training, validation e test;
- usare la stessa pipeline per validazione interna, training finale e inferenza.

Motivazione:

- riduce il rischio di data leakage;
- garantisce coerenza tra training e test;
- rende più semplice salvare e riutilizzare la pipeline finale.

---

## 5. Feature engineering finale

La pipeline applica un feature engineering compatto e interpretabile.

Feature aggregate finali:

| Feature | Significato |
| ------- | ----------- |
| `total_superstructure_count` | numero di tecniche/materiali strutturali presenti |
| `total_secondary_use_count` | numero di usi secondari dell'edificio |
| `has_fragile_material` | presenza di materiali strutturali potenzialmente fragili |
| `has_engineered_structure` | presenza di elementi strutturali più ingegnerizzati |
| `is_historic` | indicatore del valore speciale `age = 995` |

Decisione:

- comprimere i gruppi binari molto numerosi in aggregati più compatti;
- mantenere feature interpretabili e collegate alla vulnerabilità sismica;
- rimuovere le feature originali sostituite dagli aggregati dopo il feature engineering.

Motivazione:

- riduzione della dimensionalità;
- minore sparsità;
- maggiore leggibilità metodologica;
- conservazione del segnale strutturale principale.

---

## 6. Feature dimensionali

Sono state confrontate due strategie principali:

### Variante con proxy volumetrico

Feature mantenute:

```text
area_percentage
height_percentage
building_volume_proxy
```

La feature `building_volume_proxy` è definita come combinazione di area e altezza.

### Variante finale

Feature mantenute:

```text
area_percentage
height_percentage
```

Feature rimossa:

```text
building_volume_proxy
```

Decisione finale:

- mantenere `area_percentage`;
- mantenere `height_percentage`;
- creare `building_volume_proxy` come feature candidata in `features.py`;
- rimuovere `building_volume_proxy` nella pipeline finale tramite `DataCleaner`.

Motivazione:

- `area_percentage` e `height_percentage` sono feature originali, semplici e interpretabili;
- `building_volume_proxy` è ridondante rispetto alle due feature originali;
- il vantaggio osservato mantenendo il proxy non è stato sufficiente a giustificarne l'adozione finale;
- la scelta finale è più pulita e più difendibile nel report.

---

## 7. Gestione dell'età

Durante l'analisi è stato osservato il valore speciale:

```text
age = 995
```

Decisione finale:

- mantenere `age`;
- non mantenere `age_clipped`;
- non mantenere `age_group`;
- gestire `age = 995` tramite `AgeHandler`;
- introdurre `is_historic` come indicatore esplicito del valore speciale.

Comportamento della pipeline:

- `is_historic = 1` se `age = 995`;
- `is_historic = 0` altrimenti;
- il valore `995` viene sostituito con la massima età normale osservata nel training set.

Motivazione:

- `age` conserva informazione numerica diretta;
- `age_clipped` e `age_group` derivano da `age` e introducono ridondanza;
- `age_group` richiede soglie arbitrarie;
- `is_historic` conserva il significato speciale del valore `995`;
- la sostituzione del valore estremo riduce effetti numerici indesiderati.

---

## 8. Feature di uso secondario

Le feature originali di uso secondario includono:

```text
has_secondary_use
has_secondary_use_*
```

Decisione finale:

- rimuovere le feature originali di uso secondario;
- mantenere `total_secondary_use_count`.

Motivazione:

- molte sotto-categorie sono rare;
- mantenerle tutte aumenta sparsità e dimensionalità;
- `total_secondary_use_count` conserva un segnale sintetico sull'intensità degli usi secondari;
- la scelta rappresenta una compressione del segnale, non una sua eliminazione completa.

---

## 9. Feature di superstruttura

Le feature originali `has_superstructure_*` descrivono materiali e tecniche costruttive.

Decisione finale:

- rimuovere le singole feature `has_superstructure_*`;
- mantenere gli aggregati:

```text
total_superstructure_count
has_fragile_material
has_engineered_structure
```

Motivazione:

- le variabili di superstruttura sono rilevanti per la vulnerabilità sismica;
- le feature originali sono numerose e in parte ridondanti;
- gli aggregati conservano significato fisico e riducono dimensionalità;
- la rappresentazione finale è più compatta e interpretabile.

---

## 10. Feature categoriche strutturali

Feature categoriche mantenute:

```text
land_surface_condition
foundation_type
roof_type
ground_floor_type
other_floor_type
position
```

Decisione:

- mantenerle nella pipeline finale;
- applicare one-hot encoding tramite `CategoricalEncoder`.

Motivazione:

- cardinalità contenuta;
- costo dimensionale limitato;
- collegamento plausibile con vulnerabilità strutturale e condizioni dell'edificio.

Feature rimosse:

```text
plan_configuration
legal_ownership_status
```

Motivazione:

- distribuzione sbilanciata;
- contributo debole o poco robusto nei test diagnostici;
- riduzione della complessità della pipeline.

---

## 11. Encoding geografico

Le feature geografiche originali sono:

```text
geo_level_1_id
geo_level_2_id
geo_level_3_id
```

Queste variabili sono codificate come identificativi, quindi non devono essere trattate come numeriche continue o ordinali.

Decisione finale:

| Feature | Encoding finale |
| ------- | --------------- |
| `geo_level_1_id` | one-hot encoding |
| `geo_level_2_id` | frequency encoding |
| `geo_level_3_id` | frequency encoding |

Motivazione:

- `geo_level_1_id` ha cardinalità gestibile e può essere codificata tramite one-hot;
- `geo_level_2_id` e `geo_level_3_id` hanno cardinalità più alta;
- il frequency encoding evita un'esplosione dimensionale e mantiene un segnale sulla frequenza relativa delle aree;
- le categorie non viste in training vengono mappate a `0.0`.

Nota metodologica:

- il frequency encoding è appreso solo sui dati di training dello split corrente quando è inserito nella pipeline sklearn;
- questo riduce il rischio di leakage rispetto a trasformazioni calcolate globalmente prima dello split.

---

## 12. Feature selection

La feature selection è stata resa opzionale e integrata nella pipeline tramite `FeatureSelector`.

Metodi disponibili:

```text
rf
xgb
ctb
corr_matrix
chi2
mu
rlf
rfe
sfs
```

Decisione finale:

```text
Feature selection: attiva
Metodo: ctb
Soglia: 0.005
Numero massimo feature: 30
```

Motivazione:

- la feature selection è inserita dentro la pipeline sklearn;
- gli score vengono calcolati solo sui dati di training dello split;
- l'approccio evita data leakage durante validazione, cross-validation e tuning;
- il metodo `ctb` usa importanze basate su CatBoost, robuste per ranking di feature tabellari;
- la soglia e il numero massimo di feature sono gestiti come iperparametri nella fase di tuning.

Nel training finale, i parametri della feature selection per lo stacking vengono ricavati dalla fase di tuning e riutilizzati nella pipeline finale. Questa scelta evita una selezione manuale incoerente con la configurazione sperimentale.

---

## 13. PCA e dimensionality reduction

La PCA è stata implementata come opzione della pipeline.

Decisione finale:

```text
PCA: disattivata
```

Motivazione:

- i modelli ad alberi e boosting usati nel progetto non richiedono PCA;
- la PCA riduce l'interpretabilità delle feature;
- i test non hanno mostrato un miglioramento sufficiente;
- la configurazione finale privilegia feature tabellari preprocessate e feature selection rispetto a estrazione lineare di componenti.

La PCA resta disponibile come opzione sperimentale, ma non viene adottata nella pipeline finale.

---

## 14. Sample weighting

È stata considerata la possibilità di usare pesi di classe tramite `sample_weight`.

Decisione finale:

```text
Sample weighting: disattivato
```

Motivazione:

- la metrica ufficiale è micro-F1;
- il sample weighting può alterare il trade-off tra classi senza migliorare necessariamente la micro-F1;
- i test non hanno mostrato un vantaggio finale sufficiente;
- la configurazione conclusiva privilegia stabilità e coerenza con la validazione.

Il supporto tecnico al sample weighting resta disponibile nella pipeline sperimentale, ma non è usato nel modello finale.

---

## 15. Model comparison

Sono stati confrontati baseline preliminari, modelli avanzati singoli ed ensemble.

### Baseline preliminari

| Modello | micro-F1 indicativa |
| ------- | ------------------: |
| DummyClassifier | ≈ 0.569 |
| LogisticRegression | ≈ 0.592 |
| DecisionTree | ≈ 0.643 |

Decisione:

- usare queste baseline come riferimento iniziale;
- non mischiarle con il confronto avanzato finale, perché appartengono a una fase preliminare.

### Confronto modelli avanzati

Configurazione del confronto avanzato:

```text
feature_selection = False
use_pca = False
use_sample_weight = False
do_tuning = False
split_strategy = 2
```

Risultati:

| Rank | Modello | micro-F1 | macro-F1 | weighted-F1 |
| ---: | ------- | -------: | -------: | ----------: |
| 1 | XGBoost | 0.741889 | 0.687747 | 0.735980 |
| 2 | StackingEnsemble | 0.741851 | 0.688311 | 0.736433 |
| 3 | VotingEnsemble | 0.735462 | 0.673930 | 0.727158 |
| 4 | LightGBM | 0.727615 | 0.664767 | 0.719590 |
| 5 | RandomForest | 0.716160 | 0.645689 | 0.704259 |

Interpretazione:

- XGBoost è stato il migliore modello singolo nella comparazione avanzata;
- StackingEnsemble è risultato quasi equivalente in micro-F1 e leggermente migliore in macro-F1 e weighted-F1;
- gli ensemble hanno confermato l'utilità di combinare modelli diversi, ma richiedono maggiore complessità.

---

## 16. Tuning

È stato implementato tuning tramite `RandomizedSearchCV`.

Decisioni tecniche:

- usare `ModelTuner` per tuning dei modelli e delle pipeline senza feature selection;
- usare `FeatureSelectionTuner` per tuning congiunto di feature selection e modello;
- usare `f1_micro` come scoring;
- usare campionamento del training set per contenere i tempi computazionali nel training finale.

Configurazione finale:

```text
Hyperparameter tuning: attivo
Tuning iterations: 15
Tuning sample size: 50000
```

Decisione finale:

- ottimizzare i modelli base RandomForest, XGBoost e LightGBM;
- usare gli stimatori ottimizzati come base dello `StackingEnsemble`;
- mantenere la feature selection dentro la pipeline anche durante il tuning.

Motivazione:

- gli stimatori base dello stacking non devono essere versioni default;
- il tuning migliora la coerenza della configurazione conclusiva;
- il campione da 50000 osservazioni consente un compromesso pratico tra costo computazionale e qualità della ricerca.

---

## 17. Scelta finale del modello

La configurazione finale centralizzata in `src/config.py` è:

```python
FINAL_MODEL_NAME = "StackingEnsemble"
FINAL_SPLIT_STRATEGY = 4
FINAL_FEATURE_SELECTION = True
FINAL_FS_METHOD = "ctb"
FINAL_FS_THRESHOLD = 0.005
FINAL_MAX_FEATURES_TO_HOLD = 30
FINAL_USE_SAMPLE_WEIGHT = False
FINAL_USE_PCA = False
FINAL_DO_TUNING = True
FINAL_TUNING_ITER = 15
FINAL_TUNING_SAMPLE_SIZE = 50000
```

Decisione finale:

- usare `StackingEnsemble` come modello conclusivo;
- costruirlo a partire da RandomForest, XGBoost e LightGBM;
- ottimizzare prima i tre modelli base;
- integrare feature selection basata su CatBoost;
- non usare PCA;
- non usare sample weighting.

Motivazione:

- XGBoost è il migliore modello singolo nel confronto avanzato;
- StackingEnsemble consente di combinare tre famiglie di modelli robuste;
- lo stacking ottiene risultati quasi equivalenti nel confronto avanzato e una configurazione finale più completa dopo tuning e feature selection;
- la submission finale pubblica conferma la competitività della configurazione conclusiva.

---

## 18. Training finale e submission

Il training finale è gestito da `src/final_model.py`.

Workflow finale:

1. caricamento dei dati raw;
2. ricodifica temporanea del target da `1,2,3` a `0,1,2`;
3. tuning dei modelli base;
4. costruzione dello `StackingEnsemble`;
5. costruzione della pipeline completa con preprocessing e feature selection;
6. fit sul training set completo;
7. salvataggio della pipeline finale;
8. generazione della submission;
9. riconversione delle label predette in `1,2,3`.

File prodotti:

```text
models/final_pipeline.joblib
outputs/metrics/final_model_config.json
outputs/submissions/final_submission.csv
```

Decisione:

- salvare la pipeline completa, non solo il modello;
- consentire la rigenerazione della submission da modello salvato;
- mantenere training finale e inferenza separati dagli esperimenti.

Motivazione:

- la pipeline salvata contiene preprocessing, feature selection e modello;
- la submission può essere rigenerata senza ripetere tuning e training;
- il workflow è più riproducibile e più semplice da verificare.

---

## 19. Risultati finali

Risultato interno della configurazione finale:

| Modello finale | micro-F1 indicativa |
| -------------- | ------------------: |
| StackingEnsemble | ≈ 0.740423 |

Nota:

- il valore interno deriva dalla configurazione finale con tuning e feature selection;
- non va confrontato in modo meccanico con il confronto avanzato senza tuning, senza feature selection e con split diverso;
- il criterio conclusivo considera anche il risultato pubblico della submission.

Risultato pubblico della submission finale:

| Modello finale | Public score |
| -------------- | -----------: |
| StackingEnsemble | 0.7419 |

Distribuzione delle predizioni sul test set:

| Classe | Numero predizioni |
| -----: | ----------------: |
| 1 | 6176 |
| 2 | 56466 |
| 3 | 24226 |

Decisione finale:

- adottare la submission generata da `StackingEnsemble` come submission ufficiale;
- documentare i confronti precedenti come parte del processo sperimentale, non come configurazione finale.

---

## 20. Branch, integrazione e pulizia finale

Durante l'ultima fase sono state confrontate versioni parallele della pipeline.

Decisione finale di integrazione:

- non ripristinare integralmente pipeline più vecchie o meno modulari;
- mantenere la struttura finale con `main.py`, `pipeline_training_model.py` e `final_model.py`;
- recuperare la logica prestazionale utile, in particolare tuning dei modelli base, feature selection e stacking;
- non fare merge automatico di branch che eliminavano componenti finali o reintroducevano codice legacy;
- usare un branch dedicato di integrazione conclusiva prima del merge finale su `main`.

La fase finale è stata consolidata tramite commit separati per:

- integrazione della pipeline modulare finale;
- aggiunta di docstring e commenti tecnici;
- aggiornamento della documentazione finale.

Decisione documentale finale:

- aggiungere docstring e commenti tecnici al codice senza modificare la logica;
- aggiornare README e decision log in coerenza con la pipeline finale;
- non committare file temporanei, PDF generati localmente o log di training.


---

## 21. Stato finale del progetto

La versione finale del repository è pronta per la consegna.

| Elemento | Stato |
| -------- | ----- |
| Pipeline ufficiale | completata e verificata |
| Preprocessing modulare | completato |
| Feature engineering | completato |
| Encoding geografico | completato |
| Feature selection | adottata nella configurazione finale |
| PCA | implementata ma non adottata |
| Sample weighting | implementato ma non adottato |
| Model comparison | completato |
| Tuning | adottato nella configurazione finale |
| Training finale | completato |
| Pipeline salvata | generata |
| Submission finale | generata |
| Public score | `0.7419` |
| README | aggiornato |
| Decision log | aggiornato |
| Presentazione | completata |

Decisione conclusiva:

```text
Modello finale: StackingEnsemble
Feature selection: sì, metodo ctb
PCA: no
Sample weighting: no
Tuning: sì
Submission finale: outputs/submissions/final_submission.csv
Public score: 0.7419
```

Questa configurazione rappresenta il compromesso finale tra prestazione, robustezza metodologica, modularità del codice e chiarezza espositiva.
