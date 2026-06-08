# Decision Log

Questo documento raccoglie le principali decisioni metodologiche prese durante lo sviluppo del progetto **FIA Earthquake Damage Predictor**.

L'obiettivo non è sostituire i notebook, ma mantenere una traccia sintetica, ordinata e aggiornata delle scelte operative che guidano preprocessing, feature engineering, feature selection, PCA, model comparison e tuning.

---

## 1. Target e metrica principale

Il target del progetto è:

- `damage_grade`

Le classi sono:

- `1`: danno basso
- `2`: danno medio
- `3`: danno elevato

La metrica principale di valutazione è:

- **micro-F1 score**

La micro-F1 viene usata per confrontare baseline, modelli successivi, feature engineering, feature selection, PCA ed eventuali esperimenti di tuning.

Metriche quantitative di supporto:

- macro-F1;
- weighted-F1.

Per l'analisi qualitativa degli errori e delle prestazioni per classe potranno essere usati anche:

- classification report;
- confusion matrix.

---

## 2. Gestione di `building_id` e `damage_grade`

`building_id` è un identificativo tecnico dell'edificio.

Decisione:

- non deve essere usato come feature predittiva;
- deve essere mantenuto solo per merge, tracciamento righe o submission;
- deve essere rimosso prima della modellazione.

Motivazione:

- non rappresenta una proprietà fisica, geografica, strutturale o d'uso dell'edificio;
- usarlo come variabile predittiva rischierebbe di introdurre rumore o pattern non generalizzabili.

`damage_grade` è la variabile target.

Decisione:

- deve essere rimossa da `X` se accidentalmente presente;
- deve essere usata solo come variabile da predire.

Motivazione:

- mantenerla nella matrice delle feature causerebbe target leakage;
- la separazione corretta del problema supervisionato è `X = feature dell'edificio` e `y = damage_grade`.

---

## 3. Feature set compatto

Dopo l'analisi delle variabili e i test diagnostici sulle principali famiglie di feature, è stata adottata una rappresentazione compatta e interpretabile della feature matrix.

L'obiettivo non è mantenere ogni singola variabile originaria, ma costruire una pipeline stabile, leggibile e difendibile, conservando le informazioni strutturali, dimensionali e geografiche principali.

### Feature principali prima dell'encoding

La rappresentazione compatta principale contiene le seguenti feature prima dell'encoding:

- `geo_level_1_id`
- `geo_level_2_id`
- `geo_level_3_id`
- `count_floors_pre_eq`
- `age`
- `land_surface_condition`
- `foundation_type`
- `roof_type`
- `ground_floor_type`
- `other_floor_type`
- `position`
- `count_families`
- `total_superstructure_count`
- `total_secondary_use_count`
- `has_fragile_material`
- `has_engineered_structure`
- `building_volume_proxy`

Nella pipeline modulare aggiornata viene inoltre creata la feature:

- `is_historic`

Questa feature è prodotta da `AgeHandler` per distinguere i casi con `age = 995`.

---

## 4. Feature dimensionali

La feature `building_volume_proxy` è definita come:

```text
building_volume_proxy = area_percentage * height_percentage
```

Decisione:

- mantenere `building_volume_proxy`;
- rimuovere `area_percentage`;
- rimuovere `height_percentage`.

Motivazione:

- `area_percentage` e `height_percentage` descrivono due aspetti dimensionali dell'edificio;
- `building_volume_proxy` sintetizza queste informazioni in una proxy dimensionale unica;
- la scelta riduce ridondanza e complessità della feature matrix;
- il trade-off prestazionale osservato nei test diagnostici è stato considerato accettabile rispetto al guadagno in compattezza e interpretabilità.

Nota:

- questa decisione non viene motivata come miglioramento assoluto della performance;
- viene adottata come semplificazione controllata della rappresentazione dimensionale.

---

## 5. Gestione dell'età

Durante l'analisi è stato osservato il valore estremo:

- `age = 995`

Decisione finale:

- mantenere `age`;
- rimuovere `age_clipped`;
- rimuovere `age_group`;
- gestire `age = 995` tramite `AgeHandler`.

La pipeline modulare crea:

- `is_historic = 1` se `age = 995`, altrimenti `0`.

Inoltre, durante il preprocessing:

- `AgeHandler` sostituisce `age = 995` con la massima età normale osservata nel training set.

Motivazione:

- `age` conserva direttamente l'informazione sull'età dell'edificio;
- `age_clipped` e `age_group` derivano da `age` e non aggiungono informazione indipendente;
- `age_group` introduce soglie arbitrarie;
- `is_historic` conserva invece in modo esplicito l'informazione che `995` rappresenta un caso speciale;
- la sostituzione di `995` riduce l'effetto di un valore numerico estremo sulla pipeline;
- la trasformazione viene appresa solo sul training set quando la pipeline è usata correttamente.

Nota:

- i dati originali in `data/raw/` non vengono modificati;
- `is_historic` è una scelta della pipeline modulare aggiornata e va considerata parte del preprocessing finale attuale.

---

## 6. Feature di uso secondario

Le feature originali di uso secondario sono:

- `has_secondary_use`
- tutte le feature `has_secondary_use_*`

Decisione finale:

- rimuovere le feature originali di uso secondario;
- mantenere solo `total_secondary_use_count`.

Motivazione:

- molte sotto-categorie di uso secondario sono molto rare;
- mantenerle tutte aumenterebbe sparsità e dimensionalità;
- `total_secondary_use_count` conserva un segnale sintetico sull'esistenza e quantità di usi secondari;
- la scelta rappresenta una compressione del segnale, non una sua eliminazione completa.

---

## 7. Feature di superstruttura

Le feature originali `has_superstructure_*` descrivono materiali e tecniche costruttive dell'edificio.

Decisione finale:

- rimuovere le singole feature originali `has_superstructure_*`;
- mantenere tre feature aggregate:

  - `total_superstructure_count`
  - `has_fragile_material`
  - `has_engineered_structure`

Motivazione:

- le feature di superstruttura sono direttamente collegate alla vulnerabilità sismica;
- le variabili originali sono numerose, binarie e in parte ridondanti;
- gli aggregati conservano un significato fisico chiaro;
- i test diagnostici hanno mostrato che gli aggregati mantengono gran parte del segnale utile con una rappresentazione più compatta.

`has_fragile_material` aggrega la presenza di materiali potenzialmente fragili.

`has_engineered_structure` aggrega la presenza di componenti strutturali più ingegnerizzate.

---

## 8. Feature categoriche strutturali mantenute

Sono mantenute e trattate come categoriche a bassa cardinalità:

- `foundation_type`
- `roof_type`
- `ground_floor_type`
- `other_floor_type`
- `position`
- `land_surface_condition`

Motivazione:

- hanno cardinalità contenuta;
- il costo dimensionale dopo one-hot encoding è limitato;
- alcune rappresentano caratteristiche fisiche o strutturali direttamente collegate alla vulnerabilità sismica;
- i test diagnostici non hanno giustificato la loro rimozione dalla pipeline finale.

Sono invece rimosse:

- `plan_configuration`
- `legal_ownership_status`

Motivazione:

- distribuzione fortemente sbilanciata;
- associazione effettiva debole con il target;
- contributo limitato o non robusto nei test diagnostici;
- riduzione della dimensionalità e della complessità della pipeline.

---

## 9. Feature numeriche mantenute

Sono mantenute nella forma originale:

- `count_floors_pre_eq`
- `count_families`
- `age`

Motivazione:

- `count_floors_pre_eq` rappresenta una caratteristica strutturale diretta dell'edificio;
- `count_families` contiene un segnale debole ma misurabile e ha costo nullo in termini di encoding;
- `age` conserva l'informazione originale sull'età dell'edificio, con gestione specifica del valore speciale `995`.

Non vengono introdotte nella pipeline finale:

- `floor_count_group`
- `family_count_group`

Motivazione:

- le versioni raggruppate rendono le feature più interpretabili, ma comportano perdita di informazione rispetto alle variabili grezze;
- i test diagnostici non hanno mostrato un vantaggio sufficiente per sostituire le variabili originali.

---

## 10. Encoding delle feature geografiche

Le feature geografiche sono:

- `geo_level_1_id`
- `geo_level_2_id`
- `geo_level_3_id`

Queste variabili sono codificate come numeri, ma rappresentano identificativi geografici. Non devono quindi essere interpretate come variabili numeriche continue, ordinali o metriche.

Sono state confrontate quattro strategie:

1. geografiche lasciate come numeriche grezze;
2. geografiche completamente rimosse;
3. strategia ibrida con `geo_level_1_id` one-hot e `geo_level_2_id` / `geo_level_3_id` frequency encoding;
4. strategia con `geo_level_1_id` one-hot, `geo_level_2_id` frequency encoding e rimozione di `geo_level_3_id`.

Decisione finale:

- `geo_level_1_id`: one-hot encoding;
- `geo_level_2_id`: frequency encoding;
- `geo_level_3_id`: frequency encoding.

Motivazione:

- le feature geografiche sono molto informative;
- rimuoverle causa un calo evidente delle performance;
- `geo_level_1_id` ha cardinalità gestibile e può essere trattata con one-hot encoding;
- `geo_level_2_id` e `geo_level_3_id` hanno cardinalità elevata e vengono compressi tramite frequency encoding;
- il one-hot completo delle geografiche granulari produrrebbe una matrice molto ampia e sparsa.

Nota metodologica:

- il frequency encoding deve essere fittato solo sul training set;
- in validazione o test, categorie mai viste nel training vengono mappate a frequenza `0.0`;
- questa scelta evita leakage strutturale tra training e validation.

---

## 11. Preprocessing finale

La pipeline ufficiale di preprocessing è ora implementata come package modulare in:

- `src/preprocessing/`

Il vecchio file:

- `src/preprocessing.py`

resta presente come componente legacy/backward-compatible, ma non rappresenta più la source of truth della pipeline finale.

La pipeline modulare è composta dai seguenti step:

1. `feature_engineering`: crea le feature aggregate tramite `src/features.py`;
2. `DataCleaner`: rimuove identificativi, target accidentale, feature escluse e feature originali ormai compresse;
3. `AgeHandler`: gestisce il valore speciale `age = 995`;
4. `FrequencyEncoder`: applica frequency encoding a `geo_level_2_id` e `geo_level_3_id`;
5. `CategoricalEncoder`: applica one-hot encoding alle categoriche strutturali e a `geo_level_1_id`;
6. `NumericalScaler`: applicato solo quando richiesto, ad esempio per PCA o modelli sensibili alla scala.

La pipeline finale viene costruita tramite:

- `get_preprocessing_pipeline()`: solo preprocessing;
- `make_complete_pipeline()`: preprocessing + eventuale feature selection + eventuale PCA + modello.

La sequenza corretta è:

```text
raw data
→ split train/validation
→ fit preprocessing solo sul training set
→ transform validation/test
→ eventuale FeatureSelector
→ eventuale PCA
→ modello
```

Questa struttura riduce il rischio di data leakage, perché ogni trasformazione che apprende statistiche dai dati viene fittata solo sul training set o sul fold di training durante la cross-validation.

`OutlierCapper` resta disponibile come componente, ma non è incluso nella pipeline standard, perché nella configurazione attuale le colonne su cui agiva vengono già rimosse da `DataCleaner`.

---

## 12. Training pipeline e valutazione baseline

La training pipeline principale è implementata in:

- `src/pipeline_training_model.py`

La funzione principale è:

- `run_training_pipeline()`

Questa funzione consente di configurare:

- feature selection opzionale;
- PCA opzionale;
- sample weighting opzionale;
- strategia di split;
- metodo e soglie della feature selection;
- numero di componenti PCA.

La pipeline lavora sui dati raw e costruisce internamente il flusso:

```text
raw data
→ split
→ preprocessing
→ eventuale FeatureSelector
→ eventuale PCA
→ modello
→ metriche
```

Questo rende il flusso più sicuro rispetto a soluzioni che applicano preprocessing o feature selection sull'intero dataset prima dello split.

### Baseline RandomForest

Con `RandomForestClassifier` e pipeline modulare senza feature selection e senza PCA:

```text
micro-F1:    circa 0.6921
macro-F1:    circa 0.6107
weighted-F1: circa 0.6722
```

Questa baseline è il riferimento operativo attuale per confrontare feature selection, PCA e tuning.

---

## 13. Feature selection

La feature selection è stata integrata come componente opzionale della pipeline tramite:

- `src/featureselector.py`
- `FeatureSelector`

Il selector viene inserito dopo il preprocessing e prima del modello:

```text
preprocessing
→ FeatureSelector
→ model
```

Questa scelta è metodologicamente importante perché evita di fittare la selezione delle feature sull'intero dataset prima dello split.

Il `FeatureSelector` supporta più metodi:

- `rf`: Random Forest importance;
- `xgb`: XGBoost importance;
- `ctb`: CatBoost importance;
- `corr_matrix`: correlazione con il target;
- `chi2`: Chi-square;
- `mu`: mutual information;
- `rlf`: ReliefF.

Decisione operativa:

- usare inizialmente `rf` come metodo principale;
- trattare `xgb`, `ctb`, `rlf` come metodi opzionali, perché dipendono da librerie più pesanti o da configurazioni ambiente specifiche;
- non usare feature selection come default obbligatorio finché non migliora stabilmente rispetto alla configurazione migliore con PCA.

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

- la feature selection è implementata e funzionante;
- la selezione a 30 feature risulta troppo aggressiva;
- la selezione RF a 50 feature migliora leggermente la baseline;
- al momento non supera la configurazione con PCA a 40 componenti.

---

## 14. PCA / dimensionality reduction

La PCA è stata implementata come step opzionale della pipeline.

Decisione:

- PCA non deve essere applicata sui dati grezzi;
- PCA deve essere applicata solo dopo preprocessing numerico e scaling;
- quando `use_pca=True`, la pipeline forza automaticamente `scale_numeric=True`;
- PCA resta un esperimento/configurazione opzionale, non un default obbligatorio.

Motivazione:

- il dataset contiene variabili categoriche, binarie e identificativi geografici;
- applicare PCA direttamente sui dati grezzi sarebbe metodologicamente scorretto;
- PCA può essere utile dopo preprocessing completo, quando la matrice è numerica;
- la PCA permette di testare se una rappresentazione compressa migliora la generalizzazione.

Risultati osservati con RandomForest:

```text
PCA 10:
micro-F1    ≈ 0.6162
macro-F1    ≈ 0.5068
weighted-F1 ≈ 0.5763

PCA 20:
micro-F1    ≈ 0.6840
macro-F1    ≈ 0.5948
weighted-F1 ≈ 0.6650

PCA 30:
micro-F1    ≈ 0.6912
macro-F1    ≈ 0.6100
weighted-F1 ≈ 0.6773

PCA 40:
micro-F1    ≈ 0.6983
macro-F1    ≈ 0.6223
weighted-F1 ≈ 0.6866

PCA 50:
micro-F1    ≈ 0.6980
macro-F1    ≈ 0.6221
weighted-F1 ≈ 0.6864
```

È stato inoltre testato lo scaling senza PCA:

```text
RF + scaling only:
micro-F1    ≈ 0.6921
macro-F1    ≈ 0.6106
weighted-F1 ≈ 0.6722
```

Conclusione:

- il miglioramento non dipende dal solo scaling;
- PCA a 40 componenti è la migliore configurazione osservata finora;
- PCA a 10 componenti è troppo aggressiva;
- PCA 40 è una configurazione candidata per la pipeline finale.

Confronto aggiuntivo:

```text
Feature Selection RF 50 + PCA 40:
micro-F1    ≈ 0.6975
macro-F1    ≈ 0.6204
weighted-F1 ≈ 0.6856
```

La combinazione FS 50 + PCA 40 funziona, ma non supera PCA 40 senza feature selection.

---

## 15. Sample weighting

È stato testato l'uso di pesi bilanciati tramite:

- `compute_sample_weight(class_weight="balanced", y=y_train)`

Decisione:

- `sample_weight` resta disponibile come opzione;
- non viene usato come default.

Motivazione:

- i pesi bilanciati possono aiutare le classi meno frequenti;
- tuttavia, nel test osservato, migliorano la macro-F1 ma peggiorano sensibilmente la micro-F1;
- poiché la metrica principale del progetto è micro-F1, l'uso dei pesi bilanciati non è mantenuto come default.

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

- `sample_weight` è utile come esperimento orientato alla macro-F1;
- non è coerente come default se l'obiettivo principale resta micro-F1.

---

## 16. Modelli opzionali e dipendenze ambiente

La pipeline supporta:

- RandomForest;
- XGBoost;
- LightGBM.

Decisione:

- RandomForest resta il modello baseline stabile;
- XGBoost e LightGBM vengono trattati come modelli opzionali;
- se le dipendenze native non sono disponibili, questi modelli vengono saltati senza bloccare l'intera pipeline.

Motivazione:

- su macOS XGBoost e LightGBM possono richiedere `libomp`;
- un problema di ambiente non deve impedire l'esecuzione della baseline;
- la pipeline deve restare eseguibile anche se i modelli avanzati non sono disponibili localmente.

Nota:

- `lightgbm` è stato aggiunto a `requirements.txt`;
- su macOS potrebbe essere comunque necessario installare `libomp` tramite Homebrew.

---

## 17. Tuning

Il tuning degli iperparametri è stato avviato come area di lavoro separata.

Decisione:

- non integrare direttamente tuning non ancora riallineato alla pipeline attuale;
- il tuning deve essere eseguito sopra la pipeline modulare aggiornata;
- il tuning deve rispettare la sequenza anti-leakage:

```text
split/CV
→ fit preprocessing solo su train/fold
→ eventuale FeatureSelector
→ eventuale PCA
→ model
```

Nota metodologica:

- non è valido fare `fit_transform(X)` sull'intero dataset prima dello split;
- non è valido fare feature selection o tuning su tutto `X, y` prima della validation;
- il criterio principale di tuning dovrebbe essere coerente con la metrica primaria, quindi micro-F1;
- macro-F1 può restare metrica secondaria o obiettivo alternativo dichiarato.

Stato:

- tuning non ancora consolidato nella pipeline finale;
- da riallineare alla versione aggiornata di `make_complete_pipeline()`.

---

## 18. Integrazione con il lavoro di preprocessing parallelo

Nel branch `data_preprocessing` è stata introdotta una cartella separata:

- `preprocessing/`

Questa cartella contiene una pipeline parallela/sperimentale di preprocessing.

Decisione:

- non integrare direttamente questa cartella nella pipeline finale in questa fase;
- mantenere come pipeline ufficiale il package modulare `src/preprocessing/`;
- valutare successivamente se recuperare singole idee o funzioni dal branch parallelo, evitando duplicazioni e conflitti.

Motivazione:

- la pipeline finale del progetto è ora centralizzata in `src/preprocessing/`;
- la cartella `preprocessing/` usa una struttura parallela e non completamente allineata alle decisioni finali;
- l'integrazione diretta rischierebbe di introdurre duplicazione o incoerenza.

---

## 19. Manutenzione repository e pulizia branch

Dopo l'integrazione dei contributi principali nel branch `dev`, il gruppo ha deciso di avviare una pulizia dei branch remoti ormai già mergiati, superati o non più operativi.

Questa decisione non riguarda la metodologia di modellazione, ma l'organizzazione operativa del repository. L'obiettivo è mantenere la repository su GitHub più leggibile, ridurre ambiguità interne al gruppo sui branch attivi e rendere più chiaro quale ramo rappresenti lo stato aggiornato del progetto.

Branch remoti rimossi perché già integrati in `dev`:

- `cleanup/final-integration`
- `feature/03-feature-comprehension`
- `feature/feature-engineering-selection`
- `feature/feature-selection`

Branch remoti rimossi perché superati dalle versioni presenti in `dev` o non più operativi:

- `analisi/01-analisi-dati`
- `baseline/05-baseline-modeling`
- `preprocessing/02_qualita_dati`
- `sistemazione-pipeline-progetto`

La rimozione di questi branch non elimina i commit già integrati o recuperati nel progetto. La cancellazione rimuove soltanto riferimenti remoti non più operativi.

Decisione operativa:

- mantenere `main` come ramo stabile/finale;
- mantenere `dev` come ramo comune di sviluppo aggiornato;
- eliminare branch già integrati, superati o non più operativi;
- recuperare manualmente eventuali contenuti utili prima della cancellazione;
- non usare merge diretti da branch storici quando rischiano di reintrodurre versioni obsolete dei file.

---

## 20. Stato attuale e prossimi step

Alla data di questo aggiornamento:

- `src/features.py` contiene le feature ingegnerizzate mantenute nel feature set finale;
- `src/preprocessing/` implementa la pipeline modulare ufficiale;
- `src/preprocessing.py` resta come file legacy/backward-compatible;
- la pipeline lavora sui dati raw e applica preprocessing, feature selection opzionale, PCA opzionale e modello dentro una pipeline sklearn;
- `AgeHandler` gestisce il valore speciale `age = 995` e crea `is_historic`;
- `FeatureSelector` è integrato in modo opzionale e leak-safe;
- PCA è integrata come step opzionale e la configurazione a 40 componenti è la migliore osservata finora con RandomForest;
- `sample_weight` è disponibile ma non usato come default;
- XGBoost e LightGBM sono gestiti come modelli opzionali;
- la cartella `preprocessing/` resta separata e non viene usata come pipeline finale.

Prossimi step:

- riallineare il tuning alla pipeline aggiornata;
- rieseguire model comparison finale su configurazioni comparabili;
- valutare se usare PCA 40 nella configurazione finale;
- salvare metriche finali in `outputs/metrics/`;
- produrre eventuale final evaluation/submission;
- aggiornare worklog condiviso;
- comunicare al gruppo lo stato aggiornato della pipeline.
