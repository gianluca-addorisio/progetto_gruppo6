# Decision Log

Questo documento raccoglie le principali decisioni metodologiche prese durante lo sviluppo del progetto **FIA Earthquake Damage Predictor**.

L'obiettivo non è sostituire i notebook, ma mantenere una traccia sintetica, ordinata e aggiornata delle scelte operative che guidano preprocessing, feature engineering, feature selection, PCA, model comparison, tuning e scelta del modello finale.

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

La micro-F1 viene usata come criterio principale per confrontare baseline, modelli avanzati, feature engineering, feature selection, PCA ed esperimenti di tuning.

Metriche quantitative di supporto:

- macro-F1;
- weighted-F1.

Per l'analisi qualitativa degli errori e delle prestazioni per classe potranno essere usati anche:

- classification report;
- confusion matrix.

Decisione metodologica:

- la scelta del modello finale deve privilegiare la micro-F1;
- macro-F1 e weighted-F1 sono utili per interpretare la stabilità per classe, ma non sostituiscono la metrica principale.

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
- `area_percentage`
- `height_percentage`
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

Nella pipeline modulare aggiornata viene inoltre creata la feature:

- `is_historic`

Questa feature è prodotta da `AgeHandler` per distinguere i casi con `age = 995`.

Decisione aggiornata:

- `area_percentage` e `height_percentage` vengono mantenute;
- `building_volume_proxy` non viene mantenuta nella pipeline finale attuale.

Motivazione sintetica:

- `area_percentage` e `height_percentage` sono feature originali, semplici e interpretabili;
- `building_volume_proxy` è una combinazione ridondante delle due;
- il vantaggio prestazionale osservato mantenendo anche `building_volume_proxy` è risultato minimo;
- la pipeline finale preferisce quindi una scelta più pulita e metodologicamente difendibile.

---

## 4. Feature dimensionali

Sono state confrontate due strategie principali per le feature dimensionali:

### Variante A: `all_dims`

Questa variante mantiene:

- `area_percentage`
- `height_percentage`
- `building_volume_proxy`

Risultato osservato su XGBoost:

```text
micro-F1 ≈ 0.742273
```

### Variante C: `original_dims`

Questa variante mantiene:

- `area_percentage`
- `height_percentage`

e rimuove:

- `building_volume_proxy`

Risultato osservato su XGBoost:

```text
micro-F1 = 0.741889
```

Decisione finale:

- mantenere `area_percentage`;
- mantenere `height_percentage`;
- rimuovere `building_volume_proxy`.

Motivazione:

- la differenza tra le due varianti è molto piccola;
- `building_volume_proxy = area_percentage * height_percentage` è ridondante rispetto alle due feature originali;
- mantenere le due feature originali consente al modello di apprendere autonomamente eventuali relazioni utili;
- la scelta `original_dims` è più pulita, più interpretabile e più semplice da giustificare nel report.

Nota:

- la decisione non viene presentata come scelta che massimizza in modo assoluto la performance;
- viene adottata come trade-off tra performance quasi equivalente, minore ridondanza e maggiore chiarezza metodologica.

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
- `age_clipped` e `age_group` derivano da `age` e non aggiungono informazione indipendente sufficiente;
- `age_group` introduce soglie arbitrarie;
- `is_historic` conserva in modo esplicito l'informazione che `995` rappresenta un caso speciale;
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
- `area_percentage`
- `height_percentage`

Motivazione:

- `count_floors_pre_eq` rappresenta una caratteristica strutturale diretta dell'edificio;
- `count_families` contiene un segnale debole ma misurabile e ha costo nullo in termini di encoding;
- `age` conserva l'informazione originale sull'età dell'edificio, con gestione specifica del valore speciale `995`;
- `area_percentage` e `height_percentage` descrivono caratteristiche dimensionali originali e interpretabili.

Non vengono introdotte nella pipeline finale:

- `floor_count_group`
- `family_count_group`
- `building_volume_proxy`

Motivazione:

- le versioni raggruppate rendono le feature più interpretabili, ma comportano perdita di informazione rispetto alle variabili grezze;
- `building_volume_proxy` è ridondante rispetto ad `area_percentage` e `height_percentage`;
- i test non hanno mostrato un vantaggio sufficiente per giustificare il mantenimento di queste variabili derivate nella configurazione finale attuale.

---

## 10. Encoding delle feature geografiche

Le feature geografiche sono:

- `geo_level_1_id`
- `geo_level_2_id`
- `geo_level_3_id`

Queste variabili sono codificate come numeri, ma rappresentano identificativi geografici. Non devono quindi essere interpretate come variabili numeriche continue, ordinali o metriche.

Sono state confrontate più strategie:

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

La pipeline ufficiale di preprocessing è implementata come package modulare in:

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

## 12. Training pipeline e model comparison

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
- numero di componenti PCA;
- tuning opzionale;
- selezione esplicita dei modelli da eseguire tramite `models_to_run`.

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

### Baseline avanzata corrente

Configurazione:

```text
feature_selection = False
use_pca = False
do_tuning = False
use_sample_weight = False
split_strategy = 2
feature dimensionali = original_dims
```

Risultati:

```text
RandomForest      micro-F1 0.716160 | macro-F1 0.645689 | weighted-F1 0.704259
XGBoost           micro-F1 0.741889 | macro-F1 0.687747 | weighted-F1 0.735980
LightGBM          micro-F1 0.727615 | macro-F1 0.664767 | weighted-F1 0.719590
VotingEnsemble    micro-F1 0.735462 | macro-F1 0.673930 | weighted-F1 0.727158
StackingEnsemble  micro-F1 0.741851 | macro-F1 0.688311 | weighted-F1 0.736433
```

Decisione:

- il candidato principale attuale è `XGBoost`, senza PCA, senza feature selection e senza tuning;
- `StackingEnsemble` è un'alternativa quasi equivalente, con macro-F1 e weighted-F1 leggermente migliori ma micro-F1 lievemente inferiore;
- poiché la metrica principale è micro-F1, `XGBoost` resta il candidato più difendibile;
- `XGBoost` è anche più semplice da spiegare e mantenere rispetto a uno stacking ensemble.

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

### Feature selection RF a 30 feature

Configurazione:

```text
feature_selection = True
fs_method = "rf"
max_features_to_hold = 30
use_pca = False
do_tuning = False
```

Risultati:

```text
RandomForest      micro-F1 0.713858
XGBoost           micro-F1 0.738109
LightGBM          micro-F1 0.721859
VotingEnsemble    micro-F1 0.731529
StackingEnsemble  micro-F1 0.737380
```

Conclusione:

- la feature selection RF a 30 feature peggiora rispetto alla configurazione senza feature selection;
- la selezione a 30 feature è probabilmente troppo aggressiva;
- la feature selection resta disponibile come strumento, ma non viene adottata nella pipeline finale attuale.

Nota:

- eventuali test con `max_features_to_hold = 50` o con metodi `xgb`/`ctb` possono essere considerati esperimenti aggiuntivi;
- non devono però essere assunti come default senza un miglioramento quantitativo stabile.

---

## 14. PCA / dimensionality reduction

La PCA è stata implementata come step opzionale della pipeline.

Decisione metodologica:

- PCA non deve essere applicata sui dati grezzi;
- PCA deve essere applicata solo dopo preprocessing numerico e scaling;
- quando `use_pca=True`, la pipeline forza automaticamente lo scaling numerico;
- PCA resta un esperimento secondario, non una scelta finale.

Motivazione:

- il dataset contiene variabili categoriche, binarie e identificativi geografici;
- applicare PCA direttamente sui dati grezzi sarebbe metodologicamente scorretto;
- PCA può essere utile solo dopo preprocessing completo, quando la matrice è numerica;
- nel test corrente, però, la rappresentazione PCA peggiora sensibilmente rispetto alle feature originali preprocessate.

### Test PCA 40

Configurazione:

```text
feature_selection = False
use_pca = True
pca_n_components = 40
do_tuning = False
```

Risultati:

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
- può essere documentata come esperimento secondario collegato agli argomenti del corso, ma non come componente utile alla performance finale.

---

## 15. Sample weighting

È stato testato l'uso di pesi bilanciati tramite:

- `compute_sample_weight(class_weight="balanced", y=y_train)`

Decisione:

- `sample_weight` resta disponibile come opzione;
- non viene usato come default.

Motivazione:

- i pesi bilanciati possono aiutare le classi meno frequenti;
- tuttavia possono penalizzare la micro-F1;
- poiché la metrica principale del progetto è micro-F1, l'uso dei pesi bilanciati non viene mantenuto come default.

Nota:

- eventuali miglioramenti su macro-F1 vanno sempre discussi separatamente;
- non devono prevalere automaticamente sulla micro-F1.

---

## 16. Modelli avanzati e dipendenze ambiente

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

Motivazione:

- XGBoost ha ottenuto la micro-F1 migliore nella baseline avanzata corrente;
- StackingEnsemble è quasi equivalente, ma più complesso;
- per il report universitario è preferibile una soluzione con rapporto chiaro tra performance, semplicità e difendibilità.

Nota per macOS:

- XGBoost e LightGBM possono richiedere `libomp`;
- su Mac ARM può essere necessario installare `libomp` tramite Homebrew:

```bash
brew install libomp
```

- `catboost` è usato per alcuni metodi opzionali di feature selection e deve essere installato se si usano configurazioni basate su CatBoost.

---

## 17. Tuning

Il tuning degli iperparametri è stato integrato e testato in più configurazioni.

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

Conclusione:

- tuning + feature selection peggiora molto rispetto alla baseline avanzata;
- una probabile causa è che il tuner seleziona spesso pochissime feature;
- questa configurazione non viene adottata come finale.

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

Conclusione:

- il tuning senza feature selection funziona tecnicamente dopo il fix;
- non batte la baseline avanzata;
- RandomForest migliora rispetto alla propria baseline, ma XGBoost, LightGBM ed ensemble peggiorano.

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

- XGBoost tuned peggiora rispetto a XGBoost default;
- la configurazione finale attuale non usa tuning.

Decisione finale sul tuning:

- il tuning resta disponibile come strumento;
- gli esperimenti finora non giustificano il suo uso nella pipeline finale;
- la pipeline candidata usa parametri default dei modelli avanzati.

---

## 18. Integrazione con lavoro parallelo e branch remoti

Sono stati ispezionati contributi provenienti da branch paralleli, in particolare:

- `origin/nicola_branch`
- `origin/mattia_final_pipeline`

Parte del lavoro avanzato di Nicola è stata integrata selettivamente nel branch corrente.

Il branch `origin/mattia_final_pipeline` è stato ispezionato ma non mergiato automaticamente.

Motivi principali:

- rimuove `models_to_run`, utile per eseguire test selettivi;
- reintroduce una gestione del tuning sempre basata su `FeatureSelectionTuner`;
- elimina il metodo `ModelTuner.tune_pipeline`, necessario per il tuning senza feature selection;
- ripristina controlli booleani rischiosi sugli estimator sklearn negli ensemble;
- mantiene `building_volume_proxy`, in contrasto con la decisione metodologica attuale;
- i risultati riportati non superano la baseline avanzata corrente.

Decisione:

- non fare merge alla cieca di branch remoti;
- recuperare solo singole idee utili;
- mantenere separati eventuali sviluppi nuovi, come generazione automatica della submission.

Nota su `make_submission`:

- l'idea di aggiungere una modalità `make_submission=True` è utile;
- deve però essere implementata in una patch separata;
- non deve essere hardcodato `StackingEnsemble` come modello finale se la metrica principale individua XGBoost come miglior modello corrente;
- una soluzione più robusta dovrebbe consentire di scegliere esplicitamente il modello finale o selezionarlo in base a `micro_f1`.

---

## 19. Manutenzione repository e pulizia branch

La manutenzione del repository deve seguire una logica prudente.

Decisione operativa:

- mantenere `main` come ramo stabile/finale;
- mantenere `dev` come ramo comune di sviluppo aggiornato;
- usare branch personali o dedicati per modifiche non banali;
- aprire PR verso `dev` solo dopo test minimo e documentazione coerente;
- evitare merge diretti da branch storici o paralleli quando rischiano di reintrodurre versioni obsolete dei file;
- controllare sempre `git status -sb`, `git diff --stat` e, se necessario, `git diff` prima di commit, merge o push.

Situazione corrente del branch personale:

```text
branch: gianluca
ahead rispetto a origin/gianluca: 20 commit
working tree: pulito dopo il commit dei log
```

Ultimi commit rilevanti:

```text
fdbc6f7 Corregge log feature selection nella pipeline
22067a1 Reso selettiva la pipeline di training
6902377 Corregge tuning senza feature selectin
8a9363e Integra modelli avanzati, tuning e risultati metriche
```

Nota:

- non bisogna pushare o aprire PR finché README e decision log non sono coerenti con i risultati sperimentali correnti;
- la pulizia dei branch remoti obsoleti va fatta solo dopo consolidamento e accordo del gruppo.

---

## 20. Scelta candidata finale

Alla data di questo aggiornamento, la pipeline candidata finale è:

```text
Modello: XGBoost
Feature selection: no
PCA: no
Tuning: no
Sample weighting: no
Feature dimensionali: original_dims
Split strategy: 2
```

Metriche:

```text
micro-F1    0.741889
macro-F1    0.687747
weighted-F1 0.735980
```

Alternativa quasi equivalente:

```text
Modello: StackingEnsemble
Feature selection: no
PCA: no
Tuning: no
Sample weighting: no
```

Metriche:

```text
micro-F1    0.741851
macro-F1    0.688311
weighted-F1 0.736433
```

Decisione:

- XGBoost è preferito perché ottiene la micro-F1 più alta;
- la differenza rispetto a StackingEnsemble è minima;
- StackingEnsemble ottiene macro-F1 e weighted-F1 leggermente migliori, ma è più complesso;
- XGBoost è quindi più difendibile come modello finale nel report, salvo ulteriori test che cambino chiaramente il quadro.

---

## 21. Stato attuale e prossimi step

Alla data di questo aggiornamento:

- `src/features.py` contiene le feature ingegnerizzate mantenute nel feature set finale;
- `src/preprocessing/` implementa la pipeline modulare ufficiale;
- `src/preprocessing.py` resta come file legacy/backward-compatible;
- la pipeline lavora sui dati raw e applica preprocessing, feature selection opzionale, PCA opzionale e modello dentro una pipeline sklearn;
- `AgeHandler` gestisce il valore speciale `age = 995` e crea `is_historic`;
- `FeatureSelector` è integrato in modo opzionale e leak-safe;
- PCA è integrata come step opzionale, ma non è adottata nella pipeline finale attuale perché peggiora sensibilmente rispetto alla baseline avanzata;
- `sample_weight` è disponibile ma non usato come default;
- XGBoost, LightGBM e CatBoost richiedono attenzione alle dipendenze ambiente, soprattutto su macOS;
- `models_to_run` consente test selettivi sui modelli;
- il tuning funziona tecnicamente anche senza feature selection, ma non migliora la baseline avanzata corrente;
- il candidato finale attuale è XGBoost senza PCA, senza feature selection e senza tuning.

Prossimi step:

- aggiornare `README.md` in modo coerente con questo decision log;
- creare una tabella esperimenti unica e coerente in `outputs/metrics/` o nella documentazione;
- decidere se implementare `make_submission` come patch separata;
- eseguire test finale minimo della pipeline;
- pushare il branch `gianluca` solo dopo documentazione coerente;
- aprire PR verso `dev`;
- preparare materiale per report e presentazione finale.

## Decisione finale dopo submission pubblica

Dopo la generazione della submission finale sono stati testati i due candidati più competitivi:

| Modello | Configurazione | Public score |
|---|---|---:|
| XGBoost | no FS, no PCA, no tuning | 0.7397 |
| StackingEnsemble | no FS, no PCA, no tuning | 0.7384 |

La scelta finale rimane `XGBoost`, perché:
- ottiene il miglior risultato nella validazione interna;
- ottiene il miglior public score tra i candidati testati;
- è più semplice da difendere e spiegare rispetto allo stacking.

Feature selection, tuning e PCA restano documentati come esperimenti svolti, ma non vengono adottati nella configurazione finale perché peggiorano i risultati validati.

