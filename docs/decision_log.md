# Decision Log

Questo documento raccoglie le principali decisioni metodologiche prese durante lo sviluppo del progetto **FIA Earthquake Damage Predictor**.

L'obiettivo non è sostituire i notebook, ma mantenere una traccia sintetica, ordinata e aggiornata delle scelte operative che guidano preprocessing, feature engineering, feature selection e modellazione.

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

La micro-F1 viene usata per confrontare baseline, modelli successivi, feature engineering, feature selection ed eventuali esperimenti di riduzione della dimensionalità.

Come metriche quantitative di supporto vengono considerate anche:

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

## 3. Feature set finale compatto

Dopo l'analisi di feature selection e i test diagnostici sulle principali famiglie di variabili, è stata adottata una versione compatta e interpretabile della feature matrix finale.

L'obiettivo non è massimizzare ogni minimo incremento locale di performance, ma costruire una pipeline stabile, leggibile e difendibile, mantenendo le informazioni strutturali e geografiche principali.

### Feature finali prima dell'encoding

La feature matrix preparata contiene 17 colonne prima dell'encoding:

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
- rimuovere `age_group`.

Motivazione:

- `age` conserva direttamente l'informazione sull'età dell'edificio;
- `age_clipped` e `age_group` derivano da `age` e non aggiungono informazione indipendente;
- `age_group` introduce soglie arbitrarie;
- `age_clipped` modifica la distribuzione originale solo nella coda estrema;
- i test diagnostici non hanno mostrato un beneficio robusto delle feature derivate rispetto alla variabile originale.

Nota:

- il valore `age = 995` resta riconosciuto come valore estremo o codificato in modo particolare;
- non vengono modificati i dati originali in `data/raw/`.

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
- `age` conserva l'informazione originale sull'età dell'edificio.

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

La pipeline finale di preprocessing è implementata in:

- `src/features.py`
- `src/preprocessing.py`

Operazioni principali:

1. rimozione di identificativi e target se presenti;
2. creazione delle feature aggregate e del proxy dimensionale;
3. rimozione delle feature escluse o compresse;
4. one-hot encoding delle categoriche a bassa cardinalità;
5. one-hot encoding di `geo_level_1_id`;
6. frequency encoding di `geo_level_2_id` e `geo_level_3_id`;
7. passthrough o scaling delle feature numeriche a seconda del modello.

Il preprocessing produce una matrice compatta, coerente con le decisioni di feature selection e adatta alla model comparison successiva.

---

## 12. Smoke test della pipeline finale

Dopo l'implementazione della pipeline finale è stato eseguito uno smoke test con:

- split train/validation stratificato;
- `DecisionTreeClassifier(max_depth=12)`;
- metriche micro-F1, macro-F1 e weighted-F1.

Risultati osservati:

- feature prima del preprocessing: 17;
- feature dopo encoding/preprocessing: 65;
- micro-F1: circa `0.70265`;
- macro-F1: circa `0.62362`;
- weighted-F1: circa `0.68989`.

Il test conferma che:

- la pipeline funziona correttamente;
- la strategia geografica ibrida è operativa;
- la feature matrix finale mantiene una dimensionalità contenuta;
- le performance restano coerenti con i test diagnostici precedenti.

---

## 13. PCA / dimensionality reduction

Decisione:

- PCA non deve essere applicata sui dati grezzi;
- PCA può essere considerata solo come esperimento secondario;
- PCA deve essere applicata solo dopo preprocessing numerico e scaling.

Motivazione:

- il dataset contiene variabili categoriche, binarie e identificativi geografici;
- applicare PCA direttamente sui dati grezzi sarebbe metodologicamente scorretto;
- PCA può essere utile come confronto didattico/metodologico, ma non è il cuore della pipeline.

Esperimento previsto:

- preparare matrice numerica dopo preprocessing;
- applicare scaling;
- applicare PCA;
- valutare varianza spiegata;
- confrontare micro-F1 con e senza PCA.

Stato:

- non ancora implementata;
- da trattare dopo una pipeline di preprocessing stabile e una prima model comparison.

---

## 14. Feature selection e model comparison

La feature selection è stata usata per arrivare a un feature set compatto e interpretabile.

Le prossime fasi dovranno verificare quantitativamente la robustezza delle scelte effettuate attraverso:

- confronto tra modelli diversi;
- feature importance da modelli tree-based;
- eventuale permutation importance;
- confronto micro-F1 con e senza eventuali gruppi di feature;
- valutazione della stabilità delle performance.

Modelli candidati per la model comparison:

- Logistic Regression;
- Decision Tree;
- Random Forest;
- ExtraTrees;
- Gradient Boosting;
- eventuale XGBoost o LightGBM se compatibile con l'ambiente.

---

## 15. Integrazione con il lavoro di preprocessing parallelo

Nel branch `data_preprocessing` è stata introdotta una cartella separata:

- `preprocessing/`

Questa cartella contiene una pipeline parallela/sperimentale di preprocessing.

Decisione:

- non integrare direttamente questa cartella nella pipeline finale in questa fase;
- mantenere come pipeline stabile per la modellazione il codice contenuto in `src/`;
- valutare successivamente se recuperare singole idee o funzioni dal branch parallelo, evitando duplicazioni e conflitti.

Motivazione:

- la pipeline finale del progetto è già centralizzata in `src/`;
- la cartella `preprocessing/` usa una struttura parallela e non ancora allineata alle decisioni finali di feature selection;
- l'integrazione diretta rischierebbe di introdurre duplicazione o incoerenza.

---

## 16. Manutenzione repository e pulizia branch

Dopo l'integrazione dei contributi principali nel branch `dev`, il gruppo ha deciso di avviare una pulizia dei branch remoti ormai già mergiati, superati o non più operativi.

Questa decisione non riguarda la metodologia di modellazione, ma l'organizzazione operativa del repository. L'obiettivo è mantenere la repository su GitHub più leggibile, ridurre ambiguità interne al gruppo sui branch attivi e rendere più chiaro quale ramo rappresenti lo stato aggiornato del progetto.

Branch remoti rimossi perché già integrati in `dev`:

- `cleanup/final-integration`: branch di integrazione intermedia usato per raccogliere la struttura comune del progetto, i notebook già revisionati, la documentazione iniziale e le utility condivise.
- `feature/03-feature-comprehension`: branch relativo al contributo sul notebook 03, dedicato alla comprensione semantica delle feature e alle note operative preliminari.
- `feature/feature-engineering-selection`: branch relativo al notebook 04, alla feature engineering e alla prima versione del decision log metodologico.
- `feature/feature-selection`: branch relativo al notebook 06 e alla prima analisi di feature selection, poi integrato nel flusso principale di sviluppo.

Branch remoti rimossi perché superati dalle versioni presenti in `dev` o non più operativi:

- `analisi/01-analisi-dati`: branch relativo a una versione precedente del notebook 01. La versione presente in `dev` risulta più aggiornata e completa.
- `baseline/05-baseline-modeling`: branch relativo a una versione precedente del notebook 05. La versione presente in `dev` contiene modifiche successive e risulta più aggiornata.
- `preprocessing/02_qualita_dati`: branch relativo al notebook 02. La parte utile, in particolare la sezione finale di riepilogo, viene recuperata manualmente nel notebook presente su `dev`, evitando il merge diretto del branch vecchio.
- `sistemazione-pipeline-progetto`: branch storico usato per sistemare struttura, README, `.gitignore` e organizzazione iniziale della pipeline. Il contenuto utile risulta ormai superato o già assorbito nella struttura attuale di `dev`.

La rimozione di questi branch non elimina i commit già integrati o recuperati nel progetto. La cancellazione rimuove soltanto riferimenti remoti non più operativi.

Decisione operativa:

- mantenere `main` come ramo stabile/finale;
- mantenere `dev` come ramo comune di sviluppo aggiornato;
- eliminare branch già integrati, superati o non più operativi;
- recuperare manualmente eventuali contenuti utili prima della cancellazione;
- non usare merge diretti da branch storici quando rischiano di reintrodurre versioni obsolete dei file.

---

## 17. Stato attuale e prossimi step

Alla data di questo aggiornamento:

- `src/features.py` contiene solo le feature ingegnerizzate mantenute nel feature set finale;
- `src/preprocessing.py` implementa la pipeline finale compatta, incluso l'encoding geografico ibrido;
- il feature set finale è stato validato con smoke test;
- la cartella `preprocessing/` resta separata e non viene usata come pipeline finale;
- il repository è stato pulito dai branch remoti superati.

Prossimi step:

- eseguire model comparison sulla pipeline aggiornata;
- valutare eventuale PCA come esperimento secondario;
- procedere con tuning e final evaluation;
- coordinare l'integrazione del lavoro sviluppato sul branch `gianluca` con il lavoro di Claudia presente nel branch `data_preprocessing`, utilizzando il branch dedicato `merge_preprocessing` creato da Nicola per gestire il merge tra le due linee di sviluppo.