# Decision Log

Questo documento raccoglie le principali decisioni metodologiche prese durante lo sviluppo del progetto **FIA Earthquake Damage Predictor**.

L'obiettivo non è sostituire i notebook, ma mantenere una traccia sintetica delle scelte operative che guidano preprocessing, feature engineering, feature selection e modellazione.

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

Questa metrica dovrà essere usata per confrontare baseline, modelli successivi, feature engineering, feature selection ed eventuali esperimenti di riduzione della dimensionalità.

---

## 2. Gestione di `building_id`

`building_id` è un identificativo tecnico dell'edificio.

Decisione:

- non deve essere usato come feature predittiva;
- deve essere mantenuto solo per merge, tracciamento righe o submission;
- deve essere rimosso prima della modellazione.

Motivazione:

- non rappresenta una proprietà fisica, geografica o strutturale dell'edificio;
- usarlo come variabile predittiva rischierebbe di introdurre rumore o pattern non generalizzabili.

---

## 3. Feature geografiche

Le feature geografiche sono:

- `geo_level_1_id`
- `geo_level_2_id`
- `geo_level_3_id`

Decisione:

- devono essere trattate come variabili categoriche identificative;
- non devono essere trattate come variabili numeriche continue;
- non devono essere scalate come grandezze quantitative ordinate.

Motivazione:

- anche se sono codificate come numeri, rappresentano identificativi geografici;
- il valore numerico non ha significato metrico o ordinale;
- la posizione geografica può catturare differenze territoriali, costruttive, di esposizione o di intensità del danno.

Nota operativa:

- `geo_level_1_id` ha cardinalità più gestibile;
- `geo_level_2_id` e `geo_level_3_id` hanno cardinalità elevata;
- one-hot encoding può essere usato come baseline iniziale, ma potrebbe produrre matrici molto larghe;
- strategie come frequency encoding o target encoding dovranno essere valutate con attenzione;
- target encoding, se usato, dovrà essere applicato solo dentro una procedura di validazione corretta per evitare data leakage.

---

## 4. Gestione di `age = 995`

Durante l'analisi è stato osservato il valore estremo:

- `age = 995`

Decisione:

- non modificare i dati originali in `data/raw/`;
- trattare `age = 995` come valore estremo, anomalo o codificato in modo particolare;
- non usarlo direttamente come se rappresentasse una normale età dell'edificio;
- introdurre feature trasformate dell'età.

Feature candidate introdotte:

- `age_clipped`
- `age_group`

Scelta attuale:

- `age_clipped` limita l'età massima a 200;
- `age_group` discretizza l'età in classi interpretabili.

Motivazione:

- `age = 995` può distorcere statistiche, scaling e modelli sensibili ai valori estremi;
- clipping e discretizzazione mantengono informazione utile sull'età, riducendo l'effetto degli outlier.

Stato:

- decisione operativa ragionevole per la pipeline iniziale;
- impatto da validare quantitativamente tramite micro-F1.

---

## 5. Feature engineering candidate

Sono state proposte feature ingegnerizzate interpretabili, costruite a partire dalle variabili originali.

Feature candidate:

- `age_clipped`
- `age_group`
- `building_volume_proxy`
- `has_engineered_structure`
- `has_fragile_material`
- `total_secondary_use_count`
- `total_superstructure_count`

Decisione:

- mantenere queste feature nella prima versione del dataset arricchito;
- valutarne l'impatto tramite feature selection e confronto di modelli;
- non considerarle definitive fino alla validazione quantitativa.

---

## 6. `has_fragile_material`

La feature `has_fragile_material` aggrega la presenza di materiali potenzialmente fragili.

Materiali considerati:

- `has_superstructure_adobe_mud`
- `has_superstructure_mud_mortar_stone`
- `has_superstructure_stone_flag`
- `has_superstructure_mud_mortar_brick`

Decisione:

- mantenere `has_fragile_material` come feature candidata.

Motivazione:

- rappresenta un'informazione strutturale interpretabile;
- è coerente con l'ipotesi che materiali più fragili siano associati a maggiore vulnerabilità sismica;
- nel notebook di feature engineering mostra una relazione preliminare forte con `damage_grade`.

Stato:

- feature promettente dal punto di vista interpretativo;
- impatto da validare con micro-F1.

---

## 7. `has_engineered_structure`

La feature `has_engineered_structure` aggrega la presenza di componenti strutturali più ingegnerizzate.

Componenti considerati:

- `has_superstructure_cement_mortar_stone`
- `has_superstructure_cement_mortar_brick`
- `has_superstructure_rc_non_engineered`
- `has_superstructure_rc_engineered`

Decisione:

- mantenere `has_engineered_structure` come feature candidata.

Motivazione:

- rappresenta un indicatore sintetico di maggiore robustezza strutturale;
- dovrebbe essere più frequente negli edifici meno danneggiati;
- nel notebook di feature engineering mostra una relazione preliminare coerente con questa ipotesi.

Stato:

- feature promettente dal punto di vista interpretativo;
- impatto da validare con micro-F1.

---

## 8. `building_volume_proxy`

La feature `building_volume_proxy` è definita come `area_percentage * height_percentage`.

Decisione:

- mantenere `building_volume_proxy` come feature candidata.

Motivazione:

- fornisce una proxy dimensionale dell'edificio;
- può catturare informazioni non rappresentate separatamente da area e altezza;
- la relazione con il danno potrebbe non essere lineare.

Stato:

- feature interpretabile;
- utilità predittiva da validare tramite modelli.

---

## 9. Conteggi aggregati

Sono state introdotte due feature di conteggio:

- `total_superstructure_count`
- `total_secondary_use_count`

Decisione:

- mantenerle entrambe come feature candidate nella prima pipeline.

Motivazione:

- `total_superstructure_count` sintetizza il numero di materiali o tecniche costruttive presenti;
- `total_secondary_use_count` sintetizza il numero di usi secondari specifici;
- queste feature possono catturare complessità strutturale o funzionale dell'edificio.

Stato:

- utilità da validare quantitativamente;
- possibile contributo diverso tra modelli lineari e modelli tree-based.

---

## 10. Preprocessing iniziale

Decisioni operative per il preprocessing iniziale:

- rimuovere `building_id`;
- rimuovere `damage_grade` dalle feature se accidentalmente presente;
- aggiungere feature ingegnerizzate tramite `src/features.py`;
- trattare feature categoriche testuali e feature geografiche come categoriche;
- trattare `age_group` come categorica;
- applicare one-hot encoding come baseline iniziale;
- applicare scaling solo dove necessario, ad esempio per modelli lineari o metodi sensibili alla scala;
- usare split stratificato per mantenere la distribuzione delle classi.

Implementazione attuale:

- `src/features.py`
- `src/preprocessing.py`
- `src/data_loader.py`

Stato:

- implementazione validata tramite smoke test;
- pipeline ancora da estendere per model comparison, feature selection e tuning.

---

## 11. PCA / dimensionality reduction

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
- da trattare dopo una baseline e una pipeline di preprocessing stabile.

---

## 12. Feature selection

La feature selection dovrà verificare quantitativamente quali feature contribuiscono davvero alla performance.

Tecniche candidate:

- confronto tra dataset originale e dataset arricchito;
- feature importance da modelli tree-based;
- permutation importance;
- SelectKBest o metodi analoghi;
- confronto micro-F1 con tutte le feature e con subset selezionato.

Decisione:

- le feature candidate non sono definitive;
- devono essere validate con modelli e micro-F1;
- l'interpretabilità resta importante, ma non sostituisce la valutazione quantitativa.

---

## 13. Decisioni non ancora definitive

Restano da decidere o validare:

- encoding finale per `geo_level_2_id` e `geo_level_3_id`;
- impatto quantitativo di `age_clipped` e `age_group`;
- reale contributo di `has_fragile_material`;
- reale contributo di `has_engineered_structure`;
- utilità di `building_volume_proxy`;
- utilità dei conteggi aggregati;
- modello migliore per la pipeline finale;
- eventuale uso di PCA;
- strategia di feature selection;
- strategia di tuning;
- procedura finale di generazione submission.

---

## 14. Stato attuale

Alla data di questo documento:

- notebook 03 ha prodotto le ipotesi di feature comprehension;
- notebook 04 ha implementato e analizzato feature ingegnerizzate interpretabili;
- `src/` contiene una base funzionante per data loading, feature engineering, preprocessing, evaluation e utility;
- notebook 05 è stato revisionato come baseline preliminare, ma richiede una piccola correzione per compatibilità con la versione attuale di scikit-learn;
- le prossime fasi saranno feature selection, model comparison, eventuale PCA e tuning.
