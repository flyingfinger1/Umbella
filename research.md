# Research Notes: 2D→3D Bridge für Pflanzenerkennung

Diese Datei sammelt den inhaltlichen Stand: Warum dieses Projekt, was gibt es bereits, wo sind die Lücken, und welche technischen Ansätze kommen in Frage. Wird im Laufe der Recherche fortgeschrieben.

## Arbeitstagebuch

Chronologische Liste der konkreten Schritte. Format: Datum — Stichpunkt — Verweis (Datei oder Abschnitt). Kompakt halten; ausführliche Begründungen gehören in die thematischen Abschnitte unten.

### 2026-05-03 — Erste Session

- **Recherche Datensatz-Lizenzen** (Pheno4D, ROSE-X, Plant3D) — siehe Tabelle "Datensatz-Recherche".
- **Pheno4D-Größe via HEAD-Request** verifiziert: 4.44 GB.
- **ROSE-X-Größe via Range-GET** verifiziert: 1.53 GB (Nextcloud-HEAD wird vom Server geblockt).
- **Recherche existierende Apiaceae-3D-Daten** → keine vorhanden. Bestätigt synthetisch + Eigenaufnahmen als einzigen Pfad.
- **SOTA-Recherche single-view plant reconstruction (2024–2026)** → siehe Abschnitt "SOTA single-view plant reconstruction". Wichtigste Befunde: TreeFormer/Tree-D-Fusion machen unsere Methode für Bäume; Apiaceae-Lücke offen; bestimmungs-basierte Endmetrik ist unser Differenzierer.
- **Gaussian-Splatting-Eignungsanalyse** → siehe Abschnitt "Gaussian Splatting für Pflanzen". Empfohlener Stack: Skelett-/Parameter-Vorhersage primär, 3DGS optional als Render-Backend.
- **L-System-Parameter-Kurzfassung** im Chat erklärt (Phyllotaxis, Verzweigungswinkel, Internodien-Länge, Strahlen pro Dolde etc.) — hier nicht eingetragen, weil Standard-Wissen.
- **Test-Set-Methodik entworfen** → siehe Abschnitt "Test-Set & Bestimmungs-Metriken". Vorschlag: 15–30 Apiaceae-Arten, ~3000 Bilder, MRR auf "kritischem Verwechslungs-Subset" als Hauptmetrik.
- **Pheno4D + ROSE-X heruntergeladen** nach `data/raw/`. SHA256 für Integrität festgehalten:
    - Pheno4D.zip: `5e7d13db8fb08aad05fefdd5dc641482bd1a1f2f76eec472b5e8357630f5daeb`
    - ROSE-X.zip:  `1afe6180f9313c5c0a4d2b453da581e75baef8aaec7b28a039ca2dfcf398ae3c`
- **Pheno4D entpackt** nach `data/Pheno4D/` (12.4 GB, 238 Dateien, 14 Pflanzen).
- **Projektgerüst angelegt:** `.venv` (Python 3.14, numpy 2.4.4, plotly 6.7.0, später scipy 1.17.1), `src/datasets/`, `src/geometry/`, `notebooks/`, `.gitignore`.
- **Format-Eigenheit von Pheno4D entdeckt:** annotierte Dateien haben *kombiniertes* Label (`0=soil, 1=stem, ≥2=Blatt-Instanz-ID`), nicht separate semantic+instance. Tomato hat 4 Spalten, Maize 5 (4. Spalte ist Zusatzkennung, letzte ist combined). Loader entsprechend angepasst.
- **Loader gebaut** — `src/datasets/pheno4d.py` mit `Pheno4DCloud` dataclass, exposes `xyz`, `instance` (raw combined), `semantic` (mapped to 0/1/2).
- **Notebook 01 — Pheno4D-Exploration** (`notebooks/01_explore_pheno4d.py`): Plant-Liste, eine annotierte Wolke laden, Klassen-Verteilung, Plotly-3D-HTML.
- **Notebook 02 — Wachstumsanimation** (`notebooks/02_growth_timeseries.py`): Slider durch alle annotierten Zeitschritte einer Pflanze. Iterativ verbessert: Soil ausblendbar; Bbox per Frame gelockt (manueller Aspectratio + autorange=False, sonst springt die Kamera).
- **Notebook 03 — Blatt-Instanzen** (`notebooks/03_leaf_instances.py`): per-Blatt-Einfärbung, einzelnes Blatt isolieren, PCA-basierte Stats (Bbox, Length/Width/Thickness-Ratio).
- **Notebook 04 — Skelett-Extraktion** (`notebooks/04_skeleton.py`):
    - **v1**: PCA + Bin-Median pro Organ → Polylines, Blatt-Basen via Nearest-Neighbor an Stamm angedockt.
    - **v2 (Qualitätssprung)**: PCA durch **geodätische Mittelachse** ersetzt (voxel-downsample → kNN-Graph → Dijkstra zwischen Endpunkten → äquidistantes Resampling → leichtes Glätten). Robust gegenüber gebogenen Blättern.
    - **v3 (Verzweigung)**: Stamm-Polyline durch **verzweigtes Steiner-Tree-Skelett** ersetzt (greedy farthest-point auf multi-source Dijkstra). Tomato03_0325 jetzt 352 Knoten / 351 Kanten (saubere Tree-Struktur), 6 Stamm-Verzweigungen, 31 Blätter angekoppelt. Datenstruktur (`Skeleton` mit `nodes/edges/node_organ/node_role`) blieb über alle drei Versionen stabil.
- **Skeleton-Modul** in `src/geometry/skeleton.py`: `extract_polyline` (geodätisch, für Blätter), `extract_branched_skeleton` (Steiner-Tree, für Stamm), `extract_plant_skeleton` (kombiniert beides).
- **Notebook 05 — Skelett-Diagnose** (`notebooks/05_skeleton_probe.py`): einzelnes Blatt vollständig (kein Subsampling) plus zugehöriges Skelett, plus Distanz-Reports zur Hauptpunktwolke und zum Stamm. Schaffte Klarheit über zwei sich überlagernde Effekte (siehe nächster Punkt).
- **Skelett-Limit entdeckt — Blattbasis-Position falsch:** Bei breiten / gefiederten Blättern legte die PCA-basierte Endpunkt-Wahl die "leaf-base" mitten in die Blattlamina (für Tomato03/Blatt 17: 34.4 mm vom nächsten Stamm-Punkt entfernt, obwohl das Blatt nur 1.35 mm an den Stamm heranreicht). Ursache: PCA-Hauptachse läuft durch die breite Blattfläche; der dünne Petiolus hat zu wenig Volumen, um die Achse zu beeinflussen → der vermeintliche "proximale Endpunkt" landet im Blatt-Inneren.
- **Behoben durch Anchor-basierte Endpunktwahl:** `extract_polyline` bekam einen optionalen `anchor_to`-Parameter. Wenn übergeben (z. B. die Stamm-Punkte beim Blatt-Skelettieren):
    - Basis-Endpunkt = Punkt der Organ-Wolke mit kleinstem Abstand zur Anchor-Wolke
    - Spitze = geodätisch am weitesten entfernter Punkt von der Basis (statt PCA-Extremum)
    - Polyline ist per Konstruktion `base -> tip` orientiert; die alte Flip-Heuristik in `extract_plant_skeleton` ist entfallen.

  Verifiziert via Notebook 05 auf Tomato03/Blatt 17: leaf-base-Distanz zum Stamm sank von 34.4 mm auf 0.37 mm (Faktor ~93). Polyline-Knoten weiterhin alle im Blattvolumen (max 1.1 mm Abstand zum nächsten Blattpunkt).
### 2026-05-04 — Zweite Session

- **Real-Photo-Klassifikator-Pipeline** (`src/leaf/`, `notebooks/24_review_leaf_images.py`): iNaturalist-Downloader für Real-Apiaceae-Bilder als Quelle für Ansatz D. **Zwei Bug-Iterationen** während Implementierung: (1) `INAT_TAXA` Taxon-IDs anfangs erfunden — über 700 Bilder von Disteln/Schlüsselblumen runtergeladen; nach Bildkontrolle durch User aufgedeckt, korrigiert via `/v1/taxa?q=...` API; (2) `place_id=7035` für Deutschland erfunden, API antwortete HTTP 422; korrigiert via `/v1/places/autocomplete` zu 7207. Final-Filter: research_grade + place_id=7207 + min_identifications≥2. Ergebnis: 558 Bilder (83 Heracleum, 92 Conium, 129 Daucus, 58 Anthriscus, 88 Aethusa, 108 Pastinaca) — deutlich weniger als ohne Filter, dafür spürbar saubereres Signal (Heracleum-Stichprobe vom User: alle 83 visuell korrekt). Browser-basiertes Review-Tool mit Tastatur-Shortcuts (k/d/b/j) zum manuellen Cleanup; Trash ist reversibel. Kommt ins Memory: externe IDs nie raten.
- **Botanik-Korrekturen v7 + erste Real-Photo-Tests** (`src/synthetic/species.py`, `notebooks/22_v7_overnight.py`, `src/inference/predict.py`): User hat ein Anthriscus-Stockfoto durch das v6-Modell gejagt — Top-1 Pastinaca 83 % (komplett falsch). Diagnose: alle 6 Spezies wurden mit weißen Blüten gerendert, obwohl Pastinaca real **gelb** blüht. Dazu kompletter Crosscheck der numerischen Spec gegen Go Botany / Wikipedia: Anthriscus-Strahlen falsch bei 4–10 statt korrekt 5–15; Aethusa 10–20 statt 5–12; Daucus zu enge Strahlen+Pedicels; Pastinaca-Pedicels Range(20, 50) — die Wiki-Quelle hatte rays mit pedicels verwechselt; Realität 5–10 mm. v7 mit allen Korrekturen + `pedicel_rgb=(220,200,60)` für Pastinaca + `role_aware_color_callable(role_overrides=...)`. **Resultat: Test 97.5 %** (vs v6 98.2 %, leichte Regression durch breiteren Anthriscus-Range, der nun mit Heracleum überlappt). **Auf Anthriscus-Stockfoto:** Pastinaca-Bias gefixt (0 %), aber Top-1 wandert zu Conium 100 % — Anthriscus weiterhin nicht in Top-3. Bestätigt das vorhergesagte synth→real-Ceiling: das Modell hat keinen Weg, weißblühende Apiaceae ohne sichtbare Stamm-Speckles und ohne sichtbare Bracteoles von Conium zu trennen, weil die Anthriscus-Diagnostika (fein 2–3-fiederige Blätter) nicht modelliert sind. Konsequenz: Real-Photo-Klassifikator (Ansatz D) ist jetzt blockierend, nicht mehr optional.
- **Erster echter Real-Photo-Erfolg: v7 erkennt Anthriscus-Stamm 94.7 %** (`src/inference/predict_leaf.py`): User hat eine echte Wiesen-Kerbel-Pflanze fotografiert (Übersicht / Doldenkopf / Blatt / Stamm). Synth v7-Modell erreicht auf dem Stamm-Detail (grüner Stamm, keine Speckles, keine Bracts sichtbar — die diagnostische Anti-Conium-Signatur) **Top-1 Anthriscus mit 94.7 % Konfidenz** — der erste klare synth→real Treffer. Auf den anderen drei Fotos: Doldenkopf hat Anthriscus immerhin auf Rang 2 mit 37.5 %, Blatt-Foto Rang 2 mit 2.1 %, Übersicht nicht in Top-3 (Modell tippt nach Höhe auf Heracleum). Bestätigt: die Pipeline funktioniert grundsätzlich, aber braucht das *richtige* Bild — Bilder die nicht-modellierte Strukturen (Blätter) oder kontextfreie Eigenschaften (Höhe ohne Maßstab) zeigen, kollabieren in Default-Cluster. Gegenprobe: Leaf-Klassifikator (557 iNat-Bilder, 6 Klassen, ResNet-18, Test-Acc 48 %) sagt auf allen 4 Fotos Daucus mit ~90 % Konfidenz — krasser Daucus-Bias durch Klassen-Unbalance (Daucus 129 vs Anthriscus 58 Bilder) und Daten-Knappheit. Nächste Hebel: DACH-Daten erweitern (mehr Anthriscus) + Class-Weights im Training.
- **Leaf-Klassifikator v2 mit DACH-Daten + Class-Weights: erster echter Outdoor-Treffer** (`src/leaf/fetch_inaturalist.py` mit `--place AT,CH`, `notebooks/23_train_leaf_classifier.py` mit `CrossEntropyLoss(weight=...)`): nach v1-Diagnose (Daucus-Bias durch Klassen-Imbalance 129 vs 58 Bilder) zwei parallele Verbesserungen — (1) AT+CH-Daten dazugeladen (DE 557 → DACH 1257 → nach User-Curating 1217 mit ~3 % Trash-Quote), Anthriscus von 58 → 174 Bilder, (2) Inverse-Frequency Class-Weights (Aethusa-Loss × 1.85, Daucus-Loss × 0.59 etc.) im Training. **Resultat: Test-Accuracy 48.3 % → 76.0 %** (+28 pp), Anthriscus-Recall 14 % → 72 %, Aethusa 22 % → 60 %, alle Klassen mind. 60 %. Daucus-Bias verschwunden. **Auf den 8 echten Fotos:** Outdoor-Blatt-Foto trifft mit 80.1 % Anthriscus (erster reiner Real-Foto-Treffer durch das Leaf-Modell), Indoor-Stamm-Foto: Synth 94.7 % + Leaf 37.5 % Anthriscus (Doppel-Treffer durch beide Modelle — genau der Bridging-Effekt für den Ansatz D gedacht ist). Indoor-Blatt-Foto Anthriscus auf Top-2 mit 27 %, Indoor-Stamm Top-2 mit 38 %. Verbleibendes Problem: Outdoor-Übersicht/Doldenkopf werden nach wie vor von beiden Modellen falsch klassifiziert — wahrscheinlich Hintergrund-Distribution-Drift bei Synth (echte Hecke ≠ unser Render-BG) und ähnliches Pattern bei Leaf. Pipeline ist real-foto-aware aber noch nicht produktionsreif.
- **iNat-Hintergrund-Augmentation v8: Hypothese widerlegt** (`src/geometry/augment.py` mit `make_inat_background`, `notebooks/25_v8_overnight.py`): zur Adressierung des Outdoor-Pastinaca-Bias wurde die Synth-Augmentation um 50 % iNat-Backgrounds erweitert (heavy Gaussian-Blur radius=12, zufällig aus dem Apiaceae-Real-Foto-Pool gezogen). Erwartung: Modell lernt mit echten Foto-Hintergründen umzugehen → bessere Outdoor-Performance. **Ergebnis: keine Verbesserung, leichte Verschlechterung.** Synth-Test 97.5 % → 95.8 % (Heracleum-Klasse besonders betroffen, 96 % → 87 %). Auf den 8 Real-Fotos: kein neuer Anthriscus-Treffer; Indoor-Stamm-Konfidenz von 94.7 % → 62.6 % gefallen (beide Top-1, aber sicherheits-Erosion); Indoor-Doldenkopf: Anthriscus war Top-2 mit 37.5 % bei v7, jetzt 0 %. Outdoor-Verhalten unverändert (Pastinaca-Bias bleibt). **Wahrscheinlichste Erklärung:** der iNat-Pool enthält selbst Apiaceae-Pflanzen — wir haben effektiv "Apiaceae im BG" als Augmentation eingespeist, was das Modell vermutlich verwirrt statt es zu generalisieren. Zweitwahrscheinlichste: der gleichmäßige Gauss-Blur sieht anders aus als echte Out-of-Focus-Tiefenschärfe, das Modell lernt eine Augmentation-spezifische Heuristik. Drittens: Pastinaca-Bias hat tieferen Ursprung als BG-Distribution (unsere Pastinaca-Spec mit 10–30 Strahlen × 12–35 Pedicels könnte als pixeldichte Doldenkopf-Signatur zu charakteristisch sein). Konsequenz: BG-Augmentation ist nicht der richtige Hebel; nächste Iteration sollte vermutlich die Pastinaca-Doldenkopf-Geometrie hinterfragen oder Loss-basierte Strategien (z. B. Class-Weights auch im Synth-Training, Label-Smoothing).
- **Photogrammetrie-Versuch an realer Apiaceae gescheitert — Ursache präzisiert** (Scaniverse + ~50 Photo-Mode-Aufnahmen einer realen Wiesen-Kerbel-Pflanze in Vase): das resultierende Mesh enthielt **nur die Vase** als Volumen-Geometrie. Die Pflanze selbst (Stamm, Strahlen, Pedicels, Blätter) wurde nicht als 3D-Struktur rekonstruiert — stattdessen als 2D-Textur auf die Wandflächen des Hintergrunds projiziert. Klassisches "see-through depth ambiguity"-Versagen von Structure-from-Motion.

  **Wahre Ursache (per User-Insight präzisiert):** *kein* fundamentales Algorithmus-Limit, sondern ein **angulares Sensor-Auflösungsproblem**, das mit den Multi-Skala-Strukturen einer Apiaceae unvereinbar ist. Konkret: ein 0.5–1 mm Pedicel ist bei 80 cm Aufnahme-Distanz nur noch ~4 Pixel breit, am Rand der SfM-Track-Grenze. Bei 1.5 m Distanz nur ~2 Pixel — SfM versagt. Eine ganze Apiaceae-Pflanze in einer Spiral-Aufnahme zu erfassen erzwingt aber genau diese Distanz, weil Stamm und Doldenkopf zusammen ~1 m hoch sind. Analoger Fall: ein 5 cm Zaunpfosten auf 1 m Distanz hat ~160 Pixel und rekonstruiert trivial; *derselbe* Pfosten auf 100 m Distanz hat ~1.6 Pixel und erzeugt dieselben Artefakte wie unser Apiaceae. → Es ist ein **angulare-Skala-Mismatch**, nicht software- oder algorithmus-spezifisch.

  **Konsequenz für unsere Architektur — keine.** Die Hybrid-Architektur (Synth liefert 3D-Struktur, Leaf-Klassifikator liefert Real-Foto-Texturen über 2D) wurde in Antizipation genau dieses Limits gewählt.

  **Was theoretisch denkbar wäre — und warum es vermutlich auch nicht funktioniert:** Multi-Scale-Capture (Macro-Doldenkopf aus 15–25 cm + Standard-Stamm aus 50–80 cm, nachträglich registriert) hat zusätzlich zum Macro-DoF-Problem ein **Registrierungs-Problem**: zur Fusion zweier Scans unterschiedlicher Skalen werden **Überlappungsregionen** mit gemeinsamen Featurepoints benötigt — und genau die Pedicel-Region ist die, wo SfM systematisch versagt. Es gibt damit keine zuverlässigen Brücken-Features, um Macro- und Standard-Scan zu fusionieren. Selbst bei technischem Erfolg beider Einzelaufnahmen würde die Gesamt-Pflanze nicht zusammensetzbar sein. Strukturiertes Licht (Revopoint MINI 2 ~800 €) bleibt der einzige passive-3D-Pfad, der das umgeht — eigene projizierte Lichtmuster ersetzen die Featurepoint-Korrespondenz.

- **Macro-DoF-Konflikt — separater Limitierungs-Mechanismus** (zweiter Versuch mit Polycam Photo Mode, ~80 Aufnahmen, Detail Raw): Mesh hat 236k Vertices / 405k Triangles vs Scaniverse nur Vase — Polycam hat mehr rekonstruiert, aber Bbox 2.6 × 1.2 × 2.1 m → primär Raum-Geometrie, nicht Pflanze. User beim Fotografieren neue Erkenntnis: Macro-Photogrammetrie auf handheld Smartphone hat einen **fundamentalen DoF-Konflikt**, der orthogonal zum angularen Sensor-Limit wirkt. Konkret:
  - **AF springt vom Stamm** auf den texturreichen Hintergrund (genau den, den wir für SfM-Featurepoints brauchen): dünner grüner Stamm hat zu wenig Kontrast in sich selbst, der Phone-AF wählt deshalb den Hintergrund. Manuell fokussieren ist auf den meisten Phone-Apps fummelig.
  - **Wenn AF auf der Pflanze ist, ist Hintergrund unscharf:** flaches DoF bei Macro-Distanzen verliert SfM die Featurepoints im Hintergrund, die es für Kamera-Pose-Estimation braucht.
  - **Beide gleichzeitig scharf** wäre nur mit kleiner Blende möglich (f/16+), die ein Smartphone gar nicht hat. Mit DSLR + viel Licht prinzipiell lösbar; auf Handheld-Phone-Niveau aber nicht.

  Damit haben wir zwei **unabhängige** Photogrammetrie-Limits identifiziert: (1) angulare Auflösung (was wir gestern dokumentierten) und (2) Macro-DoF-Konflikt. Beide treffen Apiaceae-Pedicels. Die "Multi-Scale-Capture"-Idee ist davon ebenfalls betroffen — eine Macro-Aufnahme vom Doldenkopf hat *beide* Probleme, nicht nur das angulare. Strukturiertes Licht (Revopoint o.ä.) umgeht beide, weil es eigene Lichtmuster projiziert und nicht auf passive Hintergrund-Features angewiesen ist.

  **Visueller Befund Polycam-Mesh** (`data/real_scans/polycam_anthriscus_20260504/`, `notebooks/output/polycam_anthriscus.html`): Vase als sauberes Volumen rekonstruiert, Stamm als dünne aber zusammenhängende Linie, **Doldenkopf nur als diskonnektierte "schwebende Inseln"** über dem Stamm. Charakteristisches SfM-Mesh-Fusion-Versagen: die Blüten-Cluster haben genug lokalen Kontrast für Tracking, aber ohne die verbindenden Pedicels (sub-pixel) bleibt die Mesh-Topologie zerrissen. Pedicels selbst gar nicht erkennbar. Vergleich der zwei unabhängigen Tools (Scaniverse video → nur Vase, Plant auf Wand projiziert; Polycam photo Raw → Vase + Stamm + Insel-Fragmente) bestätigt: Pattern ist software-unabhängig und konsistent mit beiden dokumentierten Limit-Mechanismen.

### 2026-05-05 — Dritte Session

- **Hybrid-Inferenz Synth + Leaf — Existenzbeweis, NICHT validiert** (`src/inference/hybrid.py`, `notebooks/26_hybrid_eval.py`): Wrapper-Modul lädt beide Modelle, kombiniert Wahrscheinlichkeiten via fünf Ensemble-Strategien (Soft, Max, Vote, Confidence-Selection, Leaf-Priority). Naives Soft 50/50 schadet (4/9 Top-3 vs 6/9 Synth-allein) weil Synths Pastinaca-Überkonfidenz die Leaf-Predictions kontaminiert. Nach Tuning der Gewichte fand sich **Soft 30/70 (Synth 30%, Leaf 70%) mit 2/9 Top-1** — Indoor-Stamm-Treffer von Synth UND Outdoor-Blatt-Treffer von Leaf gleichzeitig erhalten. Andere Strategien blieben bei 1/9 Top-1 wie Einzelmodelle.

  **Methodische Einschränkung (User-Pushback):** dieser Ergebnis ist **kein validierter Architektur-Beweis**. Wir haben sechs Strategien (5 + variable Gewichtung) auf demselben 9-Foto-Testset evaluiert und post-hoc das beste berichtet — klassisches Hyperparameter-Tuning auf dem Test-Set mit multiple comparisons. Bei n=9 und ~16.7 % Random-Chance ist 2/9 vs 1/9 nicht signifikant. Was wir tatsächlich gezeigt haben: (a) es **existiert** eine Gewichtskombination die beide Einzelmodell-Treffer auf diesem Set erhält (Existenzbeweis auf einem spezifischen Datenpunkt), (b) naive Strategien (50/50, Vote, Max) ändern oder verschlechtern konsistent über drei unabhängig formulierte Varianten. Was wir **nicht** gezeigt haben: dass 30/70 generalisiert, dass der Hybrid systematisch besser ist, dass die Verbesserung über Zufall hinausgeht. Echte Validierung bräuchte pre-registriertes ~50-Foto-Hold-out-Set, das **nach** Strategie-Auswahl gesammelt wird.
- **Render-Augmentation + v6-Training** (`src/geometry/augment.py`, `notebooks/20_v6_overnight.py`): drei Realismus-Komponenten als Post-Processing nach dem Render: (1) prozeduraler Sky-to-Ground-Background mit per-Szene-Horizont-Variation und Farbrauschen ersetzt den konstanten grauen Hintergrund; (2) Lambert-Shading aus depth-Gradient-abgeleiteten Pseudo-Normalen mit per-Szene-Lichtrichtung; (3) per-Szene-RGB-Skalierung für Weißabgleich-Variation. **Bug-Fix während Implementierung:** initiale Background-Funktion hatte 90°-rotierten Gradient durch numpy-Broadcast-Edge-Case (`(H,1)` bool gegen `(H,1,3)` mit `np.where`). Nach Fix korrekt vertikal. Dataset v6 mit `augment=True` gebaut (38 min), Training mit identischen v5-Hyperparametern (45 min). **Resultat: Test 98.2 % (v5 war 97.8 %)**, Train-Loss 0.079 → 0.018, alle Klassen ≥95.8 %, Conium/Aethusa/Pastinaca alle 100 %, Anthriscus +2.5 pp. Wichtiger als die geringe Acc-Verbesserung: das Modell hat gelernt, mit variablem Hintergrund / Beleuchtung umzugehen. Echte Real-Transfer-Bewertung steht noch aus (wartet auf erste Polycam-Aufnahme einer Apiaceae).
- **Conium-Stamm-Speckles in L-System + v5-Training** (`src/synthetic/species.py`, `src/training/dataset.py`, `src/geometry/render.py`, `notebooks/19_train_v5_long.py`): Conium maculatum hat im realen Habitus charakteristische lila-braune Stammflecken (namensgebend "Gefleckter Schierling"). Im Generator als `stem_speckle_density: Range(0.10, 0.20)` modelliert: ~16 % der Stamm-Punkte werden im Dataset-Builder zur Rolle `stem-speckle` umgelabelt, neuer Eintrag `(105, 50, 90)` in `ROLE_TO_RGB` — purpur-braun gegen die übrigen dunkelgrünen Stamm-Punkte sehr distinktiv. Dataset v5 mit 200 Instanzen × 4 Views, ansonsten v3-identische Parameter. Training mit identischen Hyperparametern wie v3-long (20 Epochen Cosine LR, Weight Decay 1e-4) für apples-to-apples Vergleich. **Resultat: Test 97.8 % (v3-long: 88.3 %), Conium-Recall 52 % → 100 %**, alle anderen Klassen ≥93 %. Drei Erkenntnisse: (a) **Asymmetrische Confusion ist diagnostisch** — wenn Conium→Anthriscus 54× passiert aber umgekehrt nur 5×, fehlt dem Modell ein Conium-spezifisches Pixel-Signal, das Paar ist nicht "zu nah". (b) **Auflösung war nicht der Engpass** — der frühere v4-768-Versuch brachte fast nichts, weil das Problem im *Inhalt* der Render lag, nicht in Pixel-Anzahl. (c) **Pro morphologischem Diagnostikum, das an die Render-Pipeline angeschlossen wird, scheinen wir 5–10 pp Accuracy zu gewinnen** (Role-Color +32 pp, mehr Daten +13 pp, Speckles +9.5 pp). Synth-only-Pipeline jetzt an einem natürlichen Plateau; nächster Schritt ist Brücke zu realer Polycam-Aufnahme.
- **Overnight-Run v3-long + v4-768** (`notebooks/18_overnight.py`): Schritt 1 längeres Training auf v3 (20 Epochen, Cosine-LR von 1e-3 nach 1e-5, Weight Decay 1e-4) — Best Val 93.8 %, **Test 88.3 %**, aber **Conium-Recall kollabierte auf 52.5 %** (war 73 % bei 8 Epochen). Schritt 2 v4-Dataset bei 768×768 + 15 Epochen Training — **Test 89.2 % (+0.9 pp)**, Conium 54 %, Aethusa 100 %, Heracleum 100 %, Pastinaca 97 %. Drei zentrale Befunde: (a) **Auflösung war nicht der Bottleneck** — von 384 auf 768 brachte fast nichts; Hypothese widerlegt. (b) **Asymmetrische Confusion bei v4: Conium→Anthriscus 54, Anthriscus→Conium 5** — Modell hat systematische Bias gegen Conium, nicht symmetrisches Paar-Problem. (c) **Längeres Training verschlimmerte Conium** (73 → 52 %): Modell schiebt unsichere Conium-Beispiele zunehmend ins Anthriscus-Cluster. Diagnostische Merkmale funktionieren wo kategorisch (Aethusa reflexed Bracteoles 100 %, Pastinaca no-bracts 97 %, Daucus prominent Involucre 96 %), versagen wo graduell (Coniums 3–6 winzige Bracts vs. Anthriscus' gar keine — auf Pixelebene zu subtil). Plateau bei ~89 % wahrscheinlich ohne weiteres Spezies-Detail (Stamm-Marker, Blattform) nicht durchbrechbar.
- **Trainings-Set v3 + 4× mehr Daten** (`data/training/v3/`): 200 Instanzen × 6 Arten × 4 Views = 4800 Examples, 310 MB. Klassifikator-Training auf v3, gleiche Architektur und 8 Epochen wie v2. **Test-Accuracy 74.4 % (v2) → 87.8 % (v3), +13 pp**. Per-Klasse-Recall (Δ zu v2): Heracleum 95 % (+2), **Conium 73 % (+30)**, Daucus 93 %, **Anthriscus 72 % (+22)**, **Aethusa 95 % (+20)**, Pastinaca 99 %. Validation-Schwankungen abgemildert, Best-Checkpoint robust. **Verbleibender Hard Case: Conium ↔ Anthriscus** — 76 % aller Test-Fehler entfallen auf dieses Paar (32 + 34 von 87 Fehlern). Beide morphometrisch eng (überlappende Höhen, Conium 10–20 Strahlen vs. Anthriscus 4–10, Conium hat kleines Involucre 3–6 Bracts). Vermutlich Auflösungs-Limit: Coniums 3–6 Bracts schrumpfen bei 384×384 mit 1.5×bbox-Framing auf wenige Pixel und werden bei manchen Kamerawinkeln verdeckt. → nächster Hebel: Render-Auflösung 768 oder gezielte Bract-Auflösungs-Lupe.
- **Erstes 2D→Spezies-Modell trainiert (v1 → v2)** (`src/models/classifier.py`, `notebooks/17_train_classifier.py`): kompaktes CNN (~843k Params, 4× Stride-2-Conv-Block + GlobalAvgPool + Linear 6 Klassen), Splits **strikt nach Instanz** (35/8/8 Pflanzen pro Spezies × 4 Views = 840/192/168 Examples), 8 Epochen Adam, lr=1e-3, BS=16, ~5 min CPU-Training. **v1-Resultat 42.3 % Test-Accuracy** — deutlich über Zufall (16.7 %), aber Conium und Anthriscus *nie* vorhergesagt. Diagnose: `default_color_for_label` in `render.py` färbte nur die ersten 30 Organ-IDs (Stamm + erste paar Strahlen) distinkt; alles ab Label-ID 31 (Pedicels, Bracts, Bracteoles, alle weiteren Strahlen) landete uniformly off-white → diagnostische Strukturen wie Aethusas reflexed Bracteoles waren im Render farblich identisch zu Pedicels. **Fix:** `role_aware_color_callable(skel)` in `render.py` mappt Organ-ID → Rolle → kanonische Farbe (Bract braun, Bracteole ocker, Pedicel weiß, Strahl hellgrün, Stamm dunkelgrün). Trainings-Set v2 mit dieser Färbung neu gerendert. **v2-Resultat 74.4 % Test-Accuracy (+32 pp).** Conium von 0 → 43 %, Anthriscus von 0 → 50 % — die zuvor unsichtbaren Klassen werden jetzt erkannt. Verbleibende Confusion: Mid-Size-Triplet Conium/Anthriscus/Aethusa (ähnliche Topologie, unterscheidbar nur an Bract-Details), Aethusa↔Daucus (beide haben prominente Bracteoles). Validation schwankt stark — Hinweis auf Overfit + instabile BatchNorm bei nur 35 Train-Instanzen/Klasse. Wichtigste Erkenntnis: **die diagnostische Information ist im Render vorhanden**, aber Render-Pipeline-Farb-Pipeline ist genauso kritisch wie Modell-Architektur.
- **Erstes synthetisches Trainings-Set v1 gebaut** (`src/training/dataset.py`, `notebooks/15_build_training_set.py`): 50 Instanzen × 6 Arten × 4 Kamera-Winkel (0°/90°/180°/270° azimuth, 10° elevation) = **1200 Bild-Skelett-Paare** in `data/training/v1/`. Schema `umbella.training.v1`: pro Instanz ein Skeleton-JSON in `skeletons/<species>/seed{NNNN}.json`, pro View ein npz mit `rgb / label / depth / camera / target / azimuth_deg / elevation_deg` in `images/<species>/seed{NNNN}_view{V}.npz`, plus globaler `metadata.json`-Index. Bauzeit 656 s (~0.55 s/Example). **Disk-Footprint nur 77.5 MB** auf 1501 Dateien — deutlich unter Schätzung dank npz-Kompression sparser Bilder. `list_examples(root)` und `load_example(root, ex)`-Helfer für Trainings-Loops. Damit ist der erste ML-taugliche Datensatz verfügbar.
- **2D-Render-Pipeline eingebaut — Brücke geschlossen** (`src/geometry/render.py`, `notebooks/14_render_synthetic.py`): perspektivische Projektion einer gelabelten Punktwolke auf ein Bild. Vektorisiertes Punkt-Splatting mit kreisförmigem Kernel, sortier-basierter Z-Buffer (nearest-z gewinnt pro Pixel via numpy-Scatter mit far-first ordering). Output: RGB (HWC uint8) + label_map (HW int32) + depth (HW float32) — alle drei aus *einem* Aufruf, sofort als Trainings-Triplet nutzbar. Vorgegebene Plant-Farb-Palette (Stamm dunkelgrün, Strahlen mittelgrün, Pedicels/Bracts off-white). 384×384-Renders mit ~3.4–5.6k Foreground-Pixeln pro Pflanze (~2–4 % Bildfläche, realistisch). Ende-zu-Ende für eine Pflanze < 1 s. Damit ist die volle synth → 2D-Brücke verfügbar: `Spec → Params → Skeleton → Cloud → HPR → Render → (RGB, Labels, Depth)`. Bewusste Limitationen: konstanter grauer Hintergrund, keine Beleuchtung/Schatten, festes Farbschema — alle bekannt, fix lösbar.
- **Hidden Point Removal eingebaut** (`src/geometry/visibility.py`, `notebooks/13_view_sampling.py`): Katz/Tal/Basri 2007 Spherical-Flip-Algorithmus; `hpr_visible_indices(points, camera, radius_factor)` plus `hpr_multi_view(...)` für Mehrkamera-Union und Helper `camera_around(...)` zur Kamera-Platzierung. Default `radius_factor=1000` empirisch kalibriert (Sweep auf Heracleum: factor=10→0.5 % retention, factor=100→2.0 %, factor=1000→7.5 %, factor=10000→>20 %; 1000 entspricht typischer Single-View-Photogrammetrie). Validierung gegen den Domain-Gap-Befund: synthetische Heracleum-Stamm-Azimuthal-Nonuniformität fällt von 0.01 (full uniform) auf **0.75 (HPR single view)** — nahe Pheno4D-Referenz 0.99. 4-Kamera-Setup geht zurück auf 0.07 (LiDAR-artige Voll-Coverage). Wichtig: das HPR-Verfahren ist robust gegen `radius_factor`-Tuning bezüglich der Einseitigkeit (Stamm-Nonuniformity bleibt bei 0.75 unabhängig vom Faktor); der Faktor steuert nur die Punktdichte. Damit ist Domain-Gap #2 (azimuthal coverage) abgehakt.
- **Domain-Gap real ↔ synthetisch quantifiziert** (`notebooks/12_domain_gap.py`): einer Pheno4D-Tomato-Wolke + einer synthetischen Heracleum-Wolke gegenübergestellt, jeweils Stamm und ein Organ isoliert. Drei Hauptgaps gemessen:
    1. **Auflösung:** real mean NN-Distanz ~0.07 mm, synth ~0.82 mm — **~10× zu grob**. Trivial behebbar via `points_per_mm² = 5` (statt 0.5).
    2. **Azimuthal Coverage:** azimuthale Nonuniformität (Std normierter Histogrammbins) real Stamm 0.99, Blatt 1.20; synth Stamm 0.03, Organ 0.05 — **größtes semantisches Gap.** Reales LiDAR sieht einen einseitigen Bogen, unser uniformes Zylinderoberflächen-Sampling deckt 360° ab. Lösung: View-Sampling / Hidden-Point-Removal mit virtueller Kamera-Position bevor weiter trainiert wird.
    3. **Per-Organ-Verteilung:** real Tomato 6.8 % Boden / 8.8 % Stamm / 84.4 % Blatt (blattflächendominiert); synth Heracleum 0 % Boden / 45.6 % Stamm / 54.4 % Organe (stammdominiert, da Strahlen dünn). Vergleich nicht ganz fair (verschiedene Familien), aber zeigt: reale Pflanzen haben *deutlich* mehr Oberfläche-bringende Blätter und dazu Boden, der bei uns komplett fehlt.

  **Trainingsstrategie-Konsequenz:** Vor jedem synth → real Transfer-Versuch ist View-Sampling die teuerste, aber wichtigste Anpassung — Auflösung ist trivial, Per-Organ erst beim Apiaceae-Realdaten-Validieren akut.
- **Punktwolken-Sampling auf Skelett-Geometrie** (`src/geometry/pointcloud.py`, `notebooks/11_synthetic_pointclouds.py`): jede Skelett-Kante wird als Zylinder mit rollenabhängigem Radius behandelt; uniformes Sampling auf der Mantelfläche mit optionalem Gauss-Rauschen, lineares Tapering zwischen unterschiedlichen Endpunkt-Radien. Default-Radien decken Apiaceae- und Pheno4D-Rollen ab (Stamm 7 mm, Pedicel 0.5 mm, Bracteole 0.7 mm etc.). Output: `(xyz, organ_label)` im selben Format wie Pheno4D-Loader. Sechs synthetische Wolken erzeugt, eine pro Art, gespeichert unter `data/clouds/synthetic/<species>/seed000.npz`. Punkt-Counts plausibel skaliert (Aethusa 29k → Pastinaca 181k); Sampling-Zeit < 0.1 s/Pflanze. Bewusste Limitationen für später: keine View-Angle-Anisotropie, keine Self-Occlusion, kein Boden, kein Wind. Damit ist die volle synthetische Pipeline geschlossen: SpeciesSpec → ApiaceaeParams → Skeleton → Point cloud, format-kompatibel zu Pheno4D.
- **Bract-aware Features + 6-Arten-Klassifikator** (`src/eval/features.py` erweitert um 7 Bract-Features, `notebooks/10_classify_apiaceae.py`): synthetisches Korpus 100 Instanzen × 6 Arten = 600 Skelette in `data/skeletons/synthetic/<species>/seed*.json` (~48 s Generierung). Stratified-5-fold-CV mit LogReg und RandomForest auf 27 Features. **LogReg 100.0 %, RandomForest 99.8 %; Aethusa ↔ Conium fehlerfrei in beide Richtungen.** Top-Features (RF-Gini): `mean_bract_length` 0.116, `mean_bracteole_reflex_angle` 0.112, `n_bracts` 0.100 — Bract-Features dominieren die Diskrimination. **Ablation (Bract-Features entfernt):** RF fällt auf 97.2 %; überraschender Befund: nicht Aethusa, sondern *Conium* verliert Genauigkeit (100 % → 94 %), während Aethusa bei 100 % bleibt. Erklärung: Aethusa hat genug andere Diskriminatoren (Höhe, Pedicel-Zahl), Conium dagegen ist morphometrisch unauffällig und wird allein durch Anwesenheit eines kleinen Involucre identifizierbar. Wichtigster Caveat: synthetisch-auf-synthetisch ist obere Schranke; reale Performance noch unbekannt. Naming-Hinweis: `leaf_length_*`-Features greifen in Apiaceae-Skeletten auf Rays/Pedicels (alle organ_id ≥ 2) zu — funktioniert, aber semantisch unsauber.
- **Hüllblätter und Hüllchen ins L-System eingebaut** (`src/synthetic/apiaceae.py`, `src/synthetic/species.py`): neue `ApiaceaeParams`-Felder für Involucre (am Doldenboden) und Involucel (am Umbellet-Boden) — jeweils Bract-Anzahl, -Länge und Winkel relativ zur Eltern-Achse. Winkel > 90° = **reflexed** (zeigt entgegen der Eltern-Achse). Helper `add_bracts(...)` fügt einen Ring von Bracts azimutal um eine Achse hinzu, neue Rollen `"bract"` und `"bracteole"`. Sechs Spezies entsprechend der Wikipedia-Beschreibungen kalibriert: nur Conium und Daucus haben Involucre, Pastinaca gar nichts, **Aethusa cynapium hat 3–5 lange (8–18 mm) reflexed Bracteoles bei Winkel 125–155°** — das diagnostische Merkmal, das Hundspetersilie von Schierling unterscheidbar macht. Counts auf seed=0 verifiziert: Heracleum 0/207, Conium 18/155, Daucus 12/232, Anthriscus 0/96, Aethusa 0/100, Pastinaca 0/0 — passt zu botanischer Realität. **Konsequenz:** Skelett-Topologie kodiert jetzt das erste echte Diagnose-Merkmal jenseits reiner Strahlenanzahl. Feature-Extraktor sieht bracts/bracteoles allerdings noch nicht explizit — nächster Schritt für den Klassifikator.
- **Apiaceae-Generator gegen Bestimmungsschlüssel kalibriert** (`src/synthetic/species.py`, `notebooks/09_calibrated_apiaceae.py`): sechs Arten (Wiesen-Bärenklau, Schierling, Wilde Möhre, Wiesen-Kerbel, Hundspetersilie, Pastinak) mit empirischen Wertebereichen für Höhe, Strahlenanzahl, Strahlen-/Pedicel-Längen und Winkel-Konussen. `SpeciesSpec.sample(seed)` zieht eine `ApiaceaeParams`-Instanz aus den Bereichen. Tabelle + Quellen siehe Abschnitt "Apiaceae-Artenkatalog". Sanity-Check: Höhen-Verteilungen aller sechs Arten liegen innerhalb der Literaturbereiche, Variabilität innerhalb einer Art deutlich kleiner als zwischen Arten. Identifizierte Limitationen für den nächsten Iterationsschritt: schwer trennbare Paare (Conium/Aethusa, Heracleum/Pastinaca) brauchen Hüllblatt-Geometrie bzw. Blütenfarbe — Skelett-Topologie allein reicht nicht.
- **L-System-artiger Apiaceae-Generator** (`src/synthetic/apiaceae.py`, `notebooks/08_synthetic_apiaceae.py`): erste prozedurale Pipeline für Doppeldolden. Topologie Hauptstamm → optionale Lateraltriebe → terminale Dolde → primäre Strahlen (radial mit Phyllotaxis 137.5°) → Umbellet am Strahlenende → Pedicels (radial um die Strahlenachse). Output ist ein `Skeleton` im selben Schema wie Pheno4D-Extraktion — dieselbe Pipeline kann downstream beide Quellen verarbeiten. `ApiaceaeParams`-Dataclass mit ~15 Parametern (Anzahl Strahlen, Strahlenlängen, Kegelhalbwinkel, Pedicel-Geometrie, Randomness etc.). Vier erste Presets `Heracleum_like / Conium_like / Daucus_like / Anthriscus_like` mit je eigenem Seed; Knoten/Edge-Zahlen plausibel (135 / 57 / 42 / 28 primäre Strahlen inkl. Lateralumbeln). Längen, Winkel und Streuungen sind Schätzwerte — Kalibrierung gegen Bestimmungsbücher / Felddaten steht noch aus. Blätter, Hüllblätter und Stängelkrümmung bewusst noch nicht modelliert (diagnostischer Kern liegt in der Doldenstruktur).
- **End-zu-End-Sanity-Check Mais vs. Tomate** (`notebooks/07_classify_maize_vs_tomato.py`, `src/eval/features.py`): 20 strukturelle Skelett-Features (Knotenzahl, Verzweigungen, Längen, Bbox, Aspect Ratio etc.), Leave-One-Plant-Out-CV mit LogReg und RandomForest. **LogReg 100 % scan-level, 14/14 Pflanzen**; RandomForest 99.2 % (1 Tomato-Frühstadium-Fehler an Tag 5. März, korrekt nach Plant-Level-Majority). Top-Features (RF-Gini): `n_branch_nodes`, `aspect_ratio_hw`, `branching_factor`, `leaf_length_mean` — botanisch plausibel. Wachstumsstadien-Auswertung bestätigt Hypothese aus Test-Set-Methodik: vegetatives Frühstadium ist der harte Fall (85.7 % am ersten Termin, 100 % ab Tag 0307). Bestätigt End-zu-End-Pipeline (Pheno4D → Skelett → JSON-Korpus → Features → Klassifikator → Plant-Level).
- **Skelett-Korpus serialisiert** (`notebooks/06_build_skeleton_corpus.py`): alle 126 annotierten Pheno4D-Scans → JSON-Skelette in `data/skeletons/<plant_id>/<date>.json`. JSON-Schema `umbella.skeleton.v1` mit `nodes / edges / node_organ / node_role / metadata`. Lese-/Schreibroutinen `Skeleton.save_json` / `load_json` / `to_dict` / `from_dict`. Kennzahlen: 255 Mio. Punkte → 22 417 Skelett-Knoten (~11 400× Reduktion), 1 123 erkannte Blätter, 378 automatisch eingefügte Junctions, 1.88 MB Gesamt-Disk auf 126 Dateien, 169 s Laufzeit (~1.3 s/Skelett). Maize-Skelette wachsen linear (~100 → 140 Knoten über 12 Tage); Tomato verzweigt stärker (~120 → 442 Knoten). Tomato02_0325 ist Ausreißer (19.9 s, sonst 3–5 s) — vermutlich pathologische Stamm-Geometrie, später anschauen.
- **Skelett-Topologie verfeinert — Edge-Splitting bei Blatt-Anbindung:** Vorher wurde jede Blattbasis an den *nächsten Stamm-Skelett-Knoten* angehängt; bei Voxel-Spacing von ~19 mm konnte der gewählte Knoten bis zu ~10 mm entlang der Stamm-Achse versetzt sein, was visuelle "Schwebe-Effekte" der Join-Linien verursachte. Neuer Helper `_attach_point_to_tree` projiziert die Blattbasis auf den nächsten Punkt einer Stamm-**Kante**, fügt dort einen `stem-junction`-Knoten ein und splittet die Kante. Datenstruktur bleibt Baum (nur n-1 Kanten). Audit auf Tomato03_0325: 14 Junctions automatisch eingefügt, Join-Linien sind jetzt rein radial (Restdistanz = Stamm-Radius) statt diagonal verschoben.
- **v9: Online-Augmentation + curated BG-Pool — Hypothese auf 9-Foto-Test nicht bestätigt** (`src/geometry/augment.py`, `src/models/classifier.py`, `src/training/dataset.py`, `notebooks/27_v9_overnight.py`, `notebooks/28_v9_real_eval.py`). Architekturwechsel weg von v6/v8s offline-Augmentation: `build_dataset(augment=False)` schreibt nur clean RGB + label + depth, `ApiaceaeImageDataset(online_aug_pool=...)` ruft `augment_render(seed=None, ...)` per `__getitem__` mit frischem Seed → 12 Epochen × 4800 Examples = 57 600 unique Views statt 4800. Hintergrund-Pool: 88 subjekt-freie Bilder (64 Pexels-Outdoor-Texturen via API + 13 AI-generierte Indoor-Szenen Leonardo/Ideogram + 11 AI-generierte Outdoor-Szenen Ideogram, alle manuell auf Pflanzen-/Personen-Freiheit kuratiert). v8s "iNat-mit-Subjekten + heavy blur"-Ansatz wurde komplett verworfen weil iNat-Backgrounds Apiaceae enthielten. **Synth-Test-Ergebnis: 89.6 % Test-Acc** (v6 war 98.2 %, v7 ~99 %) — online aug macht das Lernproblem deutlich härter, Lernen springt erst Epoche 8 los (vorher 17 % = chance). Pro Klasse: Pastinaca 99.2 %, Conium 97.5 %, Aethusa 95.8 %, Daucus 89.2 %, Heracleum 79.2 %, Anthriscus 76.7 %. **Real-Foto-Ergebnis (n=9, dieselben Wiesen-Kerbel-Fotos wie v7/v8): v9 0/9 Top-1, 5/9 Top-3 für Anthriscus** — schlechter als v7 (1/9, 6/9). Hybrid v9+Leaf 30/70: 1/9 Top-1, 3/9 Top-3 — schlechter als der vorher gefundene v7+Leaf 30/70 mit 2/9 Top-1. v9 prädiziert auffallend oft "Hundspetersilie" (Aethusa) auf Real-Fotos, was zur internen Aethusa-Confidence von 95.8 % im Synth-Test passt — Modell hat sich auf Aethusa-spezifische Render-Features verfestigt. **Honest Reading:** zentrale Refactor-Hypothese ("online aug + curated BG verbessert Real-Foto-Generalisierung") ist nicht bestätigt. Bei n=9 ist das aber statistisch nichtssagend, der eigentliche Befund ist, dass jede Real-Foto-Aussage in diesem Projekt bisher auf einem 9-Foto-Set beruht und damit Rauschen-dominiert sein dürfte. Drei mögliche Lesarten: (a) Synth→Real-Gap ist primär Geometrie/Textur, nicht BG/Lighting — Augmentation kann das nicht überbrücken. (b) 88 BGs vs. v8s Tausende iNat-BGs — Pool zu klein. (c) n=9 ist Rauschen, beide Modelle gleich (un)tauglich. Strukturelle Konsequenz: bevor weitere Augmentation/Architektur-Tweaks lohnen, braucht es ein größeres Real-Foto-Test-Set (Größenordnung 50+ Fotos, kuratiert pre-experiment), sonst tunen wir auf Rauschen.
- **Real-Foto-Quellen sondiert + neuer Befund Fruchtmerkmale.** Phänologie Anfang Mai erlaubt nur Wiesen-Kerbel im Freiland zuverlässig — die anderen fünf Arten stehen meist nur als Blattrosetten. Eigene Bestimmung von Rosetten ist auf User-Niveau nicht zuverlässig, daher Strategie "im Mai markieren, im Juni zurück" verworfen. Anfrage an den Botanischen Garten Heidelberg ergab: keine der sechs Arten regelrecht im Bestand, blühen dort nicht früher als anderswo. **Ergänzender Hinweis aus der Antwort:** für die klassische Apiaceae-Bestimmung sind zusätzlich **Fruchtmerkmale (Doppelachänen / Mericarpe)** erforderlich, die erst Juli/August/September verfügbar sind. Das ist ein Punkt, der im aktuellen Forschungsdesign fehlt — sowohl der Synth-Generator als auch beide CNNs (Synth + Leaf) konditionieren ausschließlich auf Blütenstands-/Blatt-Merkmale, *nie* auf Fruchtform. Konsequenz für später: (a) Real-Foto-Set sollte explizit Frucht-Stadien einschließen, nicht nur Blütezeit; (b) langfristig könnte ein Frucht-Detail-Klassifikator als dritter Hybrid-Zweig sinnvoll sein (analog zum Leaf-CNN), gerade für die schwer trennbaren Paare. Empfohlene Folge-Anlaufstellen: PH-Heidelberg Ökogarten (didaktisch kultiviert, vermutlich beschriftet) und NABU-Naturgarten (Wildgarten, eher real-foto-realistisch). Mail an den Ökogarten in Vorbereitung.

### 2026-05-06 — Vierte Session

- **Frucht-Befund von gestern reframiert — von Design-Gap zu möglichem Forschungsbeitrag.** Folgehinweis vom Botanischen Garten: bei der klassischen Apiaceae-Bestimmung sind Blüten- und Fruchtmerkmale *nicht* deckungsgleich verfügbar — eine vorliegende Pflanze ist mit einem klassischen Bestimmungsgang oft gar nicht regelrecht bestimmbar, gerade weil das nötige Merkmals-Kombi nicht gleichzeitig am Exemplar zu finden ist. Daraus folgt: ein ML-Modell, das mit *nur einem* Merkmalskomplex (nur Blüten, oder nur Früchte, oder nur Blätter bei bekannter Gattung) zur Artbestimmung kommt, wäre aus botanischer Sicht ein eigenständiger Beitrag, nicht eine "verkürzte" Version des klassischen Verfahrens. Das verschiebt den Anspruchshorizont von "wir reproduzieren ein Bestimmungsbuch" hin zu "wir lösen einen Fall, in dem das Bestimmungsbuch praktisch versagt". Konkrete Konsequenzen: (a) Real-Foto-Set wird **nicht** auf Frucht-Stadien geweitet werden müssen, um valide zu sein — Blüh-Stadium-Fotos sind ein legitimer eigener Datenraum. (b) Der gestrige Eintrag "Frucht-Stadium fehlt im Forschungsdesign" ist überzeichnet — sauber wäre: ein Frucht-Branch ist eine **Erweiterungsoption** für später, kein "Gap". (c) Bei Schreiben einer späteren Veröffentlichung sollte dieses Framing explizit dokumentiert werden, weil sonst Reviewer (zu Recht) die fehlenden Fruchtmerkmale monieren würden — sie sind nicht *vergessen*, sondern *bewusst draußen*.


## Ausgangspunkt: Wie funktioniert Pflanzenerkennung heute?

Praktisch alle gängigen Apps (PlantNet, Flora Incognita, iNaturalist/Seek, PictureThis) basieren auf 2D-Bilderkennung mit trainierten neuronalen Netzen — meist CNNs, zunehmend auch Vision Transformer. Trainiert wird auf Millionen gelabelter Fotos; das Modell lernt diskriminative visuelle Merkmale wie Blattform, Aderung, Blütenstruktur, Behaarung und Wuchsform.

"Mehr Struktur" steckt höchstens darin, dass viele Apps mehrere Aufnahmewinkel bzw. Organe getrennt anfragen (Blatt, Blüte, Frucht, Rinde) und Einzelvorhersagen kombinieren. Das ist Multi-View-2D, nicht 3D.

## Echte 3D-Pflanzendaten: Wo sie existieren

3D-Daten von Pflanzen werden vor allem in Forschungskontexten erzeugt, nicht für Erkennung:

- **Phänotypisierung** in Pflanzenbiologie und Agrarwissenschaft: LiDAR, Structure-from-Motion, Laserscanner. Ziel: Wachstumsanalyse, Biomasseschätzung, Sortenvergleiche.
- **Funktional-strukturelle Pflanzenmodelle (FSPM)**: GroIMP, OpenAlea. Pflanzen werden prozedural als 3D-Geometrie modelliert. Eher Simulation als Erkennung.
- **Computer-Vision-Datensätze**: Pheno4D, Plant3D, ROSE-X, Soybean-MVS, diverse Weizen-/Gerste-Sets, TreeNet3D für Forst.

## Lücken im Feld (= mögliche Ansatzpunkte)

1. **Artenvielfalt:** Bestehende 3D-Datensätze decken eine Handvoll Nutzpflanzen ab. Wildpflanzen, Zierpflanzen, Stauden, Kräuter — kaum vorhanden. Eine Flora-Incognita-Äquivalenz in 3D existiert nicht.
2. **Crowdsourcing in 3D:** Es gibt kein "iNaturalist für Punktwolken". Mit modernen Smartphones (LiDAR in iPhone Pro, Gaussian Splatting aus Videos via Polycam, Luma) heute technisch möglich, vor 5 Jahren noch nicht.
3. **NeRF / Gaussian Splatting für Pflanzen:** Junges Feld. Pflanzen sind wegen Selbstverdeckung, dünnen Strukturen und Wind ein hartes Problem für diese Methoden — publikationsreifer Forschungsbereich.
4. **2D ↔ 3D Verknüpfung:** Modelle, die aus einem Handyfoto plausible 3D-Struktur einer bestimmten Art rekonstruieren ("single-view plant reconstruction"), sind erst am Anfang.
5. **Anwendung in der Erkennung:** Kaum jemand nutzt 3D zur Bestimmung in freier Wildbahn. Offene Frage: Bringt 3D-Information messbar etwas bei schwer unterscheidbaren Arten (Doldenblütler, Gräser), wo 2D-CNNs notorisch schwächeln?

→ Die interessantesten Nischen sind aus heutiger Sicht **die 2D→3D-Brücke** und ggf. **community-getriebener 3D-Aufbau für eine Klade, wo 2D an Grenzen stößt**. Beides ist im akademischen Raum noch nicht abgegrast.

## Konzept: Die 2D→3D-Brücke

**Grundidee:** Ein einzelnes 2D-Foto enthält implizit viel mehr 3D-Information, als man denkt — unser Gehirn rekonstruiert ja auch sofort eine räumliche Vorstellung. Die Frage ist, ob ein Modell das ebenfalls kann, speziell für Pflanzen.

### Trainings-Setup

Ein neuronales Netz wird auf Paaren aus *2D-Bild ↔ bekannter 3D-Struktur* trainiert. 3D-Daten kommen aus:

- bestehenden Datensätzen (Pheno4D etc.), oder
- synthetisch aus L-System-Simulationen, wo man die Pflanze sowohl rendern als auch ihre exakte Geometrie kennt.

Das Netz lernt: "Wenn ich diese Blattanordnung, diese Schattierung, diese Verdeckungen sehe, ist die wahrscheinlichste 3D-Struktur dahinter folgende…"

### Output-Repräsentationen (Auswahlfrage)

Im Einsatz bekommt das Modell ein neues Foto und gibt eine **Strukturhypothese** aus. Mögliche Formen:

- Punktwolke
- Skelettgraph (Stängel, Verzweigungen, Blätter als Knoten/Kanten)
- Parameter eines prozeduralen Modells (Phyllotaxis-Winkel, Anzahl Internodien, Blattlänge etc.)
- NeRF / Gaussian Splat, das aus neuen Winkeln gerendert werden kann

### Warum "Hypothese" und nicht "Rekonstruktion"

Aus *einem* Foto ist 3D prinzipiell unterbestimmt — die Rückseite hat man nie gesehen. Das Modell rät plausibel auf Basis dessen, was es bei der Art typischerweise gesehen hat.

Bei Pflanzen ist das gar nicht so verrückt: viele Arten haben **strukturelle Regelmäßigkeiten** (Phyllotaxis, Verzweigungsmuster, typische Blattstellung). Genau diese kann ein Netz lernen.

### Erwarteter Mehrwert für Erkennung

Statt das CNN nur sagen zu lassen "dieses Foto sieht aus wie Art X", hätte man eine Zwischenebene: "Aus dem Foto folgt strukturell *das* — und solche Strukturen kommen bei Arten X, Y, Z vor."

Bei den schwierigen Doldenblütlern z. B. ist die *3D-Anordnung* der Döldchen, die Verzweigungswinkel und die Höhenstaffelung diagnostisch — auf einem flachen Foto geht das oft verloren, in einer rekonstruierten Struktur nicht.

## Warum Apiaceae (Doldenblütler) als Startpunkt

- Notorisch schwer zu bestimmen, selbst für erfahrene Botaniker.
- Diagnostische Merkmale sind oft **strukturell-räumlich** (Anordnung der Döldchen, Verzweigungswinkel, Höhenverhältnisse), nicht primär texturell.
- Praktische Relevanz: Familie enthält essbare Arten neben hochgiftigen (z. B. Wiesen-Kerbel vs. Gefleckter Schierling).
- 2D-Modelle sind hier nachweislich schwächer → größerer potenzieller Mehrwert durch 3D-Information.
- Eingrenzung der Klade hält den Daten- und Modellaufwand handhabbar.

Alternativen, die ähnlich attraktiv wären: Gräser (Poaceae), bestimmte Korbblütler.

## Verwandte Forschungslinien (zum Andocken)

- **Single-image 3D reconstruction** allgemein — für Objekte und Menschen schon weit, für Pflanzen kaum.
- **Inverse procedural modeling** — aus einem Bild die Parameter eines L-Systems schätzen.
- **Neural Radiance Fields aus wenigen Bildern** — PixelNeRF, SparseNeRF.
- **Differentiable rendering** — eine Hypothese rendern und mit dem Foto vergleichen, um sie iterativ anzupassen.

## Realismus-Check

Das Forschungsgebiet ist jung genug, dass auch ein gut gemachtes Hobby-/Nebenprojekt mit klarem Fokus ("2D→3D nur für Apiaceae") publikationsreife Ergebnisse liefern könnte. Die großen Player kümmern sich eher um Nutzpflanzen, weil dort das Geld liegt.

## Datensatz-Recherche (Stand 2026-05-03)

| Datensatz | Inhalt | Lizenz | Größe | Format | Quelle |
|---|---|---|---|---|---|
| **Pheno4D** | 7 Mais + 7 Tomate, 224 Punktwolken (140 Tomate / 84 Mais), 126 davon gelabelt (~260 Mio. Punkte), zeitliche Serien über 2–3 Wochen | CC BY | **4.44 GB** (Pheno4D.zip, via HEAD-Request verifiziert) | `.xyz` (x,y,z + Label: soil/stem/leaf) | [ipb.uni-bonn.de/data/pheno4d](https://www.ipb.uni-bonn.de/data/pheno4d/), Direkt-Link `https://www.ipb.uni-bonn.de/html/projects/Pheno4D/Pheno4D.zip` |
| **ROSE-X** | 11 reale Rosenbüsche, X-ray-CT; pro Pflanze: rohe X-ray-Stacks, Binärvolumen-Masken, Organ-Labels (leaf/stem/flower), Punktwolken | CC BY 4.0 | **1.53 GB** (ROSE-X.zip, via Range-GET verifiziert) | Voxel-Volumen + Punktwolken | Single-File-Share `ROSE-X.zip` auf Nextcloud der Uni Angers: [uabox.univ-angers.fr/.../rnPm5EHFK6Xym9t](https://uabox.univ-angers.fr/index.php/s/rnPm5EHFK6Xym9t) (laut PMC-Version des Papers, [PMC7057657](https://pmc.ncbi.nlm.nih.gov/articles/PMC7057657/)). Hinweis: HEAD wird vom Server geblockt, Download nur per GET. |
| **Plant3D / P3D** | Toolkit + Beispieldaten; Originalstudie: 505 Scans (Tomate, Tabak, Sorghum) über 35 Bedingungen × 20 Zeitpunkte, Faro-ARM-Scanner | MIT (Code-Repo); Datensatzlizenz nicht separat ausgewiesen | Repo enthält nur Sample-`.pcd` + TF-Modelle; vollständige 505-Scan-Sammlung nicht öffentlich | `.pcd` (PCL) | [github.com/iziamtso/P3D](https://github.com/iziamtso/P3D), [Bioinformatics-Paper Ziamtsov & Navlakha 2020](https://academic.oup.com/bioinformatics/article/36/12/3949/5814204) |

**Konsequenz für unser Projekt:**

- Keiner der drei Datensätze enthält Apiaceae. Sie sind nur als *Methodik-/Pretraining-Daten* nutzbar.
- Pheno4D ist der sauberste Kandidat (CC BY, etablierter Host, zeitliche Serien) und mit 4.44 GB lokal handhabbar.
- ROSE-X bleibt als Benchmark/Organ-Segmentierung interessant, ist aber zu klein (n=11) zum Trainieren.
- Plant3D: 505-Scan-Sammlung müsste bei den Autoren (Ziamtsov / Navlakha, Salk Institute) angefragt werden; Lizenz vorher klären.
- → Apiaceae-Daten kommen **synthetisch (L-System)** und/oder aus **Eigenaufnahmen**. Punkt 2 der Recherche-Liste rückt damit nach vorne.

## Apiaceae-Artenkatalog (Erstkalibrierung)

Sechs Arten als Initialset für den synthetischen Generator (`src/synthetic/species.py`). Auswahl folgt der Test-Set-Methodik unten: enthält die sicherheitsrelevanten Verwechslungspaare und einen breiten morphologischen Spread (kleine vs. große Pflanzen, wenige vs. viele Strahlen). Werte aus den verlinkten Wikipedia-Artikeln; für jede Eigenschaft wird der dort angegebene Bereich genutzt, wo Quellen variieren konservativ (mid-range). Feinjustierung gegen Rothmaler / Schmeil-Fitschen / Felddaten steht aus.

| Art | Höhe (cm) | primäre Strahlen | Strahlenlänge (mm) | Pedicels/Umbellet | Pedicel-Länge (mm) | Hüllblätter | Hüllchen | Reflex.-Bracteolen |
|---|---|---|---|---|---|---|---|---|
| *Heracleum sphondylium* (Wiesen-Bärenklau) | 80–200 | 12–30 | 60–125 | 15–30 | 5–15 | nein | ja | nein |
| *Conium maculatum* (Gefleckter Schierling) | 80–250 | 10–20 | 10–35 | 12–18 | 5–10 | ja (klein) | ja | nein |
| *Daucus carota* (Wilde Möhre) | 30–100 | 30–50 | 30–70 | 10–20 | 2–12 | **ja (fiederartig)** | ja | nein |
| *Anthriscus sylvestris* (Wiesen-Kerbel) | 60–170 | 4–10 | 15–30 | 6–12 | 5–10 | nein | ja | nein |
| *Aethusa cynapium* (Hundspetersilie) | 30–80 | 10–20 | 6–26 | 15–25 | 3–8 | nein | ja | **ja (3–4 lang, abwärts)** |
| *Pastinaca sativa* (Pastinak) | 60–180 | 15–25 | 50–100 | 12–35 | 20–50 | nein | nein | nein |

Quellen ([Heracleum sphondylium](https://en.wikipedia.org/wiki/Heracleum_sphondylium), [Conium maculatum](https://en.wikipedia.org/wiki/Conium_maculatum), [Daucus carota](https://en.wikipedia.org/wiki/Daucus_carota), [Anthriscus sylvestris](https://en.wikipedia.org/wiki/Anthriscus_sylvestris), [Aethusa cynapium](https://en.wikipedia.org/wiki/Aethusa_cynapium), [Pastinaca sativa](https://en.wikipedia.org/wiki/Parsnip)).

Schwer trennbare Paare (nur an quantitativen Strahlen-/Pedicel-Werten):
- **Conium ↔ Aethusa**: beide 10–20 primäre Strahlen, ähnliche Pedicel-Zahl. Hauptunterscheidung muss über die **reflexed bracteoles** der Hundspetersilie laufen — das wird im aktuellen L-System noch nicht abgebildet.
- **Heracleum ↔ Pastinaca**: ähnliche Höhen, ähnliche Strahlenlängen. Hauptunterscheidung visuell über **Blütenfarbe** (weiß vs. gelb) und Strahlenanzahl-Median (~20 vs. ~20). Auf Skelett-Ebene allein schwierig.

Beides bestätigt: pure Skelett-Topologie reicht nicht für alle Verwechslungspaare. Mindestens **Hüllblatt-/Hüllchen-Geometrie** und später **Blattform** müssen ergänzt werden.

## SOTA single-view plant reconstruction (2024–2026)

Stand: Mai 2026. Das Feld hat sich in den letzten ~18 Monaten signifikant bewegt; die Methode "ein Foto → 3D-Pflanzenstruktur" ist nicht mehr exotisch, aber der **Apiaceae-Schwerpunkt** und die **Verknüpfung mit Bestimmungs-Performance** bleiben offen.

### Direkt verwandt (single-view → 3D-Pflanzenstruktur)

| Arbeit | Jahr | Repräsentation | Klade | Quelle |
|---|---|---|---|---|
| **TreeFormer** | 2025 | Skelettgraph via tree-constrained graph generation | Bäume | (in Übersichten zitiert) |
| **Tree-D Fusion** | 2024 | Simulation-ready Modelle, Diffusion-Prior | Bäume | (in Übersichten zitiert) |
| **DeepTree** | 2024 | "situated latents" | Bäume | (in Übersichten zitiert) |
| **TreeStructor** | 2025 | Forest-Rekonstruktion mit neural ranking | Bäume | (in Übersichten zitiert) |
| **Neural Hierarchical Decomposition for Single Image Plant Modeling** | CVPR 2025 | hierarchische Zerlegung | allgemein | CVPR 2025 |
| **Learning to reconstruct botanical trees from single images** | 2021/22 | implizite Felder | Bäume | [ACM TOG](https://dl.acm.org/doi/10.1145/3478513.3480525) |
| **Evaluation of one-image 3D reconstruction for plant model generation** | 2025 | Vergleich Hunyuan3D 2.0, Trellis, One2345++, InstantMesh, Direct3D, Unique3D | allgemein | [Plant Methods](https://link.springer.com/article/10.1186/s13007-025-01482-6) |
| **PlantDreamer** | 2025 | Diffusion-guided Gaussian Splatting | allgemein | (in Übersichten zitiert) |
| **ProcGen3D** | Nov 2025 | neural procedural graphs für Image-to-3D | objektübergreifend | [arXiv 2511.07142](https://arxiv.org/html/2511.07142) |
| **Learning to Infer Parameterized Representations of Plants from 3D Scans** | 2025 | parametrisches Pflanzenmodell aus Scans (Inverse Modeling) | allgemein | [arXiv 2505.22337](https://arxiv.org/html/2505.22337v1) |

### Multi-view / Splatting (methodisch nah, aber nicht single-image)

- **High-fidelity wheat plant reconstruction using 3D Gaussian splatting and NeRF** (GigaScience 2025) — 0.74 mm Accuracy gegen Handheld-Scanner. [PMC11945317](https://pmc.ncbi.nlm.nih.gov/articles/PMC11945317/)
- **Plant3R** (2026) — MASt3R + 3DGS für Weizen. [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S2643651526000373)
- **PlantGaussian** (2025) — cross-time / cross-scene. [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S2214514125000261)
- **PlantSegNeRF** (2025) — few-shot, cross-species, joint-channel NeRF mit multi-view instance matching. [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S2589721725001060)
- **GaussianPlant** (Dez 2025) — structure-aligned 3DGS. [arXiv 2512.14087](https://arxiv.org/html/2512.14087)
- **Object-Centric 3DGS for Strawberry Plant Reconstruction** (Nov 2025) — [arXiv 2511.02207](https://arxiv.org/html/2511.02207)
- **NeRF-based 3D reconstruction pipeline for tomato crop morphology** (2024) — [Frontiers](https://www.frontiersin.org/journals/plant-science/articles/10.3389/fpls.2024.1439086/full)

### Survey-Anker

- **A survey on 3D reconstruction techniques in plant phenotyping: From classical methods to NeRF, 3DGS, and beyond** — [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S2643651525001438)
- **Recent Developments in Image-Based 3D Reconstruction Using Deep Learning** (2025) — [MDPI Electronics](https://www.mdpi.com/2079-9292/14/15/3032)
- **A Review of Optical-Based 3D Reconstruction and Multi-Source Fusion for Plant Phenotyping** — [PMC12158188](https://pmc.ncbi.nlm.nih.gov/articles/PMC12158188/)

### Was das für unser Projekt heißt

- **Methode ist da:** Single-view → Skelettgraph ist Stand der Forschung (TreeFormer, Tree-D Fusion, DeepTree). Wir müssen das Verfahren nicht neu erfinden, sondern auf eine andere Klade adaptieren.
- **Fokus liegt fast vollständig auf Bäumen / Nutzpflanzen.** Krautige Pflanzen mit Doldenstruktur kommen in keiner der gefundenen Arbeiten vor.
- **Generative Image-to-3D-Tools (Hunyuan3D, Trellis, One2345++, InstantMesh) sind 2025 erstmals systematisch auf Pflanzen evaluiert worden** — Qualität ist gemischt, weil sie auf rigide Objekte optimiert sind. Für Apiaceae mit dünnen Stängeln und feinen Döldchen erwartbar schwach → eigener, auf Pflanzen-Strukturpriors zugeschnittener Ansatz hat Daseinsberechtigung.
- **Bestimmungs-Endmetrik ist der Differenzierer.** Die SOTA-Arbeiten messen Rekonstruktionsqualität (Chamfer Distance, Visualisierung). Niemand misst, ob die Rekonstruktion die *Artbestimmung* messbar verbessert. Genau das ist unsere Hypothese und unser potentieller Eigenbeitrag.
- **Inverse Procedural Modeling als Pivot:** Statt Punktwolke direkt vorherzusagen, könnte man L-System-Parameter vorhersagen (siehe ProcGen3D, "Learning to Infer Parameterized Representations"). Das passt zur botanischen Interpretierbarkeit, die wir wollen.

## Gaussian Splatting für Pflanzen — Eignung für dünne Strukturen

### Allgemein (2024–2026)

- Historische Schwäche von 3DGS: dünne lineare Strukturen (Drähte, Zäune, Antennen, Vegetation) zeigen Artefakte, weil Gaussians schlecht zu langen geraden Strukturen passen.
- Grundproblem: 3DGS optimiert **photometrischen** Loss, nicht **geometrischen**. Ein Stängel kann optisch korrekt aussehen, aber als 3D-Geometrie unsauber sein.
- 2025: kleinere Verbesserungen (z. B. SPZ 2.0 mit Quaternion-Encoding für lineare Features) lindern das Problem partiell für geospatiale Anwendungen, sind aber kein vollständiger Fix.

### Speziell für Pflanzen

Bessere Nachrichten als allgemein erwartet:

- 3DGS bildet dünne Stängel oft **besser ab als gedacht**, weil die adaptive Dichte-Kontrolle automatisch viele Gaussians in stängelreiche Regionen packt (genannt z. B. in [GigaScience 2025, Stuart et al.](https://academic.oup.com/gigascience/article/doi/10.1093/gigascience/giaf022/8096368) — 0.74 mm Accuracy bei Weizen).
- Aber: "Stem reconstruction remains future work" — quantitativ ist die Stängelgeometrie noch nicht gelöst, vor allem bei dünnen Trieben unter Selbstverdeckung.

### Bekannte Pflanzen-3DGS-Methoden

| Arbeit | Jahr | Schwerpunkt | Quelle |
|---|---|---|---|
| **Splanting** | SIGGRAPH Asia 2024 | allgemeine Pflanzenerfassung mit 3DGS | [ACM](https://dl.acm.org/doi/10.1145/3681758.3698009) |
| **GigaScience-Weizen-Paper** | 2025 | Weizen, Vergleich 3DGS vs. NeRF, 0.74 mm vs. Handheld-Scanner | [PMC11945317](https://pmc.ncbi.nlm.nih.gov/articles/PMC11945317/) |
| **PlantGaussian** | 2025 | cross-time / cross-scene Visualisierung | [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S2214514125000261) |
| **PlantDreamer** | ICCV 2025 W | Diffusion-guided 3DGS, Pflanzen-Generierung | [CVF](https://openaccess.thecvf.com/content/ICCV2025W/CVPPA/papers/Hartley_PlantDreamer_Achieving_Realistic_3D_Plant_Models_with_Diffusion-Guided_Gaussian_Splatting_ICCVW_2025_paper.pdf) |
| **VPGS** | 2026 | virtuelle Pflanzenrekonstruktion + Rendering | [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0097849326000579) |
| **P3DFusion** | 2025 | cross-scene high-fidelity, Vision-Foundation-Models + 3DGS | [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S1161030125003077) |
| **LeafFit** | 2026 | speziell Blätter, Asset-Erstellung | [Wiley](https://onlinelibrary.wiley.com/doi/10.1111/cgf.70374) |
| **GaussianPlant** | Dez 2025 | structure-aligned 3DGS | [arXiv 2512.14087](https://arxiv.org/html/2512.14087) |
| **Plant3R** | 2026 | MASt3R + 3DGS für Weizen | [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S2643651526000373) |

### Bekannte Schwächen für unseren Use Case

- **Blätter (dünne Flächen):** Implizite Isosurface-Extraktion versagt oft; Ball-Pivoting ist sensitiv gegenüber ungleichmäßiger Gaussian-Dichte.
- **Repetitive Texturen** (viele kleine ähnliche Blätter / Döldchen): SfM-Initialwolke wird sparse, adaptive Density Control hat in textur-armen Regionen Probleme.
- **Speicher:** komplexe Blattstrukturen brauchen viele Gaussians; mitigierbar über Instanz-Wiederverwendung.
- **Wind/Outdoor:** Vanilla 3DGS hat keine temporale Modellierung. Bewegliche Pflanzenteile zwischen den Aufnahmen erzeugen Floaters. Dynamische 3DGS-Varianten existieren, sind aber komplexer zu trainieren.

### Konsequenz für unser Apiaceae-Projekt

- **3DGS ist ein gutes *Render-/Visualisierungs-Backend*** — wir können eine vorhergesagte Strukturhypothese photorealistisch rendern und z. B. gegen das Eingangsfoto einen photometrischen Loss laufen lassen (differentiable rendering loop).
- **3DGS ist als *interne Repräsentation* für Klassifikation ungeeignet** — eine Wolke aus Gaussians hat keine Skelett-Topologie und keine botanisch interpretierbaren Parameter. Für die Bestimmung brauchen wir Skelettgraph oder L-System-Parameter.
- **Empfohlener Stack:**
  1. Skelett-/Parameter-Vorhersage als primäre Hypothese (interpretierbar, klassifikations-tauglich)
  2. 3DGS optional darüber als photorealistisches Rendering, sowohl zur Validierung gegen das Eingangsfoto als auch zur Anschauung
- **Doldenstruktur ist genau die Art von dünner, repetitiver, selbstverdeckender Geometrie**, bei der 3DGS Probleme hat. Aber die Probleme sind bekannt und werden 2025/26 aktiv adressiert — gut, aber kein Selbstläufer.

## Test-Set & Bestimmungs-Metriken — Methodik

Das Ziel ist messbar zu zeigen, dass die 3D-Zwischenebene die Bestimmungsgenauigkeit gegenüber einer reinen 2D-Baseline erhöht. Unten zusammengetragen, wie die Plant-ID-Community das macht, plus ein konkreter Vorschlag für unser Setup.

### Etablierte Praxis (PlantCLEF / LifeCLEF / Plant-ID-Apps)

**Metriken**
- **Top-1 Accuracy** — Hauptmetrik in PlantCLEF 2019
- **Top-3 / Top-5 Accuracy** — bei eng verwandten Arten realistischer; in Apps wie PlantNet werden Nutzer:innen mehrere Vorschläge gezeigt
- **Mean Reciprocal Rank (MRR)** — mittelt 1/Rank des korrekten Treffers; bestraft "Top-5 ja, aber an Position 5"
- **Confusion-Matrix** — zeigt, *welche* Arten miteinander verwechselt werden; bei Apiaceae diagnostisch relevant (welches sicherheitsrelevante Paar wird vertauscht?)
- **MRR auf "schwierige Subsets"** — PlantCLEF 2020 hat erstmals separat MRR über selten fotografierte / schwer zu bestimmende Arten ausgewertet. Genau diese Logik passt zu unserem Apiaceae-Fokus.

**Ground-Truth-Aufbau**
- Mehrere Botaniker:innen annotieren parallel; Konfliktauflösung durch Drittgutachter oder Konsens
- PlantCLEF unterscheidet vier Expertisestufen (Flora-Experte / Experte / Amateur / Novize) — sinnvoll, weil ein Modell auch *gegen* unterschiedliche Nutzer:innen-Niveaus verglichen werden kann
- Bekannte Fehlerquellen im Ground Truth: Taxonomie-Änderungen / Synonyme, mehrere Arten in einem Bild, Entwicklungsstadium (vegetativ vs. blühend), Bildqualität

**SOTA-Referenzwerte (zur Einordnung)**
- ViT-Large/16 auf PlantCLEF 2017: 91.15 % Top-1
- ViT-Large/16 auf ExpertLifeCLEF 2018: 83.54 %
- 10.000-Arten-Ensembles für Europa/Nordamerika: ~90 %, für Amazonia: ~40 %
- Beste PlantCLEF-Submissions: MRR 92 %, Top-5 96 %

### Niemand hat einen Apiaceae-spezifischen Benchmark publiziert

Suche nach "Apiaceae/Umbelliferae + CNN + Identification" liefert nichts Spezifisches. Allgemeine Plant-ID-Benchmarks (PlantCLEF, ExpertLifeCLEF) enthalten Doldenblütler nur als kleine Untermenge — ihre Performance auf genau dieser Familie wird nicht separat berichtet. Das ist eine **eigene publizierbare Lücke**: alleine die Frage "wie gut sind heutige 2D-CNNs auf Apiaceae-Bestimmung wirklich?" hat noch niemand sauber beantwortet.

### Vorschlag für unser Setup

**Artenliste (Vorschlag)**

Beschränkung auf 15–30 mitteleuropäische Apiaceae-Arten, gewählt nach drei Kriterien:
1. **Sicherheitsrelevante Verwechslungspaare:** Wiesen-Kerbel ↔ Gefleckter Schierling; Wilder Pastinak ↔ Riesen-Bärenklau; Wilde Möhre ↔ Hundspetersilie
2. **Häufigkeit / Verfügbarkeit für Eigenaufnahmen** im Bad-Schoenborn-Umfeld (Wegränder, Wiesen)
3. **Strukturelle Diversität** der Dolden (einfache vs. doppelte Dolden, Hüllblätter mit/ohne, Strahlenanzahl-Spektrum)

**Datenakquisition**
- Pro Art ≥ 30 Beobachtungen, jede mit mehreren Fotos aus verschiedenen Blickwinkeln
- Pro Beobachtung *mindestens* eine Übersichtsaufnahme der ganzen Pflanze plus Detail Dolde
- Optional: iPhone-LiDAR-Scan derselben Pflanze als räumlicher Ground Truth (geht direkt in unser 2D→3D-Trainingsziel)
- GPS, Datum, Phänologie-Stadium (vegetativ / Knospe / Blüte / Frucht) immer mitlabeln

**Validierung**
- Erste Bestimmung selbst (RESEARCH.md-Glossar + Bestimmungsschlüssel)
- Zweite, unabhängige Validierung durch botanischen Sparringspartner (Punkt 5 der Recherche-Liste — z. B. Botanischer Garten Heidelberg)
- Konfliktfälle als "uncertain" markieren, nicht in Test-Split aufnehmen

**Splits**
- Train / Val / Test mit Trennung **auf Beobachtungs-Ebene**, nicht Foto-Ebene — sonst landen Fotos derselben Pflanze in Train *und* Test
- Geographische Trennung wenn möglich (z. B. Test-Set aus anderem Naturraum)
- Für die "schwierige Subset"-Metrik à la PlantCLEF 2020: dedizierter Sub-Test nur mit den Sicherheits-Verwechslungspaaren

**Vergleichsmodelle**
1. **Baseline A:** Off-the-shelf 2D-Klassifikator — entweder PlantNet-API-Calls (echte App-Performance) oder ein selbst feingetuntes ResNet/ViT auf PlantCLEF + iNaturalist-Apiaceae-Subset
2. **Baseline B:** Multi-View-2D — mehrere Fotos derselben Pflanze, Late Fusion der Vorhersagen
3. **Unsere Methode:** 2D-Foto → 3D-Strukturhypothese (Skelettgraph oder L-System-Parameter) → Klassifikator auf der 3D-Zwischenebene
4. **Optional Ablation:** 3D-Hypothese + Originalfoto kombiniert (Late Fusion auf Feature-Ebene)

**Berichtete Metriken**
- Top-1, Top-3, Top-5 Accuracy
- MRR insgesamt und MRR auf "kritisches Verwechslungs-Subset"
- Confusion-Matrix mit besonderem Fokus auf sicherheitsrelevante Paare
- Performance pro Phänologie-Stadium (deckt die Hypothese auf, dass 3D im vegetativen Stadium besonders hilft, wenn Blüten fehlen)
- Wenn iPhone-LiDAR-Ground-Truth vorhanden: zusätzlich **Chamfer Distance** der vorhergesagten 3D-Hypothese gegen den realen Scan — als Sanity-Check, dass die 3D-Zwischenebene überhaupt strukturell sinnvoll ist und nicht nur ein Performance-Trick

**Mindestumfang für eine ehrliche Aussage**
- Grobe Schätzung: 20 Arten × 30 Beobachtungen × 5 Fotos = **3.000 Bilder**, davon ~10 % im Test-Set
- Annotation: 2 Botaniker × ~600 Test-Bilder × 1 min = ~20 h Personenaufwand auf der Validierungsseite — überschaubar

### Risiken / offene Frage

- **iNaturalist-Lecks:** Wenn die 2D-Baseline gegen ein Modell trainiert wird, das implizit unser Test-Set schon kennt (z. B. selbe Beobachtungen via iNaturalist gelabelt), ist der Vergleich verzerrt. Lösung: Test-Set ausschließlich aus Eigenaufnahmen, nichts auf iNaturalist hochgeladen vor Auswertung.
- **Ground-Truth-Subjektivität bei *wirklich* schwierigen Paaren:** Selbst Botaniker:innen sind sich bei Apiaceae nicht immer einig. → Solche Fälle als "uncertain" markieren statt forcen.
- **Fairer Vergleich:** Unsere Methode darf nicht heimlich mehr Eingangs-Information nutzen (z. B. mehrere Fotos für 3D-Rekonstruktion), als die Baseline. Single-image vs. single-image, gleiche Compute-Budget-Klasse.

## Offene Recherche-Punkte

- [x] Lizenzbedingungen und Größe von Pheno4D, ROSE-X, Plant3D im Detail prüfen — siehe Tabelle oben. Plant3D-Vollscan-Verfügbarkeit noch offen (Mail an Autoren).
- [x] State of the Art bei single-view plant reconstruction (Papers 2024–2026) sichten — siehe Abschnitt "SOTA single-view plant reconstruction" unten.
- [x] Existierende Apiaceae-3D-Daten überhaupt vorhanden? **Nein.** Geprüft per Suche nach Familie, Trivialnamen (umbellifer, Doldenblütler, fennel, wild carrot) und typischen Gattungen (Daucus, Heracleum, Anthriscus). Verfügbare 3D-Plant-Datensätze decken Mais, Tomate, Tabak, Sorghum, Rose, Hülsenfrüchte, Weizen, Cotton, Reis, Kartoffel, Raps, Kohl ab — keine Apiaceae. Karotte existiert nur in 2D (Wurzelform-Phänotypisierung, Harvard Dataverse). → Apiaceae-Daten müssen **synthetisch (L-System / GroIMP)** und/oder **per Eigenaufnahme (iPhone-LiDAR, Polycam, SfM)** entstehen. Bestätigt die Strategie und untermauert den originellen Beitrag des Projekts (echte Lücke, nicht nur Nische).
- [ ] Existierende Apiaceae-3D-Daten überhaupt vorhanden? Wenn nein: synthetisch + Eigenaufnahmen
- [ ] State of the Art bei single-view plant reconstruction: aktuelle Papers (2024–2026) sichten
- [ ] Gaussian-Splatting-Pipelines: Eignung für dünne Strukturen wie Pflanzenstängel
- [ ] Botanischer Sparringspartner finden (z. B. Botanischer Garten Heidelberg / Universität)
- [x] Test-Set definieren: wie misst man "bessere Bestimmung" sauber? — siehe Abschnitt "Test-Set & Bestimmungs-Metriken — Methodik" oben.

## Glossar

- **Klade:** Abstammungsgemeinschaft — gemeinsamer Vorfahre + alle Nachkommen. Z. B. Apiaceae als Klade innerhalb der Bedecktsamer.
- **Phyllotaxis:** Blattstellung, also die geometrische Regel, nach der Blätter um den Stängel angeordnet sind (typische Winkel: 137,5°).
- **L-System:** Lindenmayer-System, formales Grammatik-System zur prozeduralen Beschreibung von Pflanzenwachstum.
- **NeRF:** Neural Radiance Field, neuronale Repräsentation einer 3D-Szene als Funktion (Position, Blickrichtung) → (Farbe, Dichte).
- **Gaussian Splatting:** 3D-Repräsentation mit vielen kleinen Gauss-Verteilungen, schnell zu rendern.
