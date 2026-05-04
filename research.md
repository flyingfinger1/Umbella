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
