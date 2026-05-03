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
