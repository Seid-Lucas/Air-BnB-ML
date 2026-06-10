# 📌 Multi-Modale Airbnb Preisvorhersage & Datenanalyse (Berlin)

Dieses Repository enthält ein umfassendes, fortgeschrittenes Data-Science- und Machine-Learning-Projekt zur Analyse und multi-modalen Preisvorhersage von Airbnb-Inseraten in Berlin. Das Projekt verarbeitet strukturierte Tabellendaten, geografische Koordinaten, zeitliche Verläufe, unstrukturierten Text (Reviews/Beschreibungen) sowie Bilddaten über ein Convolutional Neural Network (CNN).

Ein besonderer Fokus liegt auf der methodisch sauberen Modellierung unter strikter Vermeidung von **Data Leakage**.

---

## 🗺️ Inhaltsverzeichnis (TOC)

1. [📥 Datenimport und Überblick](#1-datenimport-und-überblick)
2. [🧹 Datenbereinigung](#2-datenbereinigung)
3. [🧪 Feature Engineering](#3-feature-engineering)
4. [📊 Explorative Datenanalyse (EDA)](#4-explorative-datenanalyse-eda)
5. [🤖 Modellierung (Baseline & Fair)](#5-modellierung-baseline--fair)
6. [🌍 Geodatenanalyse](#6-geodatenanalyse)
7. [🗓️ Kalenderdaten](#7-kalenderdaten)
8. [📝 NLP & Sentiment-Analyse (Reviews & TF-IDF)](#8-nlp--sentiment-analyse-reviews--tf-idf)
9. [🖼️ Bildbasierte Features (CNN)](#9-bildbasierte-features-cnn)
10. [📈 Modellvergleich & Hyperparameter-Tuning](#10-modellvergleich--hyperparameter-tuning)
11. [🔎 Fehleranalyse (Residuals)](#11-fehleranalyse-residuals)
12. [🏆 Finales Multi-Modal Modell & Auswertung](#12-finales-multi-modal-modell--auswertung)

---

## 🚀 Technologie-Stack & Installation

Das Projekt basiert auf Python 3.13+ und nutzt moderne Bibliotheken für Machine Learning, Deep Learning, Geodatenverarbeitung und NLP.

### Voraussetzungen

Installieren Sie die benötigten Pakete mit folgendem Befehl:

```bash
pip install "numpy<2.0" geopy folium textblob torch torchvision pillow tqdm scikit-learn seaborn pandas matplotlib
```

### Kern-Bibliotheken
*   **Datenverarbeitung:** `pandas`, `numpy`
*   **Visualisierung:** `matplotlib`, `seaborn`, `folium` (interaktive Karten)
*   **Machine Learning:** `scikit-learn`
*   **Deep Learning (Computer Vision):** `torch`, `torchvision` (ResNet50)
*   **NLP & Textanalyse:** `nltk`, `textblob`
*   **Geodaten-Berechnung:** `geopy`

---

## 📂 Projekt-Pipeline im Detail

### 1. 📥 Datenimport und Überblick
*   **Datenbasis:** Einlesen der komprimierten Inside-Airbnb-Daten (`listings.csv.gz`) für Berlin.
*   **Dimensionen:** Der Datensatz umfasst initial **13.945 Zeilen und 79 Spalten**.
*   **Identifikation von Schwachstellen:** Analyse von fehlenden Werten (Sparsity) pro Spalte zur Bestimmung der Bereinigungsstrategie.

### 2. 🧹 Datenbereinigung
*   **Spalten-Filterung:** Automatischer Ausschluss von Features mit einer Missing-Rate von über 50 % (z. B. `calendar_updated`).
*   **Metadaten-Bereinigung:** Entfernung von URLs, IDs und Freitextelementen, die keinen direkten prädiktiven Mehrwert bieten (z. B. `host_picture_url`, `scrape_id`).
*   **Datentyp-Konvertierung:** Transformation von Währungs-Strings (z. B. `"$1,000.00"`) in numerische Fließkommazahlen (`float`).
*   **Imputation:** Sinnvolles Auffüllen fehlender Werte (z. B. Ersetzen von NaN bei Review-Metriken durch `0`).

### 3. 🧪 Feature Engineering
*   **Berliner Bezirks-Mapping:** Zuordnung von über 100 feingranularen Nachbarschaften (`neighbourhood_cleansed`) zu den 12 offiziellen Berliner Bezirken (z. B. *Mitte, Friedrichshain-Kreuzberg, Pankow, Charlottenburg-Wilmersdorf*).
*   **Ökonomische Indikatoren:** Erstellung von Features wie `price_per_person`.
*   **Textstatistiken:** Extraktion von Längenmerkmalen (`description_length`, `description_word_count`) aus den Inseratstexten.
*   **Zielvariablen-Transformation:** Logarithmierung des Preises (`log_price = np.log1p(price)`), um Rechtsschiefe zu minimieren und die Modellkonvergenz zu verbessern.
*   **Kategoriale Codierung:** Anwendung von One-Hot-Encoding auf Features wie `room_type`, `property_type` und `Bezirk`.

⚠️ **Wichtig: Data Leakage Vermeidung**
Zur Gewährleistung einer validen Evaluierung wurden Variablen, die direkt oder indirekt mit der Zielvariable korrespondieren oder aus der Zukunft stammen, für das faire Modell strikt gesperrt:
```python
LEAKAGE_COLUMNS = [
    "price", "log_price", "price_per_person",
    "estimated_revenue_l365d", "avg_price_last_365d",
    "estimated_occupancy_l365d"
]
```

### 4. 📊 Explorative Datenanalyse (EDA)
*   Statistische und visuelle Aufbereitung der Verteilungen.
*   Analyse der Angebotsdichte pro Berliner Bezirk.
*   Untersuchung der geschätzten durchschnittlichen Tagespreise im geografischen Vergleich.

### 5. 🤖 Modellierung (Baseline & Fair)
Gegenüberstellung eines klassischen linearen Modells und eines nicht-linearen Ensemble-Verfahrens:
*   **Linear Regression (Baseline):** Dient als statistische Baseline.
*   **Random Forest Regressor (Fair):** Trainiert ausschließlich auf bereinigten, zulässigen Features unter Ausschluss der `LEAKAGE_COLUMNS`.

### 6. 🌍 Geodatenanalyse
*   Nutzung von `geopy`, um Distanzen zu zentralen Sehenswürdigkeiten Berlins (z. B. Brandenburger Tor, Alexanderplatz) via Geodaten-Berechnung (`geodesic`) zu ermitteln.
*   Generierung interaktiver Heatmaps und Punkt-Cluster-Karten mittels `folium` zur Identifikation von Hotspots.

### 7. 🗓️ Kalenderdaten
*   Extraktion von Zeitreihen-Features aus Buchungskalendern und Review-Historien.
*   Berechnung von Saisonalitäten, Buchungszyklen und zeitlichen Abständen (`days_since_last_review`).

### 8. 📝 NLP & Sentiment-Analyse (Reviews & TF-IDF)
*   **Sentiment Scoring:** Analyse der Gästebewertungen mit `TextBlob` zur Ermittlung der Polarität (positiv/negativ).
*   **Text-Mining:** Verarbeitung der Beschreibungen (`description`) mittels `TfidfVectorizer` zur Erfassung signifikanter Schlüsselwörter, reduziert via Hauptkomponentenanalyse (PCA).

### 9. 🖼️ Bildbasierte Features (CNN)
*   Extraktion visueller Features aus den Teaser-Bildern der Inserate (`picture_url`).
*   Verwendung eines via **PyTorch** geladenen, vortrainierten **ResNet50**-Netzwerks zur Erzeugung hochdimensionaler Feature-Embeddings, die als Input für die finale Preisprognose dienen.

### 10. 📈 Modellvergleich & Hyperparameter-Tuning
*   Systematischer Vergleich aller Zwischenmodelle anhand standardisierter Metriken: **MAE, RMSE, R²-Score**.
*   Optimierung des Random Forest Modells und fortgeschrittener Regressoren mittels `GridSearchCV`.

### 11. 🔎 Fehleranalyse (Residuals)
*   Explizite Analyse der Modellresiduellwerte (Vorhersagefehler).
*   Untersuchung, in welchen Preissegmenten (z. B. Luxussegmente vs. Budget-Zimmer) das Modell Unter- oder Überanpassung zeigt.

### 12. 🏆 Finales Multi-Modal Modell & Auswertung
*   **Late/Early Fusion:** Zusammenführung der Tabellenfeatures, NLP-Textkomponenten, Geodaten-Features und CNN-Bild-Embeddings zu einem einheitlichen, multi-modalen Datensatz.
*   **Endauswertung:** Validierung des finalen Modells auf dem Testset zur präzisen und fairen Bestimmung des Inseratswerts.

---

## 📊 Erste Ergebnisse aus der Modellierung

In einer ersten Validierung der klassischen tabellarischen Daten wurden folgende Metriken erzielt:

| Modell | MAE | RMSE | R²-Score |
| :--- | :---: | :---: | :---: |
| **Linear Regression** | 0.46 | 0.61 | -0.00 |
| **Random Forest (Fair)** | **0.25** | **0.35** | **0.66** |

*Hinweis: Der R²-Wert von 0.66 zeigt, dass das faire Random-Forest-Modell bereits ohne die erweiterten Text- und Bildfeatures rund 66% der Varianz des logierten Preises erklären kann.*

---

## 🛠️ Ausführung

Öffnen Sie das Jupyter Notebook in Ihrer favorisierten Entwicklungsumgebung (z. B. VS Code oder JupyterLab) und führen Sie die Zellen sequenziell aus:

```bash
jupyter lab
```

Stellen Sie sicher, dass sich die Quelldatei unter `Data/listings.csv.gz` befindet.
