# EcoMoveNet – Implementation Guide

## Overview
Full implementation of **Algorithm 1: EcoMoveNet** — a Gamified AI Framework for
Sustainable Commuting and Behavioral Change — with comparative analysis between
the **NYC TLC Trip Record** dataset and the **EcoMoveNet app** dataset.


---

## Installation

```bash
pip install -r requirements.txt
python -m textblob.download_corpora   # Download TextBlob corpora (one-time)
```

---

## Running the Pipeline

```bash
python run_ecomovenet.py \
    --nyc  /path/to/yellow_tripdata_2024-01.parquet \
    --eco  /path/to/ecomovenet_data.csv \
    --transcripts /path/to/interview_transcripts.txt \
    --out  ./results
```

| Argument | Description |
|----------|-------------|
| `--nyc`  | NYC TLC file (`.parquet` or `.csv`) — downloaded from https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page |
| `--eco`  | EcoMoveNet app dataset (`.csv`) |
| `--transcripts` | Interview transcript text file (`.txt`) for sentiment analysis (optional) |
| `--out`  | Output directory (default: `./outputs`) |

---

## Algorithm Steps Implemented

| Step | Description | Module |
|------|-------------|--------|
| 1–2  | Load datasets, remove missing values/noise | `DataLoader` |
| 3    | Extract travel features (time, distance, passengers, origin, destination) | `FeatureEngineer.extract_features()` |
| 4    | Normalise numeric + encode categorical variables | `FeatureEngineer.encode_and_scale()` |
| 5    | Spatio-temporal analysis – peak hours & travel zones | `SpatioTemporalAnalyzer.identify_peak_hours()` |
| 6    | Cluster frequent routes via location similarity | `SpatioTemporalAnalyzer.cluster_routes()` |
| 7–9  | Carpool candidate matching + ranking | `CarpoolMatcher` |
| 10–11| Train MLP neural model + predict ride-sharing probability | `RideSharingPredictor` |
| 10b  | Sequence-based travel behavior modelling (PyTorch LSTM) | `LSTMTravelBehaviorModel` |
| 12–13| Compute engagement scores + personalise incentives | `GamificationEngine` |
| 14   | Compute EcoScore (shared rides + emission reduction) | `EcoScoreCalculator` |
| 15   | Sentiment analysis on transcript lines using TextBlob + BERT + ALBERT | `BehavioralAnalyzer`, `TransformerSentimentClassifier` |
| 16   | Generate personalised carpool recommendations | `CarpoolMatcher.generate_recommendations()` |
| 17   | Evaluate with Accuracy, Precision, Recall, F1, AUC | `RideSharingPredictor.evaluate()` |

---

## Ablation Study

Five model variants are evaluated automatically:

| Variant | Components |
|---------|-----------|
| Baseline | Logistic Regression only |
| Temporal Only | + Hour, DOW, duration features |
| Spatial + Temporal | + Distance, location clusters |
| Spatial + Temporal + Behavior | + Fill ratio, carbon, efficiency |
| **Hybrid (Full Model)** | + Incentive/payment features, MLP |

Metrics: **Accuracy (%), Precision (%), Recall (%), F1-score (%), AUC-ROC, RMSE**

---

## Outputs Generated

| File | Description |
|------|-------------|
| `ablation_results.csv`        | Ablation table as CSV |
| `comparative_analysis.json`   | NYC TLC vs EcoMoveNet metrics |
| `bert_sentiment_results.csv`  | BERT sentiment predictions on transcript lines |
| `albert_sentiment_results.csv`| ALBERT sentiment predictions on transcript lines |
| `ablation_study.png`          | Bar chart – ablation metrics |
| `comparative_analysis.png`    | Side-by-side dataset comparison |
| `ecoscore_distribution.png`   | EcoScore histograms |
| `sentiment_analysis.png`      | Sentiment pie + polarity histogram |
| `peak_hours_heatmap.png`      | Hour × Day demand heatmap |

---

## EcoMoveNet Dataset Column Requirements

| Column | Description |
|--------|-------------|
| `ride_id` | Unique ride identifier |
| `ride_type` | `"Carpool"` or `"Private"` |
| `pickup_datetime` | ISO datetime |
| `dropoff_datetime` | ISO datetime |
| `pickup_location` | GPS or locality name |
| `dropoff_location` | GPS or locality name |
| `trip_distance_km` | Distance in km |
| `num_passengers` | Actual passengers |
| `avg_carpoolers` | Average carpoolers (carpool rides) |
| `fare_rs` | Fare in INR |
| `payment_type` | Cash / Card / Wallet / UPI |
| `rate_type` | Standard / Shared / Premium |
| `carbon_saved_kg` | Carbon saved vs private (carpool only) |
| `carbon_emitted_kg` | Estimated carbon emitted |
| `efficiency_score` | Sustainability score (0–1) |

---

## NYC TLC Dataset

Download monthly Parquet files from:
https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page

The pipeline auto-maps TLC column names (`tpep_pickup_datetime`, `PULocationID`, etc.)
and derives a carpool proxy label from `passenger_count > 1`.
