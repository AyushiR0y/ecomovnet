"""
EcoMoveNet: A Gamified AI Framework for Sustainable Commuting and Behavioral Change
Algorithm 1: EcoMoveNet Implementation
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from pathlib import Path

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans, DBSCAN
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, confusion_matrix,
                              classification_report)
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from scipy.spatial.distance import cosine
from textblob import TextBlob
import re
from datetime import datetime, timedelta
import json
from typing import List
import importlib


# ─────────────────────────────────────────────
# STEP 1 & 2: Load & Preprocess Datasets
# ─────────────────────────────────────────────

class DataLoader:
    """Handles loading and initial preprocessing of both NYC TLC and EcoMoveNet datasets."""

    def __init__(self, nyc_path: str = None, eco_path: str = None, max_nyc_rows: int = 300000):
        self.nyc_path = nyc_path
        self.eco_path = eco_path
        self.max_nyc_rows = max_nyc_rows
        self.nyc_df = None
        self.eco_df = None

    def _get_nyc_columns_to_load(self, available_columns: List[str]) -> List[str]:
        """Return a minimal NYC TLC column set required by downstream pipeline steps."""
        needed_lower = {
            'pickup_datetime', 'dropoff_datetime',
            'tpep_pickup_datetime', 'tpep_dropoff_datetime',
            'lpep_pickup_datetime', 'lpep_dropoff_datetime',
            'pickup_location', 'dropoff_location',
            'pulocationid', 'dolocationid',
            'trip_distance_km', 'trip_distance',
            'num_passengers', 'passenger_count',
            'fare_rs', 'fare_amount',
            'payment_type', 'rate_type', 'ratecodeid'
        }
        lower_to_actual = {}
        for col in available_columns:
            lower_to_actual.setdefault(str(col).lower().strip(), col)

        selected = [lower_to_actual[c] for c in needed_lower if c in lower_to_actual]
        return selected

    def load_nyc_data(self) -> pd.DataFrame:
        """Load and preprocess NYC TLC trip record data."""
        print("[DataLoader] Loading NYC TLC dataset...")
        nyc_path_l = str(self.nyc_path).lower()

        if nyc_path_l.endswith('.parquet'):
            try:
                import pyarrow.parquet as pq
                parquet_file = pq.ParquetFile(self.nyc_path)
                selected_cols = self._get_nyc_columns_to_load(parquet_file.schema.names)

                if self.max_nyc_rows and self.max_nyc_rows > 0:
                    parts = []
                    rows_loaded = 0
                    for rg_idx in range(parquet_file.num_row_groups):
                        if rows_loaded >= self.max_nyc_rows:
                            break
                        rg_df = parquet_file.read_row_group(rg_idx, columns=selected_cols).to_pandas()
                        remaining = self.max_nyc_rows - rows_loaded
                        if len(rg_df) > remaining:
                            rg_df = rg_df.head(remaining)
                        parts.append(rg_df)
                        rows_loaded += len(rg_df)

                    df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=selected_cols)
                    print(f"  -> Loaded up to {self.max_nyc_rows:,} NYC rows for memory safety.")
                else:
                    df = pd.read_parquet(self.nyc_path, columns=selected_cols)
            except Exception as err:
                print(f"  -> Column-pruned parquet read failed ({err}); falling back to pandas parquet load.")
                df = pd.read_parquet(self.nyc_path)
                if self.max_nyc_rows and self.max_nyc_rows > 0 and len(df) > self.max_nyc_rows:
                    df = df.head(self.max_nyc_rows)
                    print(f"  -> Truncated NYC dataframe to {self.max_nyc_rows:,} rows.")
        else:
            header_cols = pd.read_csv(self.nyc_path, nrows=0).columns.tolist()
            selected_cols = self._get_nyc_columns_to_load(header_cols)
            csv_kwargs = {'usecols': selected_cols} if selected_cols else {}
            if self.max_nyc_rows and self.max_nyc_rows > 0:
                csv_kwargs['nrows'] = self.max_nyc_rows
            df = pd.read_csv(self.nyc_path, **csv_kwargs)

        # Standardise column names
        df.columns = df.columns.str.lower().str.strip()

        # Map common TLC column variants
        rename_map = {
            'tpep_pickup_datetime':  'pickup_datetime',
            'tpep_dropoff_datetime': 'dropoff_datetime',
            'lpep_pickup_datetime':  'pickup_datetime',
            'lpep_dropoff_datetime': 'dropoff_datetime',
            'pulocationid':          'pickup_location',
            'dolocationid':          'dropoff_location',
            'trip_distance':         'trip_distance_km',
            'passenger_count':       'num_passengers',
            'fare_amount':           'fare_rs',
            'ratecodeID':            'rate_type',
            'ratecodeid':            'rate_type',
            'payment_type':          'payment_type',
        }
        df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns}, inplace=True)

        # Remove rows with critical missing values
        critical = ['pickup_datetime', 'dropoff_datetime', 'pickup_location', 'dropoff_location']
        df.dropna(subset=[c for c in critical if c in df.columns], inplace=True)

        # Parse datetimes
        df['pickup_datetime']  = pd.to_datetime(df['pickup_datetime'],  errors='coerce')
        df['dropoff_datetime'] = pd.to_datetime(df['dropoff_datetime'], errors='coerce')
        df.dropna(subset=['pickup_datetime', 'dropoff_datetime'], inplace=True)

        # Derive carpool label: passenger_count > 1 → carpool proxy
        if 'num_passengers' in df.columns:
            df['num_passengers'] = pd.to_numeric(df['num_passengers'], errors='coerce').fillna(1)
            df['ride_type'] = df['num_passengers'].apply(lambda x: 'Carpool' if x > 1 else 'Private')
        else:
            df['num_passengers'] = 1
            df['ride_type'] = 'Private'

        # Miles → km
        if 'trip_distance_km' in df.columns:
            df['trip_distance_km'] = pd.to_numeric(df['trip_distance_km'], errors='coerce').fillna(0) * 1.60934

        # Numeric fare
        if 'fare_rs' in df.columns:
            df['fare_rs'] = pd.to_numeric(df['fare_rs'], errors='coerce').fillna(0)

        # Add source tag
        df['data_source'] = 'NYC_TLC'
        df['ride_id'] = ['NYC_' + str(i) for i in range(len(df))]

        print(f"  → Loaded {len(df):,} NYC TLC records.")
        self.nyc_df = df
        return df

    def load_eco_data(self) -> pd.DataFrame:
        """Load EcoMoveNet app dataset."""
        print("[DataLoader] Loading EcoMoveNet dataset...")
        df = pd.read_csv(self.eco_path)
        df.columns = df.columns.str.lower().str.strip()

        df['pickup_datetime']  = pd.to_datetime(df['pickup_datetime'],  errors='coerce')
        df['dropoff_datetime'] = pd.to_datetime(df['dropoff_datetime'], errors='coerce')
        df.dropna(subset=['pickup_datetime', 'dropoff_datetime'], inplace=True)

        numeric_cols = ['trip_distance_km', 'num_passengers', 'avg_carpoolers',
                        'fare_rs', 'carbon_saved_kg', 'carbon_emitted_kg', 'efficiency_score']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        df['data_source'] = 'EcoMoveNet'
        print(f"  → Loaded {len(df):,} EcoMoveNet records.")
        self.eco_df = df
        return df

    def get_combined(self) -> pd.DataFrame:
        """Return a harmonised union of both datasets."""
        common = ['ride_id', 'ride_type', 'pickup_datetime', 'dropoff_datetime',
                  'pickup_location', 'dropoff_location', 'trip_distance_km',
                  'num_passengers', 'fare_rs', 'payment_type', 'rate_type', 'data_source']

        frames = []
        for df in [self.nyc_df, self.eco_df]:
            if df is not None:
                cols = [c for c in common if c in df.columns]
                frames.append(df[cols])

        combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        return combined


# ─────────────────────────────────────────────
# STEP 3 & 4: Feature Engineering
# ─────────────────────────────────────────────

class FeatureEngineer:
    """Extract, normalise, and encode travel features."""

    def __init__(self):
        self.scaler  = StandardScaler()
        self.cat_maps = {}
        self.feature_cols = []

    def extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Step 3 – extract temporal and spatial features."""
        print("[FeatureEngineer] Extracting travel features...")
        df = df.copy()

        # ── Resolve column aliases before anything else ───────────────────────
        # Covers raw TLC files AND already-loaded frames passed in a second time
        _alias_map = {
            'trip_distance':         'trip_distance_km',
            'tripdistance':          'trip_distance_km',
            'distance':              'trip_distance_km',
            'passenger_count':       'num_passengers',
            'passengers':            'num_passengers',
            'fare_amount':           'fare_rs',
            'fare':                  'fare_rs',
            'total_amount':          'fare_rs',
            'tpep_pickup_datetime':  'pickup_datetime',
            'lpep_pickup_datetime':  'pickup_datetime',
            'tpep_dropoff_datetime': 'dropoff_datetime',
            'lpep_dropoff_datetime': 'dropoff_datetime',
            'pulocationid':          'pickup_location',
            'dolocationid':          'dropoff_location',
        }
        df.columns = df.columns.str.lower().str.strip()
        df.rename(columns={k: v for k, v in _alias_map.items()
                            if k in df.columns and v not in df.columns}, inplace=True)

        # ── Guarantee all required columns exist with safe defaults ───────────
        required_defaults = {
            'trip_distance_km': 0.0,
            'num_passengers':   1,
            'fare_rs':          0.0,
            'ride_type':        'Private',
            'pickup_location':  'Unknown',
            'dropoff_location': 'Unknown',
        }
        for col, default in required_defaults.items():
            if col not in df.columns:
                df[col] = default

        # ── Coerce numerics ───────────────────────────────────────────────────
        df['trip_distance_km'] = pd.to_numeric(df['trip_distance_km'], errors='coerce').fillna(0)
        df['num_passengers']   = pd.to_numeric(df['num_passengers'],   errors='coerce').fillna(1).clip(lower=1)
        df['fare_rs']          = pd.to_numeric(df['fare_rs'],          errors='coerce').fillna(0)

        # ── Parse datetimes safely ────────────────────────────────────────────
        df['pickup_datetime']  = pd.to_datetime(df['pickup_datetime'],  errors='coerce')
        df['dropoff_datetime'] = pd.to_datetime(df['dropoff_datetime'], errors='coerce')
        df.dropna(subset=['pickup_datetime', 'dropoff_datetime'], inplace=True)
        df = df.reset_index(drop=True)

        # ── Temporal features ─────────────────────────────────────────────────
        df['pickup_hour']    = df['pickup_datetime'].dt.hour
        df['pickup_dow']     = df['pickup_datetime'].dt.dayofweek
        df['pickup_month']   = df['pickup_datetime'].dt.month
        df['is_weekend']     = df['pickup_dow'].isin([5, 6]).astype(int)
        df['is_peak_hour']   = df['pickup_hour'].apply(
            lambda h: 1 if (7 <= h <= 9) or (17 <= h <= 20) else 0)

        # Duration
        df['trip_duration_min'] = (
            (df['dropoff_datetime'] - df['pickup_datetime']).dt.total_seconds() / 60
        ).clip(lower=0, upper=300)

        # Distance bins
        df['distance_bucket'] = pd.cut(
            df['trip_distance_km'],
            bins=[0, 2, 5, 10, 20, 1e9],
            labels=['very_short', 'short', 'medium', 'long', 'very_long'],
            right=True
        ).astype(str).replace('nan', 'very_short')

        # Passenger fill ratio
        if 'avg_carpoolers' in df.columns:
            df['fill_ratio'] = (df['num_passengers'] /
                                df['avg_carpoolers'].replace(0, np.nan)).fillna(1).clip(0, 1)
        else:
            df['fill_ratio'] = (df['num_passengers'] / 4).clip(0, 1)

        # Carbon benefit
        if 'carbon_saved_kg' in df.columns:
            df['carbon_benefit'] = pd.to_numeric(df['carbon_saved_kg'], errors='coerce').fillna(0)
        else:
            is_carpool = df['ride_type'].astype(str).str.lower() == 'carpool'
            df['carbon_benefit'] = np.where(
                is_carpool,
                df['trip_distance_km'] * 0.21 * 0.3 * (df['num_passengers'] - 1),
                0.0
            )

        print("  → Features extracted.")
        return df

    def encode_and_scale(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Step 4 – normalise numeric, encode categoricals."""
        print("[FeatureEngineer] Normalising and encoding features...")
        df = df.copy()

        numeric_feats = ['trip_distance_km', 'num_passengers', 'fare_rs',
                         'pickup_hour', 'pickup_dow', 'trip_duration_min',
                         'fill_ratio', 'carbon_benefit', 'is_peak_hour', 'is_weekend']
        numeric_feats = [c for c in numeric_feats if c in df.columns]

        cat_feats = ['payment_type', 'rate_type', 'distance_bucket']
        cat_feats = [c for c in cat_feats if c in df.columns]

        # Fill numeric NaNs
        for col in numeric_feats:
            df[col] = df[col].fillna(df[col].median())

        # Encode categoricals
        for col in cat_feats:
            col_s = df[col].astype(str)
            if fit:
                uniq = sorted(col_s.dropna().unique().tolist())
                self.cat_maps[col] = {v: i for i, v in enumerate(uniq)}

            cmap = self.cat_maps.get(col, {})
            df[col + '_enc'] = col_s.map(cmap).fillna(-1).astype(int)

        # Scale numeric
        if fit:
            df[numeric_feats] = self.scaler.fit_transform(df[numeric_feats])
        else:
            df[numeric_feats] = self.scaler.transform(df[numeric_feats])

        self.feature_cols = numeric_feats + [c + '_enc' for c in cat_feats]
        return df


# ─────────────────────────────────────────────
# STEP 5 & 6: Spatio-Temporal Analysis & Clustering
# ─────────────────────────────────────────────

class SpatioTemporalAnalyzer:
    """Identify peak hours, travel zones, and cluster frequent routes."""

    def __init__(self, n_clusters: int = 8):
        self.n_clusters = n_clusters
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)

    def identify_peak_hours(self, df: pd.DataFrame) -> dict:
        """Step 5 – find peak travel hours and zones."""
        print("[SpatioTemporal] Identifying peak hours...")
        if 'pickup_hour' not in df.columns:
            df['pickup_hour'] = pd.to_datetime(df['pickup_datetime']).dt.hour

        hour_counts = df.groupby('pickup_hour').size()
        threshold   = hour_counts.quantile(0.75)
        peak_hours  = hour_counts[hour_counts >= threshold].index.tolist()

        results = {
            'peak_hours':      peak_hours,
            'hourly_demand':   hour_counts.to_dict(),
            'peak_threshold':  float(threshold)
        }
        print(f"  → Peak hours: {peak_hours}")
        return results

    def cluster_routes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Step 6 – cluster frequent travel routes."""
        print("[SpatioTemporal] Clustering travel routes...")
        df = df.copy()

        # Encode pickup/dropoff as numeric
        for col in ['pickup_location', 'dropoff_location']:
            df[col + '_code'] = LabelEncoder().fit_transform(df[col].astype(str))

        feats = df[['pickup_location_code', 'dropoff_location_code']].copy()
        for col in ['pickup_hour', 'trip_distance_km']:
            if col in df.columns:
                feats[col] = df[col].values

        feats = feats.fillna(0)
        k = min(self.n_clusters, len(feats) - 1)
        labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(feats)
        df['route_cluster'] = labels
        print(f"  → Identified {k} route clusters.")
        return df


# ─────────────────────────────────────────────
# STEP 7–9: Carpool Matching
# ─────────────────────────────────────────────

class CarpoolMatcher:
    """Match carpool candidates based on route similarity and time overlap."""

    def __init__(self, time_window_min: int = 15, similarity_threshold: float = 0.7):
        self.time_window   = time_window_min
        self.sim_threshold = similarity_threshold

    def compute_route_similarity(self, t1: pd.Series, t2: pd.Series) -> float:
        """Jaccard-style route similarity."""
        same_pickup  = t1['pickup_location'] == t2['pickup_location']
        same_dropoff = t1['dropoff_location'] == t2['dropoff_location']
        if same_pickup and same_dropoff:
            return 1.0
        elif same_pickup or same_dropoff:
            return 0.6
        return 0.0

    def time_overlap(self, t1: pd.Series, t2: pd.Series) -> bool:
        """Check whether two trips fall within the time window."""
        try:
            delta = abs((t1['pickup_datetime'] - t2['pickup_datetime']).total_seconds() / 60)
            return delta <= self.time_window
        except Exception:
            return False

    def find_carpool_candidates(self, df: pd.DataFrame, sample_size: int = 2000) -> pd.DataFrame:
        """Steps 7-9 – find and rank carpool pairs."""
        print("[CarpoolMatcher] Finding carpool candidates...")
        df = df.sample(min(sample_size, len(df)), random_state=42).reset_index(drop=True)
        candidates = []

        for i in range(len(df)):
            for j in range(i + 1, min(i + 50, len(df))):
                t1, t2 = df.iloc[i], df.iloc[j]
                sim  = self.compute_route_similarity(t1, t2)
                if sim >= self.sim_threshold and self.time_overlap(t1, t2):
                    score = sim * 0.6 + (1 - min(
                        abs((t1['pickup_datetime'] - t2['pickup_datetime']).total_seconds() / 60)
                        / self.time_window, 1)) * 0.4
                    candidates.append({
                        'ride_id_1': t1.get('ride_id', i),
                        'ride_id_2': t2.get('ride_id', j),
                        'route_similarity': sim,
                        'match_score': score,
                        'shared_pickup': t1['pickup_location'] == t2['pickup_location'],
                        'shared_dropoff': t1['dropoff_location'] == t2['dropoff_location'],
                    })

        result = pd.DataFrame(candidates).sort_values('match_score', ascending=False)
        print(f"  → {len(result):,} carpool candidates found.")
        return result

    def generate_recommendations(self, df: pd.DataFrame, candidates: pd.DataFrame,
                                  top_n: int = 10) -> pd.DataFrame:
        """Step 16 – generate personalised carpool recommendations."""
        top = candidates.head(top_n).copy()
        top['recommendation'] = top.apply(
            lambda r: f"Match rides {r['ride_id_1']} ↔ {r['ride_id_2']} "
                      f"(score={r['match_score']:.2f})", axis=1)
        return top


# ─────────────────────────────────────────────
# STEP 10 & 11: Neural Prediction Model
# ─────────────────────────────────────────────

class RideSharingPredictor:
    """Train and evaluate ride-sharing probability model."""

    def __init__(self):
        self.model = MLPClassifier(
            hidden_layer_sizes=(128, 64, 32),
            activation='relu',
            max_iter=500,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1
        )
        self.is_fitted = False

    def prepare_target(self, df: pd.DataFrame) -> pd.Series:
        return (df['ride_type'].astype(str).str.lower() == 'carpool').astype(int)

    def train(self, X: pd.DataFrame, y: pd.Series):
        """Step 10 – train neural prediction model."""
        print("[RideSharingPredictor] Training neural model...")
        X = X.fillna(0)
        self.model.fit(X, y)
        self.is_fitted = True
        print("  → Model trained.")

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Step 11 – predict ride-sharing probability."""
        X = X.fillna(0)
        return self.model.predict_proba(X)[:, 1]

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X = X.fillna(0)
        return self.model.predict(X)

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
        """Step 17 – evaluate with full metrics."""
        print("[RideSharingPredictor] Evaluating model...")
        X_test = X_test.fillna(0)
        y_pred      = self.predict(X_test)
        y_prob      = self.predict_proba(X_test)
        avg         = 'binary' if len(np.unique(y_test)) == 2 else 'macro'

        metrics = {
            'accuracy':  accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average=avg, zero_division=0),
            'recall':    recall_score(y_test, y_pred, average=avg, zero_division=0),
            'f1_score':  f1_score(y_test, y_pred, average=avg, zero_division=0),
        }
        try:
            metrics['auc_roc'] = roc_auc_score(y_test, y_prob)
        except Exception:
            metrics['auc_roc'] = None

        for k, v in metrics.items():
            if v is not None:
                print(f"  {k:12s}: {v:.4f}")
        return metrics


class LSTMTravelBehaviorModel:
    """Sequence-based travel behavior model using an LSTM network."""

    def __init__(self, seq_len: int = 6, epochs: int = 20, batch_size: int = 16):
        self.seq_len = seq_len
        self.epochs = epochs
        self.batch_size = batch_size
        self.scaler = StandardScaler()
        self.model = None
        self.device = 'cpu'

    def _build_sequences(self, df: pd.DataFrame, feature_cols: List[str], y: pd.Series):
        work = df.copy().sort_values('pickup_datetime')
        ordered_idx = work.index
        X_num = work[feature_cols].fillna(0).astype(float)
        X_scaled = self.scaler.fit_transform(X_num)
        y_seq = y.loc[ordered_idx].values

        X_out, y_out = [], []
        for i in range(self.seq_len, len(work)):
            X_out.append(X_scaled[i - self.seq_len:i, :])
            y_out.append(y_seq[i])

        if not X_out:
            return None, None
        return np.array(X_out), np.array(y_out)

    def train_evaluate(self, df: pd.DataFrame, feature_cols: List[str], y: pd.Series) -> dict:
        print("[LSTMTravelBehaviorModel] Training LSTM travel behavior model...")
        try:
            torch = importlib.import_module('torch')
            nn = importlib.import_module('torch.nn')
            optim = importlib.import_module('torch.optim')
        except Exception as e:
            print(f"  → Skipped (PyTorch unavailable): {e}")
            return {'status': 'skipped', 'reason': 'pytorch_not_available'}

        if 'pickup_datetime' not in df.columns:
            return {'status': 'skipped', 'reason': 'pickup_datetime_missing'}

        X_seq, y_seq = self._build_sequences(df, feature_cols, y)
        if X_seq is None or len(X_seq) < 80:
            return {'status': 'skipped', 'reason': 'insufficient_sequence_data'}

        split = int(len(X_seq) * 0.8)
        X_train, X_test = X_seq[:split], X_seq[split:]
        y_train, y_test = y_seq[:split], y_seq[split:]

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        class LSTMNet(nn.Module):
            def __init__(self, input_dim: int):
                super().__init__()
                self.lstm = nn.LSTM(input_size=input_dim, hidden_size=64, batch_first=True)
                self.dropout = nn.Dropout(0.2)
                self.fc1 = nn.Linear(64, 32)
                self.relu = nn.ReLU()
                self.fc2 = nn.Linear(32, 1)

            def forward(self, x):
                out, _ = self.lstm(x)
                out = out[:, -1, :]
                out = self.dropout(out)
                out = self.relu(self.fc1(out))
                return self.fc2(out)

        self.model = LSTMNet(X_train.shape[2]).to(self.device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=1e-3)

        X_train_t = torch.tensor(X_train, dtype=torch.float32).to(self.device)
        y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(self.device)
        X_test_t = torch.tensor(X_test, dtype=torch.float32).to(self.device)

        self.model.train()
        for _ in range(self.epochs):
            idx = torch.randperm(X_train_t.shape[0], device=self.device)
            for i in range(0, X_train_t.shape[0], self.batch_size):
                batch_idx = idx[i:i + self.batch_size]
                xb = X_train_t[batch_idx]
                yb = y_train_t[batch_idx]

                optimizer.zero_grad()
                logits = self.model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()

        self.model.eval()
        with torch.no_grad():
            logits = self.model(X_test_t)
            y_prob = torch.sigmoid(logits).cpu().numpy().ravel()
        y_pred = (y_prob >= 0.5).astype(int)

        metrics = {
            'status': 'trained',
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1_score': f1_score(y_test, y_pred, zero_division=0),
        }
        try:
            metrics['auc_roc'] = roc_auc_score(y_test, y_prob)
        except Exception:
            metrics['auc_roc'] = None

        print(f"  → LSTM accuracy: {metrics['accuracy']:.4f}")
        return metrics


class TransformerSentimentClassifier:
    """HuggingFace pipeline wrapper for transcript sentiment classification."""

    def __init__(self, model_name: str, display_name: str):
        self.model_name = model_name
        self.display_name = display_name
        self._pipe = None

    def _get_pipeline(self):
        if self._pipe is not None:
            return self._pipe
        pipeline = importlib.import_module('transformers').pipeline
        self._pipe = pipeline(
            task='text-classification',
            model=self.model_name,
            truncation=True,
            max_length=256
        )
        return self._pipe

    @staticmethod
    def _normalise_sentiment(label: str) -> str:
        up = str(label).upper()
        if 'POS' in up or up in {'LABEL_1', '4 STARS', '5 STARS'}:
            return 'positive'
        if 'NEG' in up or up in {'LABEL_0', '1 STAR', '2 STARS'}:
            return 'negative'
        if '3 STAR' in up or 'NEUTRAL' in up:
            return 'neutral'
        return 'neutral'

    def analyse(self, texts: List[str]) -> pd.DataFrame:
        print(f"[TransformerSentimentClassifier] Running {self.display_name} sentiment classifier...")
        try:
            pipe = self._get_pipeline()
        except Exception as e:
            print(f"  → Skipped ({self.display_name} unavailable): {e}")
            return pd.DataFrame(columns=['text', 'label', 'score', 'sentiment'])

        rows = []
        for text in texts:
            pred = pipe(str(text))[0]
            sentiment = self._normalise_sentiment(pred.get('label', ''))
            rows.append({
                'text': str(text)[:160],
                'label': pred.get('label'),
                'score': float(pred.get('score', 0.0)),
                'sentiment': sentiment,
            })
        return pd.DataFrame(rows)


# ─────────────────────────────────────────────
# STEP 12 & 13: Gamification Engine
# ─────────────────────────────────────────────

class GamificationEngine:
    """Compute engagement scores and personalise incentives."""

    INCENTIVE_TIERS = [
        (0.8,  "🥇 Platinum EcoRider – 20 % fare discount + premium badge"),
        (0.6,  "🥈 Gold EcoRider   – 10 % fare discount + leaderboard feature"),
        (0.4,  "🥉 Silver EcoRider – 5 % fare discount + streak badge"),
        (0.0,  "🌱 Green Starter   – Welcome bonus + first carpool coupon"),
    ]

    def compute_engagement_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """Step 12 – derive user engagement score from gamification logs."""
        df = df.copy()
        carpool_flag = (df['ride_type'].astype(str).str.lower() == 'carpool').astype(float)

        # Frequency proxy (normalised trip count per user — approximate over dataset)
        df['engagement_score'] = (
            carpool_flag * 0.5 +
            df.get('fill_ratio', pd.Series(np.zeros(len(df)), index=df.index)).fillna(0) * 0.3 +
            df.get('efficiency_score',
                   pd.Series(np.zeros(len(df)), index=df.index)).fillna(0).clip(0, 1) * 0.2
        )
        return df

    def personalise_incentives(self, df: pd.DataFrame) -> pd.DataFrame:
        """Step 13 – personalise incentives based on engagement history."""
        df = df.copy()

        def get_incentive(score):
            for threshold, label in self.INCENTIVE_TIERS:
                if score >= threshold:
                    return label
            return self.INCENTIVE_TIERS[-1][1]

        df['incentive_tier'] = df['engagement_score'].apply(get_incentive)
        return df


# ─────────────────────────────────────────────
# STEP 14: EcoScore Computation
# ─────────────────────────────────────────────

class EcoScoreCalculator:
    """Compute EcoScore from shared rides and emission reduction."""

    # Emission factor: kg CO₂ per km for a standard car
    EMISSION_FACTOR = 0.21

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """Step 14 – compute EcoScore."""
        print("[EcoScore] Computing EcoScore...")
        df = df.copy()

        is_carpool = (df['ride_type'].astype(str).str.lower() == 'carpool')

        if 'carbon_saved_kg' in df.columns:
            df['carbon_saved'] = df['carbon_saved_kg'].fillna(0)
        else:
            passengers = df['num_passengers'].fillna(1).clip(lower=1)
            distance   = df['trip_distance_km'].fillna(0)
            df['carbon_saved'] = np.where(
                is_carpool,
                distance * self.EMISSION_FACTOR * (passengers - 1) / passengers,
                0.0
            )

        # Normalise EcoScore 0–100
        max_saved = df['carbon_saved'].max()
        df['eco_score'] = (df['carbon_saved'] / max_saved * 100).clip(0, 100) if max_saved > 0 else 0.0

        print(f"  → Mean EcoScore: {df['eco_score'].mean():.2f}")
        return df


# ─────────────────────────────────────────────
# STEP 15: Sentiment & Behavioural Change Analysis
# ─────────────────────────────────────────────

class BehavioralAnalyzer:
    """Sentiment analysis on user feedback to estimate behavioural change."""

    def load_transcript_lines(self, transcript_path: str, min_len: int = 8) -> List[str]:
        """Load interview transcript utterances from a .txt file."""
        p = Path(transcript_path)
        if not p.exists():
            return []

        text = p.read_text(encoding='utf-8', errors='ignore')
        lines = []
        for raw in text.splitlines():
            line = raw.strip()
            if not line:
                continue
            if ':' in line and len(line.split(':', 1)[0]) <= 30:
                line = line.split(':', 1)[1].strip()
            if len(line) >= min_len:
                lines.append(line)
        return lines

    def analyse_sentiment(self, feedback_texts: list) -> pd.DataFrame:
        """Step 15 – apply sentiment analysis on feedback T."""
        print("[BehavioralAnalyzer] Running sentiment analysis...")
        results = []
        for text in feedback_texts:
            blob = TextBlob(str(text))
            polarity    = blob.sentiment.polarity      # -1 to 1
            subjectivity = blob.sentiment.subjectivity  # 0 to 1
            label = ('positive' if polarity > 0.1 else
                     'negative' if polarity < -0.1 else 'neutral')
            results.append({
                'text':         text[:80],
                'polarity':     round(polarity, 3),
                'subjectivity': round(subjectivity, 3),
                'sentiment':    label,
                'behavior_change_signal': 1 if polarity > 0.2 else 0
            })
        df = pd.DataFrame(results)
        print(f"  → Sentiment: {df['sentiment'].value_counts().to_dict()}")
        return df

    def generate_sample_feedback(self, n: int = 100) -> list:
        """Generate synthetic feedback corpus for demonstration."""
        samples = [
            "I love carpooling, it saves me money and reduces traffic!",
            "The app is great but matching takes too long.",
            "Carpooling has been a game changer for my commute.",
            "I switched from driving alone to sharing rides daily.",
            "Sometimes the route suggestions are off.",
            "Fantastic initiative for the environment.",
            "The EcoScore motivates me to carpool more.",
            "I wish there were more carpool options in my area.",
            "Saved a lot of carbon this month, feeling great!",
            "The gamification keeps me engaged every week.",
        ]
        return (samples * (n // len(samples) + 1))[:n]


# ─────────────────────────────────────────────
# ABLATION STUDY
# ─────────────────────────────────────────────

class AblationStudy:
    """Systematic ablation over model component variants."""

    VARIANTS = {
        "Baseline (Logistic Regression)": {
            "use_spatial":   False,
            "use_temporal":  False,
            "use_behavior":  False,
            "use_incentive": False,
            "model":         LogisticRegression(max_iter=500, random_state=42)
        },
        "Temporal Only": {
            "use_spatial":   False,
            "use_temporal":  True,
            "use_behavior":  False,
            "use_incentive": False,
            "model":         RandomForestClassifier(n_estimators=100, random_state=42)
        },
        "Spatial + Temporal": {
            "use_spatial":   True,
            "use_temporal":  True,
            "use_behavior":  False,
            "use_incentive": False,
            "model":         GradientBoostingClassifier(n_estimators=100, random_state=42)
        },
        "Spatial + Temporal + Behavior": {
            "use_spatial":   True,
            "use_temporal":  True,
            "use_behavior":  True,
            "use_incentive": False,
            "model":         GradientBoostingClassifier(n_estimators=150, random_state=42)
        },
        "Hybrid Spatio-Temporal + Behavior + Incentive Model": {
            "use_spatial":   True,
            "use_temporal":  True,
            "use_behavior":  True,
            "use_incentive": True,
            "model":         MLPClassifier(
                                 hidden_layer_sizes=(128, 64, 32),
                                 activation='relu', max_iter=500,
                                 random_state=42, early_stopping=True)
        },
    }

    def run(self, df: pd.DataFrame, feature_cols: list, target: pd.Series) -> pd.DataFrame:
        """Execute ablation study across all model variants."""
        print("\n[AblationStudy] Running ablation study...")
        base_temporal  = [c for c in feature_cols if any(k in c for k in ['hour', 'dow', 'duration', 'weekend', 'peak'])]
        base_spatial   = [c for c in feature_cols if any(k in c for k in ['distance', 'location', 'cluster', 'bucket'])]
        base_behavior  = [c for c in feature_cols if any(k in c for k in ['fill', 'efficiency', 'carbon', 'engagement'])]
        base_incentive = [c for c in feature_cols if any(k in c for k in ['fare', 'payment', 'rate'])]
        fallback       = [c for c in feature_cols if c not in base_temporal + base_spatial + base_behavior + base_incentive]

        results = []
        for name, cfg in self.VARIANTS.items():
            feat_set = list(fallback)
            if cfg['use_temporal']:  feat_set += base_temporal
            if cfg['use_spatial']:   feat_set += base_spatial
            if cfg['use_behavior']:  feat_set += base_behavior
            if cfg['use_incentive']: feat_set += base_incentive

            feat_set = [c for c in dict.fromkeys(feat_set) if c in df.columns]
            if not feat_set:
                feat_set = [c for c in feature_cols if c in df.columns]

            X = df[feat_set].fillna(0)
            y = target

            if len(X) < 50 or y.nunique() < 2:
                continue

            X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2,
                                                        random_state=42, stratify=y)
            model = cfg['model']
            model.fit(X_tr, y_tr)
            y_pred = model.predict(X_te)
            y_prob = model.predict_proba(X_te)[:, 1] if hasattr(model, 'predict_proba') else None

            avg = 'binary'
            acc = accuracy_score(y_te, y_pred)
            f1  = f1_score(y_te, y_pred, average=avg, zero_division=0)
            pre = precision_score(y_te, y_pred, average=avg, zero_division=0)
            rec = recall_score(y_te, y_pred, average=avg, zero_division=0)
            auc = roc_auc_score(y_te, y_prob) if y_prob is not None else None

            # RMSE for regression proxy (predict carpool probability)
            rmse = np.sqrt(np.mean((y_te.values - (y_prob if y_prob is not None else y_pred.astype(float)))**2))

            components = "+".join(filter(None, [
                "Spatial"   if cfg['use_spatial']   else "",
                "Temporal"  if cfg['use_temporal']  else "",
                "Behavior"  if cfg['use_behavior']  else "",
                "Incentive" if cfg['use_incentive'] else "",
                "Baseline"  if not any([cfg['use_spatial'], cfg['use_temporal'],
                                        cfg['use_behavior'], cfg['use_incentive']]) else ""
            ]))

            results.append({
                'Model Variant':     name,
                'Components Used':   components,
                'Accuracy (%)':      round(acc * 100, 2),
                'Precision (%)':     round(pre * 100, 2),
                'Recall (%)':        round(rec * 100, 2),
                'F1-score (%)':      round(f1 * 100, 2),
                'AUC-ROC':           round(auc, 4) if auc else 'N/A',
                'RMSE':              round(rmse, 4),
            })
            print(f"  [{name}] Acc={acc:.3f}  F1={f1:.3f}  RMSE={rmse:.4f}")

        return pd.DataFrame(results)


# ─────────────────────────────────────────────
# COMPARATIVE ANALYSIS: NYC TLC vs EcoMoveNet
# ─────────────────────────────────────────────

class ComparativeAnalyzer:
    """Head-to-head dataset comparison across key sustainability metrics."""

    def compare(self, nyc_df: pd.DataFrame, eco_df: pd.DataFrame) -> dict:
        """Generate comparative statistics between both datasets."""
        print("\n[ComparativeAnalyzer] Running comparative analysis...")

        def safe_mean(df, col, filter_col=None, filter_val=None):
            if df is None or col not in df.columns:
                return None

            series = df[col]
            if filter_col and filter_col in df.columns:
                mask = df[filter_col].astype(str).str.lower() == str(filter_val).lower()
                series = series[mask]

            vals = pd.to_numeric(series, errors='coerce').dropna()
            if vals.empty:
                return None
            return round(vals.mean(), 4)

        def safe_rate(df, col, target_val):
            if df is None or col not in df.columns:
                return None
            return round((df[col].astype(str).str.lower() == str(target_val).lower()).mean() * 100, 2)

        def count_rows(df):
            return len(df) if df is not None else 0

        def has_col(df, col):
            return df is not None and col in df.columns

        def infer_carpool_col(df):
            if df is None:
                return None
            if 'carbon_saved_kg' in df.columns:
                return 'carbon_saved_kg'
            if 'carbon_saved' in df.columns:
                return 'carbon_saved'
            return None

        nyc_carpool_col = infer_carpool_col(nyc_df)
        eco_carpool_col = infer_carpool_col(eco_df)

        comparison = {
            'dataset_size': {
                'NYC TLC':    count_rows(nyc_df),
                'EcoMoveNet': count_rows(eco_df),
            },
            'carpool_rate_pct': {
                'NYC TLC':    safe_rate(nyc_df, 'ride_type', 'carpool'),
                'EcoMoveNet': safe_rate(eco_df, 'ride_type', 'carpool'),
            },
            'avg_trip_distance_km': {
                'NYC TLC':    safe_mean(nyc_df, 'trip_distance_km'),
                'EcoMoveNet': safe_mean(eco_df, 'trip_distance_km'),
            },
            'avg_passengers': {
                'NYC TLC':    safe_mean(nyc_df, 'num_passengers'),
                'EcoMoveNet': safe_mean(eco_df, 'num_passengers'),
            },
            'avg_fare': {
                'NYC TLC':    safe_mean(nyc_df, 'fare_rs'),
                'EcoMoveNet': safe_mean(eco_df, 'fare_rs'),
            },
            'avg_carbon_saved_carpool': {
                'NYC TLC':    safe_mean(nyc_df, nyc_carpool_col, filter_col='ride_type', filter_val='carpool') if nyc_carpool_col else None,
                'EcoMoveNet': safe_mean(eco_df, eco_carpool_col, filter_col='ride_type', filter_val='carpool') if eco_carpool_col else None,
            },
            'eco_score_mean': {
                'NYC TLC':    safe_mean(nyc_df, 'eco_score') if has_col(nyc_df, 'eco_score') else None,
                'EcoMoveNet': safe_mean(eco_df, 'eco_score') if has_col(eco_df, 'eco_score') else None,
            }
        }
        return comparison


# ─────────────────────────────────────────────
# MAIN ORCHESTRATOR
# ─────────────────────────────────────────────

class EcoMoveNetPipeline:
    """Full EcoMoveNet pipeline orchestrator."""

    def __init__(self, nyc_path: str, eco_path: str, transcript_path: str = None,
                 max_nyc_rows: int = 300000):
        self.loader    = DataLoader(nyc_path, eco_path, max_nyc_rows=max_nyc_rows)
        self.engineer  = FeatureEngineer()
        self.spatio    = SpatioTemporalAnalyzer()
        self.matcher   = CarpoolMatcher()
        self.predictor = RideSharingPredictor()
        self.lstm_model = LSTMTravelBehaviorModel()
        self.gamify    = GamificationEngine()
        self.eco_score = EcoScoreCalculator()
        self.behavior  = BehavioralAnalyzer()
        self.bert_classifier = TransformerSentimentClassifier(
            model_name='nlptown/bert-base-multilingual-uncased-sentiment',
            display_name='BERT'
        )
        self.albert_classifier = TransformerSentimentClassifier(
            model_name='textattack/albert-base-v2-imdb',
            display_name='ALBERT'
        )
        self.ablation  = AblationStudy()
        self.compare   = ComparativeAnalyzer()
        self.transcript_path = transcript_path

    def run(self) -> dict:
        print("=" * 65)
        print("   EcoMoveNet Pipeline – Starting")
        print("=" * 65)

        # 1–2: Load
        nyc_df = self.loader.load_nyc_data()
        eco_df = self.loader.load_eco_data()

        # Work on EcoMoveNet as primary; NYC as reference
        df_raw = eco_df.copy()

        # 3–4: Feature engineering (raw view for analytics + model-specific processed splits)
        df_raw = self.engineer.extract_features(df_raw)

        # 5–6: Spatio-temporal
        peak_info = self.spatio.identify_peak_hours(df_raw)
        df_raw = self.spatio.cluster_routes(df_raw)

        # 7–9: Carpool matching
        candidates = self.matcher.find_carpool_candidates(df_raw)
        recommendations = self.matcher.generate_recommendations(df_raw, candidates)

        # 10–11: Prediction model
        y = self.predictor.prepare_target(df_raw)
        idx_train, idx_test = train_test_split(
            df_raw.index, test_size=0.2, random_state=42, stratify=y)

        train_raw = df_raw.loc[idx_train].copy()
        test_raw  = df_raw.loc[idx_test].copy()

        train_proc = self.engineer.encode_and_scale(train_raw, fit=True)
        test_proc  = self.engineer.encode_and_scale(test_raw, fit=False)

        feature_cols = [c for c in self.engineer.feature_cols if c in train_proc.columns]

        # Remove direct/derived label leaks for realistic ride-type prediction.
        leaky_features = {
            'num_passengers',
            'fill_ratio',
            'carbon_benefit',
            'rate_type_enc',
            'fare_rs',
        }
        feature_cols = [c for c in feature_cols if c not in leaky_features]

        X_train = train_proc[feature_cols].fillna(0)
        X_test  = test_proc[feature_cols].fillna(0)
        y_train = y.loc[idx_train]
        y_test  = y.loc[idx_test]

        self.predictor.train(X_train, y_train)
        metrics = self.predictor.evaluate(X_test, y_test)

        full_proc = self.engineer.encode_and_scale(df_raw.copy(), fit=False)
        lstm_metrics = self.lstm_model.train_evaluate(full_proc, feature_cols, y)
        X_full = full_proc[feature_cols].fillna(0)
        df_raw['ridesharing_prob'] = self.predictor.predict_proba(X_full)

        # 12–13: Gamification
        df_raw = self.gamify.compute_engagement_score(df_raw)
        df_raw = self.gamify.personalise_incentives(df_raw)

        # 14: EcoScore
        df_raw = self.eco_score.compute(df_raw)

        # NYC uses a separate FeatureEngineer (independent scaler — different units/ranges)
        nyc_engineer = FeatureEngineer()
        nyc_featured = nyc_engineer.extract_features(nyc_df)
        nyc_df = self.eco_score.compute(nyc_featured)

        # 15: Behavioral analysis from transcripts
        transcript_lines = []
        if self.transcript_path:
            transcript_lines = self.behavior.load_transcript_lines(self.transcript_path)
        if not transcript_lines:
            transcript_lines = self.behavior.generate_sample_feedback(200)

        sentiment_df = self.behavior.analyse_sentiment(transcript_lines)
        bert_sentiment_df = self.bert_classifier.analyse(transcript_lines)
        albert_sentiment_df = self.albert_classifier.analyse(transcript_lines)

        # Ablation study
        ablation_df = full_proc.copy()
        for c in ['ride_type', 'pickup_datetime', 'dropoff_datetime', 'pickup_location', 'dropoff_location']:
            if c in df_raw.columns and c not in ablation_df.columns:
                ablation_df[c] = df_raw[c]
        ablation_results = self.ablation.run(ablation_df, feature_cols, y)

        # Comparative analysis
        comparison = self.compare.compare(nyc_df, df_raw)

        print("\n" + "=" * 65)
        print("   EcoMoveNet Pipeline – Complete")
        print("=" * 65)

        return {
            'eco_df':           df_raw,
            'nyc_df':           nyc_df,
            'peak_info':        peak_info,
            'carpool_candidates': candidates,
            'recommendations':  recommendations,
            'metrics':          metrics,
            'lstm_metrics':     lstm_metrics,
            'sentiment_df':     sentiment_df,
            'bert_sentiment_df': bert_sentiment_df,
            'albert_sentiment_df': albert_sentiment_df,
            'ablation_results': ablation_results,
            'comparison':       comparison,
        }