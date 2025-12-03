# ...existing code...
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def assign_budget_tertiles(prices):
    """
    Assign budget category based on price tertiles (for small datasets -> stable groups).
    Returns array of labels: 'Low Budget', 'Mid-Range', 'High-End'
    """
    prices = np.asarray(prices, dtype=float)
    if len(prices) == 0:
        return np.array([], dtype=object)

    # If all prices equal, assign Mid-Range
    if np.allclose(prices, prices[0]):
        return np.array(['Mid-Range'] * len(prices))

    # compute 33rd and 66th percentiles
    p33, p66 = np.percentile(prices, [33, 66])
    labels = []
    for p in prices:
        if p <= p33:
            labels.append('Low Budget')
        elif p <= p66:
            labels.append('Mid-Range')
        else:
            labels.append('High-End')
    return np.array(labels)

def map_clusters_to_budget_by_price(prices, cluster_labels):
    """
    Map numeric cluster labels to budget categories by sorting cluster average prices.
    Returns mapping dict cluster_id -> budget_label and array of budget labels per sample.
    """
    df = pd.DataFrame({'price': prices, 'cluster': cluster_labels})
    cluster_means = df.groupby('cluster')['price'].mean().sort_values()
    ordered_clusters = list(cluster_means.index)
    budget_order = ['Low Budget', 'Mid-Range', 'High-End']
    # If more than 3 clusters, map extremes accordingly: smallest->Low, largest->High, middle->Mid
    mapping = {}
    for i, c in enumerate(ordered_clusters):
        if i == 0:
            mapping[c] = 'Low Budget'
        elif i == len(ordered_clusters) - 1:
            mapping[c] = 'High-End'
        else:
            mapping[c] = 'Mid-Range'
    budget_labels = df['cluster'].map(mapping).values
    return mapping, budget_labels

def train_kmeans(features, prices, enforce_tertiles=False, prefer_k=3):
    """
    Train KMeans or assign budget by tertiles depending on data size or enforce_tertiles.
    Args:
      features: numpy array (n_samples, n_features) - scaled features (price, lat, lon)
      prices: raw price array (n_samples,)
      enforce_tertiles: if True, skip KMeans and use tertiles
      prefer_k: number of clusters to try for KMeans (default 3)
    Returns:
      dict with keys:
        - method: 'tertiles' or 'kmeans'
        - labels: budget labels per sample (Low/Mid/High)
        - metrics: dict (silhouette, davies_bouldin, calinski_harabasz or None if not computed)
        - extra: dict (kmeans_model if used, cluster_mapping if used)
    """
    n = len(prices)
    result = {'method': None, 'labels': None, 'metrics': {}, 'extra': {}}

    # Use tertiles for small datasets or if explicitly requested
    if enforce_tertiles or n < 30:
        logger.info("Using tertiles for budget categorization (small dataset or enforced).")
        labels = assign_budget_tertiles(prices)
        result.update({'method': 'tertiles', 'labels': labels})
        result['metrics'] = {'n_samples': n}
        return result

    # Otherwise run KMeans with prefer_k clusters (fallback to 3)
    k = max(2, int(prefer_k))
    try:
        km = KMeans(n_clusters=k, init='k-means++', n_init=10, max_iter=300, random_state=42)
        cluster_ids = km.fit_predict(features)
        # compute metrics if valid
        metrics = {}
        try:
            if len(set(cluster_ids)) > 1 and len(cluster_ids) > k:
                metrics['silhouette'] = float(silhouette_score(features, cluster_ids))
            else:
                metrics['silhouette'] = None
        except Exception:
            metrics['silhouette'] = None
        try:
            metrics['davies_bouldin'] = float(davies_bouldin_score(features, cluster_ids)) if len(set(cluster_ids)) > 1 else None
        except Exception:
            metrics['davies_bouldin'] = None
        try:
            metrics['calinski_harabasz'] = float(calinski_harabasz_score(features, cluster_ids)) if len(set(cluster_ids)) > 1 else None
        except Exception:
            metrics['calinski_harabasz'] = None

        # Map numeric clusters to budget categories by average price
        mapping, budget_labels = map_clusters_to_budget_by_price(prices, cluster_ids)

        result.update({
            'method': 'kmeans',
            'labels': budget_labels,
            'metrics': metrics,
            'extra': {
                'kmeans_model': km,
                'cluster_mapping': mapping
            }
        })
        return result
    except Exception as e:
        logger.exception("KMeans training failed, falling back to tertiles: %s", e)
        labels = assign_budget_tertiles(prices)
        result.update({'method': 'tertiles', 'labels': labels})
        result['metrics'] = {'n_samples': n, 'error': str(e)}
        return result

def summarize_budget_groups(prices, labels):
    """
    Summarize per-budget-group counts and average prices.
    Returns dict: { 'Low Budget': {count, avg_price, min,max}, ... }
    """
    df = pd.DataFrame({'price': prices, 'budget': labels})
    summary = {}
    for b in ['Low Budget', 'Mid-Range', 'High-End']:
        group = df[df['budget'] == b]
        if len(group) == 0:
            summary[b] = {'count': 0, 'avg_price': None, 'min_price': None, 'max_price': None}
        else:
            summary[b] = {
                'count': int(len(group)),
                'avg_price': float(group['price'].mean()),
                'min_price': float(group['price'].min()),
                'max_price': float(group['price'].max())
            }
    return summary

def knn_recommend(features, original_df, query_point_scaled, k=5):
    """
    features: scaled feature matrix (n_samples, n_features)
    original_df: DataFrame with at least ['price','latitude','longitude'] for output
    query_point_scaled: 1D array scaled like features
    Returns: list of recommendations with distance and confidence and budget label (by price tertiles)
    """
    n = features.shape[0]
    k = min(k, n)
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(features)
    distances, indices = nbrs.kneighbors([query_point_scaled])
    distances = distances[0]
    indices = indices[0]

    # normalize distances to confidence [0,1], smaller dist -> higher confidence
    maxd = distances.max() if distances.size > 0 else 0.0
    eps = 1e-9
    recs = []
    prices = original_df['price'].to_numpy()
    budget_labels = assign_budget_tertiles(prices)

    for rank, (idx, dist) in enumerate(zip(indices, distances), start=1):
        if maxd > eps:
            confidence = float(1.0 - (dist / (maxd + eps)))
        else:
            confidence = 1.0  # all zero distances (identical)
        row = original_df.iloc[idx].to_dict()
        row.update({
            'distance': float(dist),
            'confidence': round(confidence, 4),
            'rank': int(rank),
            'budget': str(budget_labels[idx])
        })
        recs.append(row)
    avg_conf = float(np.mean([r['confidence'] for r in recs])) if recs else None
    return {'recommendations': recs, 'avg_confidence': avg_conf}