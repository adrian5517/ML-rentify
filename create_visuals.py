"""
create_visuals.py

Generates two matplotlib visualizations (local-only):
 - kmeans_budget.png: properties scatter (lat, lon) colored by budget group (Low/Mid/High)
 - knn_recommendations.png: same scatter plus query point and top-k recommendations highlighted

Usage:
    python create_visuals.py

The script reads MONGO_URI from .env, so ensure .env is present.
"""
import os
from dotenv import load_dotenv
import logging
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
from pymongo import MongoClient

# local modules
from preprocess import preprocess_data
import ml_model

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

load_dotenv()
MONGO_URI = os.getenv('MONGO_URI')

def get_collection():
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
    client.server_info()
    db = client['test']
    return db['properties']


def _location_to_str(loc):
    try:
        if isinstance(loc, dict):
            # prefer address-like fields if present
            for key in ('address', 'addr', 'street', 'city', 'municipality'):
                if key in loc and loc.get(key):
                    return str(loc.get(key))
            # fallback to lat/lon
            lat = loc.get('latitude')
            lon = loc.get('longitude')
            if lat is not None and lon is not None:
                return f"lat:{lat:.6f}, lon:{lon:.6f}"
            return str(loc)
        return str(loc)
    except Exception:
        return str(loc)


def plot_kmeans(df_clean, labels, out_path='kmeans_budget.png'):
    # df_clean expected columns: price, latitude, longitude, name, location_str
    lat = df_clean['latitude'].astype(float)
    lon = df_clean['longitude'].astype(float)
    prices = df_clean['price'].astype(float)
    names = df_clean.get('name', pd.Series([''] * len(df_clean))).astype(str)
    # location string for display
    if 'location_str' in df_clean.columns:
        locs = df_clean['location_str'].astype(str)
    elif 'location' in df_clean.columns:
        locs = df_clean['location'].astype(str).apply(lambda x: _location_to_str(x) if pd.notnull(x) else '')
    else:
        locs = pd.Series([''] * len(df_clean))

    # ensure labels align with df_clean length (trim or pad if necessary)
    labels_list = list(labels) if labels is not None else []
    if len(labels_list) < len(df_clean):
        labels_list = labels_list + ['Unknown'] * (len(df_clean) - len(labels_list))
    elif len(labels_list) > len(df_clean):
        labels_list = labels_list[:len(df_clean)]
    labels = labels_list

    # color map for budgets (consistent and friendly)
    budget_to_color = {'Low Budget': '#2ca02c', 'Mid-Range': '#1f77b4', 'High-End': '#d62728'}
    colors = [budget_to_color.get(b, '#7f7f7f') for b in labels]

    fig, ax = plt.subplots(figsize=(11, 6))

    # marker size scaled by price (simple linear scaling)
    pmin = float(prices.min()) if len(prices) else 0.0
    pmax = float(prices.max()) if len(prices) else 1.0
    if pmax > pmin:
        sizes = 80 + ((prices - pmin) / (pmax - pmin)) * 220
    else:
        sizes = np.full(len(prices), 120)

    # plot properties
    scatter = ax.scatter(lon, lat, c=colors, s=sizes, edgecolor='k', linewidth=0.6, alpha=0.9)

    # annotate each point with name and price (name above, price below)
    for i, (x, y) in enumerate(zip(lon, lat)):
        # show address instead of property name (shortened) and price below
        price = prices.iloc[i] if i < len(prices) else 0
        addr = locs.iloc[i] if i < len(locs) else ''
        addr_short = (addr[:40] + '...') if isinstance(addr, str) and len(addr) > 43 else addr
        # address above, price below for non-technical readability
        ax.text(x + 0.0006, y + 0.00025, f"{addr_short}", fontsize=8, va='bottom')
        ax.text(x + 0.0006, y - 0.00025, f"₱{int(price):,}", fontsize=8, va='top')

    # compute and plot group centroids (use labels grouping)
    dfz = df_clean.reset_index(drop=True).copy()
    dfz['budget'] = labels[:len(dfz)]
    for lbl in sorted(set(labels), key=lambda x: x or ''):
        grp = dfz[dfz['budget'] == lbl]
        if len(grp) == 0:
            continue
        cen_lon = float(grp['longitude'].astype(float).mean())
        cen_lat = float(grp['latitude'].astype(float).mean())
        ax.scatter([cen_lon], [cen_lat], marker='X', s=220, c='k', edgecolor='w', linewidth=1.2)

    # legend for budgets and centroid marker
    import matplotlib.patches as mpatches
    handles = [mpatches.Patch(color=c, label=l) for l, c in budget_to_color.items()]
    centroid_handle = mpatches.Patch(facecolor='k', label='Cluster Center')
    handles.append(centroid_handle)
    ax.legend(handles=handles, title='Budget Category', loc='upper right')

    ax.set_title('KMeans Clustering of Rentals (Based on Price & Location)')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.grid(True, alpha=0.25)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    logging.info(f'Saved KMeans budget plot to %s', out_path)
    plt.close()


def plot_knn_basic(df_clean, recs, query_point, out_path='knn_basic.png'):
    """Simple KNN plot: shows properties, the query point, and the k nearest neighbors.
    No tables or long annotations — suitable for quick non-technical view.
    """
    lat = df_clean['latitude'].astype(float)
    lon = df_clean['longitude'].astype(float)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(lon, lat, c='#d3d3d3', s=60, label='Properties', alpha=0.8, edgecolor='k')

    recs_list = recs.get('recommendations', [])
    if recs_list:
        rec_lons = [r.get('longitude') for r in recs_list]
        rec_lats = [r.get('latitude') for r in recs_list]
        # neighbor style: light blue fill, green ring
        ax.scatter(rec_lons, rec_lats, facecolor='#a6cee3', edgecolor='#2ca02c', s=360, linewidth=2, marker='o', label='Nearest Neighbors')
        # annotate with simple price and address and rank (compact)
        for r in recs_list:
            x = r.get('longitude')
            y = r.get('latitude')
            price = r.get('price', 0)
            addr = r.get('location') or r.get('location_str') or ''
            addr_str = _location_to_str(addr) if addr else ''
            addr_short = (addr_str[:30] + '...') if isinstance(addr_str, str) and len(addr_str) > 33 else addr_str
            # show address prominently, then price
            ax.annotate(f"#{r.get('rank')} {addr_short}\n₱{int(price):,}", (x, y), textcoords='offset points', xytext=(6, -6), fontsize=8)

    q_lon, q_lat = query_point[2], query_point[1]
    # user input marker: bold black X
    ax.scatter([q_lon], [q_lat], c='k', s=220, marker='X', edgecolor='w', linewidth=1.2, label='User Input')

    ax.set_title('KNN - Nearest Neighbors (basic)')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    logging.info(f'Saved basic KNN plot to {out_path}')
    plt.close()


def plot_knn_recommendations(df_clean, recs, query_point, out_path='knn_recommendations.png'):
    """
    Detailed KNN recommendations plot: includes annotations (name, price) and a bottom table
    listing the recommended properties with confidence scores — intended for review.
    """
    # expect df_clean to have columns: price, latitude, longitude, name
    lat = df_clean['latitude'].astype(float)
    lon = df_clean['longitude'].astype(float)

    # build base figure and axis
    from matplotlib import gridspec
    recs_list = recs.get('recommendations', [])
    if recs_list:
        # make room for table below
        fig = plt.figure(figsize=(10, 8))
        gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1], hspace=0.4)
        ax = fig.add_subplot(gs[0])
    else:
        fig, ax = plt.subplots(figsize=(10, 7))

    ax.scatter(lon, lat, c='#b0b0b0', s=60, label='Properties', alpha=0.8, edgecolor='k')

    # plot recommendations with annotations
    if recs_list:
        rec_lons = [r.get('longitude') for r in recs_list]
        rec_lats = [r.get('latitude') for r in recs_list]
        # detailed neighbor style: light blue fill, green ring
        ax.scatter(rec_lons, rec_lats, facecolor='#a6cee3', edgecolor='#2ca02c', s=420, linewidth=2, marker='o', label='Nearest Neighbor')
        for r in recs_list:
            lon_r = r.get('longitude')
            lat_r = r.get('latitude')
            price_r = r.get('price')
            addr = r.get('location') or r.get('location_str') or ''
            addr_str = _location_to_str(addr) if addr else ''
            addr_short = (addr_str[:40] + '...') if isinstance(addr_str, str) and len(addr_str) > 43 else addr_str
            label = f"#{r.get('rank')} {addr_short}\n₱{int(price_r):,}"
            ax.annotate(label, (lon_r, lat_r), textcoords='offset points', xytext=(6, -6), fontsize=9)

    # plot query point
    q_lon, q_lat = query_point[2], query_point[1]
    ax.scatter([q_lon], [q_lat], c='k', s=260, marker='X', edgecolor='w', linewidth=1.2, label='User Input')

    ax.set_title(f"KNN Recommendations (top {len(recs_list)})")
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # if recommendations exist, add table with rank, name, price, confidence
    if recs_list:
        table_df = pd.DataFrame([{ 'Rank': r.get('rank'), 'Name': r.get('name'), 'Price': f"₱{int(r.get('price')):,}", 'Confidence': round(r.get('confidence',0),3) } for r in recs_list])
        ax_table = fig.add_subplot(gs[1])
        ax_table.axis('off')
        cell_text = table_df.values.tolist()
        col_labels = table_df.columns.tolist()
        table = ax_table.table(cellText=cell_text, colLabels=col_labels, loc='center')
        table.auto_set_font_size(False); table.set_fontsize(9); table.scale(1, 1.2)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    logging.info(f'Saved KNN recommendations plot to {out_path}')
    plt.close()


def main():
    try:
        col = get_collection()
    except Exception as e:
        logging.error('Failed to connect to MongoDB: %s', e)
        return

    docs = list(col.find())
    if not docs:
        logging.error('No documents found in collection')
        return

    df = pd.DataFrame(docs)

    # Preprocess and scale
    scaled, scaler, df_clean = preprocess_data(df, scaler=None, fit=True)
    if df_clean.empty:
        logging.error('No valid data after preprocessing')
        return

    # For KMeans budget grouping use tertiles (suitable for small datasets)
    prices = df_clean['price'].to_numpy()
    kres = ml_model.train_kmeans(scaled, prices, enforce_tertiles=True)
    labels = kres.get('labels', [])

    # write a simple CSV summary for non-technical users: Name, Price, Address, Budget
    def write_summary_csv(df_clean_local, labels_local, out_path='properties_summary.csv'):
        df2 = df_clean_local.reset_index(drop=True).copy()
        df2['Budget'] = list(labels_local)[:len(df2)]
        # address extraction
        if 'location_str' in df2.columns:
            df2['Address'] = df2['location_str'].astype(str)
        elif 'location' in df2.columns:
            df2['Address'] = df2['location'].apply(lambda x: _location_to_str(x) if pd.notnull(x) else '')
        else:
            df2['Address'] = ''
        out = pd.DataFrame({
            'Name': df2.get('name', pd.Series([''] * len(df2))).astype(str),
            'Price': df2['price'].astype(float).apply(lambda x: int(x)),
            'Address': df2['Address'],
            'Budget': df2['Budget'],
            'Latitude': df2['latitude'].astype(float),
            'Longitude': df2['longitude'].astype(float)
        })
        out.to_csv(out_path, index=False)
        logging.info('Wrote property summary to %s', out_path)

    write_summary_csv(df_clean, labels, out_path='properties_summary.csv')

    # Plot KMeans (budget categories)
    plot_kmeans(df_clean, labels, out_path='kmeans_budget.png')

    # Build a query point: use median price and mean coords as example
    q_price = float(np.median(prices))
    q_lat = float(df_clean['latitude'].astype(float).mean())
    q_lon = float(df_clean['longitude'].astype(float).mean())
    query_point = [q_price, q_lat, q_lon]

    # Scale query
    query_scaled = scaler.transform([query_point])[0]

    # KNN recommendations
    recs = ml_model.knn_recommend(scaled, df_clean.reset_index(drop=True), query_scaled, k=5)

    # Plot a simple/basic KNN (neighbors + query)
    plot_knn_basic(df_clean, recs, query_point, out_path='knn_basic.png')

    # Plot detailed KNN recommendations (annotations + table)
    plot_knn_recommendations(df_clean, recs, query_point, out_path='knn_recommendations.png')

    logging.info('Done. Generated kmeans_budget.png and knn_recommendations.png')

if __name__ == '__main__':
    main()
