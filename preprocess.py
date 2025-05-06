import pandas as pd

def preprocess_data(df):
    df = df.dropna(subset=['price', 'location'])
    df['latitude'] = df['location'].apply(lambda loc: loc.get('latitude'))
    df['longitude'] = df['location'].apply(lambda loc: loc.get('longitude'))
    df = df.dropna(subset=['latitude', 'longitude'])
    return df[['price', 'latitude', 'longitude']]
