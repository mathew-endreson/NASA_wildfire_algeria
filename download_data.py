# download_data.py

import math
import requests
import os
import pandas as pd
from PIL import Image

def latlon_to_tile_coords(lat, lon, zoom):
    """Converts latitude and longitude to Slippy Map tile coordinates."""
    lat_rad = math.radians(lat)
    n = 2.0 ** zoom
    xtile = int((lon + 180.0) / 360.0 * n)
    ytile = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    return (xtile, ytile)

def download_tile(date_str, zoom, xtile, ytile, save_path):
    """Downloads a single tile from NASA GIBS."""
    URL_TEMPLATE = (
        "https://gibs.earthdata.nasa.gov/wmts/epsg3857/best/"
        "MODIS_Terra_CorrectedReflectance_TrueColor/default/{date}/"
        "GoogleMapsCompatible_Level9/{zoom}/{y}/{x}.jpg"
    )
    url = URL_TEMPLATE.format(date=date_str, zoom=zoom, y=ytile, x=xtile)
    
    if os.path.exists(save_path):
        print(f"Skipped {save_path}, file already exists.")
        return False

    try:
        response = requests.get(url, stream=True, timeout=15)
        response.raise_for_status()
        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        # Verify that the downloaded file is a valid image before confirming success
        try:
            Image.open(save_path).verify()
            print(f"Downloaded and verified {save_path}")
            return True
        except (IOError, SyntaxError):
            print(f"Downloaded file {save_path} is corrupted. Deleting.")
            os.remove(save_path)
            return False

    except requests.exceptions.RequestException as e:
        print(f"Failed to download {url}. Reason: {e}")
        return False

def main():
    """Main function to run the data acquisition process."""
    # --- Configuration ---
    FIRMS_CSV_PATH = 'firms_data.csv' 
    ZOOM_LEVEL = 9 
    MAX_SAMPLES_PER_CLASS = 300 # Aim for 300 fire and 300 no-fire images

    # Create directories
    os.makedirs('data/fire', exist_ok=True)
    os.makedirs('data/no_fire', exist_ok=True)

    df = pd.read_csv(FIRMS_CSV_PATH)
    df = df[['latitude', 'longitude', 'acq_date']]

    processed_tiles = set()
    fire_samples = 0
    no_fire_samples = 0

    print("--- Starting Image Download Process ---")
    for index, row in df.iterrows():
        if fire_samples >= MAX_SAMPLES_PER_CLASS and no_fire_samples >= MAX_SAMPLES_PER_CLASS:
            print("Reached sample limit for both classes. Stopping.")
            break
            
        lat, lon, date = row['latitude'], row['longitude'], row['acq_date']
        xtile, ytile = latlon_to_tile_coords(lat, lon, ZOOM_LEVEL)
        
        # --- Download "fire" tile ---
        if fire_samples < MAX_SAMPLES_PER_CLASS:
            tile_id = (date, xtile, ytile)
            if tile_id not in processed_tiles:
                path = f"data/fire/{date}_{ZOOM_LEVEL}_{xtile}_{ytile}.jpg"
                if download_tile(date, ZOOM_LEVEL, xtile, ytile, path):
                    fire_samples += 1
                processed_tiles.add(tile_id)

        # --- Download nearby "no_fire" tile ---
        if no_fire_samples < MAX_SAMPLES_PER_CLASS:
            no_fire_xtile, no_fire_ytile = xtile + 5, ytile - 5
            tile_id = (date, no_fire_xtile, no_fire_ytile)
            if tile_id not in processed_tiles:
                path = f"data/no_fire/{date}_{ZOOM_LEVEL}_{no_fire_xtile}_{no_fire_ytile}.jpg"
                if download_tile(date, ZOOM_LEVEL, no_fire_xtile, no_fire_ytile, path):
                    no_fire_samples += 1
                processed_tiles.add(tile_id)

    print("\n--- Dataset Download Process Complete ---")
    print(f"Total fire samples downloaded: {fire_samples}")
    print(f"Total no_fire samples downloaded: {no_fire_samples}")


if __name__ == "__main__":
    main()
    