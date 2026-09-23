import os
import pandas as pd
import requests
from io import BytesIO
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import logging
from collections import Counter

# Set up logging for skipped images
logging.basicConfig(
    filename='download_errors.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Constants
CSV_FILE = 'AI_Classification-Project.csv'
BASE_DIR = 'dataset'
IMAGE_SIZE = (128, 128)
MAX_WORKERS = 16

def create_directory_structure():
    """Create the required directory structure for the dataset."""
    splits = ['train', 'val', 'test']
    labels = ['REAL', 'FAKE']
    for split in splits:
        for label in labels:
            os.makedirs(os.path.join(BASE_DIR, split, label), exist_ok=True)
            
def download_and_process_image(row):
    """Download, resize, and save an image based on the dataframe row."""
    image_id = row['image_id']
    url = row['image_url']
    label = row['label']
    split = row['dataset_split']
    
    # Check for missing values that would prevent downloading or saving
    if pd.isna(split) or pd.isna(label) or pd.isna(url):
        logging.error(f"Skipped image_id {image_id}: Missing URL, label, or split.")
        return split, label, False
        
    save_path = os.path.join(BASE_DIR, str(split), str(label), f"{image_id}.jpg")
    
    # If the image was already downloaded successfully, skip it
    if os.path.exists(save_path):
        return split, label, True
        
    try:
        # Fetch the image with a timeout
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        # Open the image from bytes
        image = Image.open(BytesIO(response.content))
        
        # Convert to RGB to ensure compatibility when saving as JPEG
        if image.mode != 'RGB':
            image = image.convert('RGB')
            
        # Resize to standard dimensions
        image = image.resize(IMAGE_SIZE)
        
        # Save the image
        image.save(save_path, 'JPEG')
        return split, label, True
        
    except Exception as e:
        logging.error(f"Skipped image_id {image_id} (URL: {url}): {e}")
        return split, label, False

def main():
    # Load CSV, gracefully falling back to 'copy' if original is missing
    if not os.path.exists(CSV_FILE):
        fallback = 'AI_Classification-Project copy.csv'
        if os.path.exists(fallback):
            print(f"'{CSV_FILE}' not found, using '{fallback}' instead.")
            csv_path = fallback
        else:
            print(f"Error: {CSV_FILE} not found in the current directory.")
            return
    else:
        csv_path = CSV_FILE
        
    print(f"Loading dataset from {csv_path}...")
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Failed to load CSV: {e}")
        return
        
    # Verify required columns exist
    required_cols = {'image_id', 'image_url', 'label', 'dataset_split'}
    if not required_cols.issubset(df.columns):
        print(f"Error: CSV must contain the following columns: {required_cols}")
        print(f"Found columns: {df.columns.tolist()}")
        return
        
    print("Creating directory structure...")
    create_directory_structure()
    
    print(f"Starting download of {len(df)} images with {MAX_WORKERS} workers...")
    
    success_counts = Counter()
    
    # Convert dataframe to a list of dicts for faster iteration
    rows = df.to_dict('records')
    
    # Download images concurrently
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit all tasks
        futures = {executor.submit(download_and_process_image, row): row for row in rows}
        
        # Process as they complete and update progress bar
        for future in tqdm(as_completed(futures), total=len(futures), desc="Downloading"):
            split, label, success = future.result()
            if success:
                success_counts[(split, label)] += 1
                
    # Print Final Summary
    print("\n--- Download Summary ---")
    splits = ['train', 'val', 'test']
    labels = ['REAL', 'FAKE']
    total_success = 0
    
    for split in splits:
        for label in labels:
            count = success_counts.get((split, label), 0)
            print(f"{split.upper()} - {label}: {count} images successfully downloaded")
            total_success += count
            
    print(f"\nTotal images successfully downloaded: {total_success} / {len(df)}")
    print("Check 'download_errors.log' for details on any skipped images.")

if __name__ == "__main__":
    main()
