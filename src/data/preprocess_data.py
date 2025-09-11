# src/data/preprocess_data.py

import cv2
from pathlib import Path
from tqdm import tqdm
from joblib import Parallel, delayed

# --- Paramètres à configurer ---
SOURCE_DIR = Path("/home/jovyan/buckets/ehsebou-manuscrits/Paragraph-Images")
TARGET_DIR = Path("/home/jovyan/buckets/ehsebou-manuscrits/Paragraph-Images-Preprocessed-400px")
TARGET_HEIGHT = 400
# --------------------------------

def process_image(img_path):
    try:
        # Charger en niveaux de gris
        image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"Skipping {img_path}, cannot be read.")
            return

        # Redimensionner en gardant l'aspect ratio
        h, w = image.shape
        scale = TARGET_HEIGHT / h
        new_w = int(w * scale)
        
        resized_image = cv2.resize(image, (new_w, TARGET_HEIGHT), interpolation=cv2.INTER_AREA)

        # Sauvegarder dans le dossier cible
        target_path = TARGET_DIR / img_path.name
        cv2.imwrite(str(target_path), resized_image)

    except Exception as e:
        print(f"Error processing {img_path}: {e}")

if __name__ == "__main__":
    if not SOURCE_DIR.exists():
        print(f"Source directory not found: {SOURCE_DIR}")
        exit()

    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    
    image_paths = list(SOURCE_DIR.glob("*.png"))
    print(f"Found {len(image_paths)} images to process.")

    # Utiliser joblib pour paralléliser le traitement sur tous les cœurs de CPU
    Parallel(n_jobs=-1)(delayed(process_image)(p) for p in tqdm(image_paths, desc="Preprocessing images"))

    print("\nPreprocessing complete.")
    print(f"Processed images are saved in: {TARGET_DIR}")