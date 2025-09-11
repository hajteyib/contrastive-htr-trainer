# Fichier : src/data/prepare_dataset.py

import cv2
from pathlib import Path
from tqdm import tqdm
from joblib import Parallel, delayed
from PIL import Image
import argparse
from typing import Optional

def process_and_validate_image(img_path: Path, target_dir: Path, target_height: int) -> Optional[Path]:
    """
    Tente de charger, valider, redimensionner et sauvegarder une image.
    Retourne le nouveau chemin si succès, None sinon.
    """
    try:
        # Utiliser PIL pour une validation robuste contre les formats corrompus
        Image.open(img_path).verify()
        
        # Utiliser OpenCV pour le traitement, c'est plus rapide
        image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            return None # Fichier non lisible par OpenCV

        h, w = image.shape
        if h == 0 or w == 0: return None # Image vide
        
        # Redimensionner en gardant l'aspect ratio
        scale = target_height / h
        new_w = int(w * scale)
        if new_w == 0: return None
        
        resized_image = cv2.resize(image, (new_w, target_height), interpolation=cv2.INTER_AREA)

        # Sauvegarder dans le dossier cible
        target_path = target_dir / img_path.name
        cv2.imwrite(str(target_path), resized_image)
        
        return target_path.resolve()
    except Exception:
        # Ignore silencieusement et retourne None pour toute image qui cause une erreur
        return None

def main():
    parser = argparse.ArgumentParser(description="Preprocess a large dataset of handwriting images for HTR experiments.")
    parser.add_argument("--source_dir", type=str, required=True, help="Path to the source directory with original images.")
    parser.add_argument("--target_dir", type=str, required=True, help="Path to the target directory for preprocessed images.")
    parser.add_argument("--list_file", type=str, required=True, help="Path to the output text file for valid image paths.")
    parser.add_argument("--height", type=int, default=128, help="Target height to resize all images to.")
    args = parser.parse_args()

    source_dir = Path(args.source_dir)
    target_dir = Path(args.target_dir)
    list_file = Path(args.list_file)
    target_height = args.height

    if not source_dir.is_dir():
        print(f"ERREUR: Le dossier source n'a pas été trouvé : {source_dir}")
        return

    target_dir.mkdir(parents=True, exist_ok=True)
    list_file.parent.mkdir(parents=True, exist_ok=True)
    
    all_source_paths = sorted(list(source_dir.glob("*.png")))
    print(f"Trouvé {len(all_source_paths)} images candidates dans {source_dir}.")
    print("Démarrage du pré-traitement et de la validation (utilisation de tous les coeurs CPU)...")

    results = Parallel(n_jobs=-1)(
        delayed(process_and_validate_image)(p, target_dir, target_height) for p in tqdm(all_source_paths)
    )

    valid_target_paths = sorted([path for path in results if path is not None])
    print(f"\nPré-traitement terminé. {len(valid_target_paths)} images ont été traitées avec succès.")
    
    with open(list_file, "w") as f:
        for path in valid_target_paths:
            f.write(f"{path}\n")
            
    print(f"✅ La liste des chemins valides a été sauvegardée dans : {list_file}")

if __name__ == "__main__":
    main()