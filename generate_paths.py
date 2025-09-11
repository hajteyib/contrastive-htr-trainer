import os
from pathlib import Path

# --- Paramètres ---
# Le dossier contenant vos images pré-traitées
image_dir = Path("/home/jovyan/buckets/ehsebou-manuscrits/Paragraph-Images-Preprocessed-400px")

# Le fichier de sortie que nous voulons créer
output_file = Path("/home/jovyan/buckets/ehsebou-manuscrits/valid_image_paths.txt")
# ------------------

# Vérifier que le dossier d'images existe
if not image_dir.is_dir():
    print(f"ERROR: Image directory not found at {image_dir}")
    exit()

# Lister tous les fichiers .png dans le dossier
image_paths = sorted(list(image_dir.glob("*.png")))

# Écrire les chemins absoluts dans le fichier de sortie
with open(output_file, "w") as f:
    for path in image_paths:
        # path.resolve() garantit que le chemin est absolu et sans ambiguïté
        f.write(f"{path.resolve()}\n")

print(f"Successfully created {output_file} with {len(image_paths)} absolute paths.")
