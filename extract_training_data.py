import os
import numpy as np
from Parameters import Parameters
from ScoobyDooTrainingExtractor import extract_scooby_training_data


def main():
    print("=" * 60)
    print("EXTRAGERE DATE DE ANTRENARE DIN SCENE COMPLETE")
    print("=" * 60)

    # =======================
    # Inițializare parametri
    # =======================
    params = Parameters()
    params.dim_window = 36
    params.dim_hog_cell = 6
    params.overlap = 0.3
    params.number_negative_examples = 0   # va fi actualizat după extragere
    params.use_flip_images = True

    # Afișare configurație
    print("\nConfiguratie:")
    print(f"  Dim window           : {params.dim_window}")
    print(f"  Dim HOG cell         : {params.dim_hog_cell}")
    print(f"  Overlap              : {params.overlap}")
    print(f"  Nr. exemple negative : {params.number_negative_examples}")
    print(f"  Flip imagini         : {params.use_flip_images}\n")

    # =======================
    # Extrage exemple
    # =======================
    print("→ Extragere exemple de antrenare (pozitive și negative)...")
    positive_features, negative_features = extract_scooby_training_data(params)

    # =======================
    # Căi de salvare
    # =======================
    os.makedirs(params.dir_save_files, exist_ok=True)

    positive_path = os.path.join(
        params.dir_save_files,
        f"positive_proper_{params.dim_hog_cell}_{positive_features.shape[0]}.npy"
    )
    negative_path = os.path.join(
        params.dir_save_files,
        f"negative_proper_{params.dim_hog_cell}_{negative_features.shape[0]}.npy"
    )

    # =======================
    # Salvare pe disc
    # =======================
    np.save(positive_path, positive_features)
    np.save(negative_path, negative_features)

    # =======================
    # Raport final
    # =======================
    print("\n" + "=" * 60)
    print("SALVARE FINALIZATĂ")
    print("=" * 60)
    print(f"Pozitive:")
    print(f"  Path  : {positive_path}")
    print(f"  Shape : {positive_features.shape}")
    print(f"\nNegative:")
    print(f"  Path  : {negative_path}")
    print(f"  Shape : {negative_features.shape}")
    print(f"  Număr negative efective: {params.number_negative_examples}")
    print("=" * 60)

    print("\n✓ Gata! Poți folosi aceste fișiere în RunProject.py")
    print("Exemplu:")
    print(f"  positive_features = np.load('{positive_path}')")
    print(f"  negative_features = np.load('{negative_path}')")


if __name__ == "__main__":
    main()
