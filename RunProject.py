from Parameters import *
from FacialDetector import *
import pdb
from Visualize import *


params: Parameters = Parameters()
params.dim_window = 36
params.dim_hog_cell = 6
params.overlap = 0.3
params.threshold = 4
params.number_negative_examples = 10000
params.has_annotations = True
params.use_hard_mining = False
params.use_flip_images = True

facial_detector: FacialDetector = FacialDetector(params)

print("="*60)
print("ÎNCĂRCARE DATE ANTRENARE")
print("="*60)

# Încarcă descriptori din scene complete (NU din cropped!)
positive_features_path = os.path.join(params.dir_save_files, 'positive_proper_6_5208.npy')
negative_features_path = os.path.join(params.dir_save_files, 'negative_proper_6_20000.npy')

if os.path.exists(positive_features_path) and os.path.exists(negative_features_path):
    positive_features = np.load(positive_features_path)
    negative_features = np.load(negative_features_path)
    print('✓ Am încărcat descriptorii din scene complete')
    print(f'  Pozitive: {positive_features.shape}')
    print(f'  Negative: {negative_features.shape}')
else:
    print('EROARE: Nu există fișierele de descriptori!')
    print('Rulează mai întâi: python extract_training_data.py')
    exit(1)

# Verificare dimensiuni
print(f'\nVerificare dimensiuni:')
print(f'Positive features shape: {positive_features.shape}')
print(f'Negative features shape: {negative_features.shape}')

# Pregătire date antrenare
training_examples = np.concatenate((positive_features, negative_features), axis=0)
train_labels = np.concatenate((np.ones(len(positive_features)), 
                               np.zeros(len(negative_features))))

print(f'Training examples shape: {training_examples.shape}')
print(f'Train labels shape: {train_labels.shape}')

# Antrenare clasificator
print("\n" + "="*60)
print("ANTRENARE CLASIFICATOR")
print("="*60)
facial_detector.train_classifier(training_examples, train_labels)

# Hard negative mining (opțional)
if params.use_hard_mining:
    print("\n" + "="*60)
    print("HARD NEGATIVE MINING")
    print("="*60)
    
    hard_negatives = facial_detector.get_hard_negative_descriptors()
    
    if len(hard_negatives) > 0:
        print(f"Găsite {len(hard_negatives)} hard negatives")
        
        # Adaugă la negative existente
        all_negatives = np.concatenate((negative_features, hard_negatives), axis=0)
        
        # Re-antrenează
        training_examples = np.concatenate((positive_features, all_negatives), axis=0)
        train_labels = np.concatenate((np.ones(len(positive_features)), 
                                       np.zeros(len(all_negatives))))
        
        print("Re-antrenare cu hard negatives...")
        facial_detector.train_classifier(training_examples, train_labels)
    else:
        print("Nu s-au găsit hard negatives")

# Detecție
print("\n" + "="*60)
print("DETECȚIE PE IMAGINI DE TEST")
print("="*60)
detections, scores, file_names = facial_detector.run()

# Evaluare
if params.has_annotations:
    print("\n" + "="*60)
    print("EVALUARE REZULTATE")
    print("="*60)
    facial_detector.eval_detections(detections, scores, file_names)
    show_detections_with_ground_truth(detections, scores, file_names, params)
else:
    show_detections_without_ground_truth(detections, scores, file_names, params)