from Parameters import *
from FacialDetector import *
from Visualize import *
import numpy as np
import os

params = Parameters()
params.dim_window = 64
params.dim_hog_cell = 4
params.overlap = 0.3
params.threshold = 3
params.has_annotations = True
params.use_hard_mining = False

facial_detector = FacialDetector(params)

print("="*60)
print("ÎNCĂRCARE DATE ANTRENARE")
print("="*60)

positive_features_path = os.path.join(params.dir_save_files, 'positive_proper_2_6547.npy')
negative_features_path = os.path.join(params.dir_save_files, 'negative_proper_2_20000.npy')

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

training_examples = np.concatenate((positive_features, negative_features), axis=0)
train_labels = np.concatenate((np.ones(len(positive_features)), 
                               np.zeros(len(negative_features))))

print(f'Training examples shape: {training_examples.shape}')
print(f'Train labels shape: {train_labels.shape}')

print("\n" + "="*60)
print("ANTRENARE CLASIFICATOR")
print("="*60)
facial_detector.train_classifier(training_examples, train_labels)

if params.use_hard_mining:
    print("\n" + "="*60)
    print("HARD NEGATIVE MINING")
    print("="*60)
    
    hard_negatives = facial_detector.get_hard_negative_descriptors()
    
    if len(hard_negatives) > 0:
        print(f"Găsite {len(hard_negatives)} hard negatives")
        all_negatives = np.concatenate((negative_features, hard_negatives), axis=0)
        training_examples = np.concatenate((positive_features, all_negatives), axis=0)
        train_labels = np.concatenate((np.ones(len(positive_features)), 
                                       np.zeros(len(all_negatives))))
        print("Re-antrenare cu hard negatives...")
        facial_detector.train_classifier(training_examples, train_labels)
    else:
        print("Nu s-au găsit hard negatives")

print("\n" + "="*60)
print("DETECȚIE ȘI EVALUARE PE SETUL DE TEST")
print("="*60)

detections, scores, file_names = facial_detector.run()

if params.has_annotations:
    print("\n" + "="*60)
    print("EVALUARE REZULTATE")
    print("="*60)
    facial_detector.eval_detections(detections, scores, file_names)
    show_detections_with_ground_truth(detections, scores, file_names, params)
else:
    show_detections_without_ground_truth(detections, scores, file_names, params)