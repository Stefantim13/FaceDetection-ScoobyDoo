from Parameters import *
import numpy as np
import glob
import cv2 as cv
import pickle
import ntpath
from skimage.feature import hog
import os
from collections import defaultdict
import warnings

from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils.class_weight import compute_class_weight


class FacialDetector:
    def __init__(self, params: Parameters):
        self.params = params
        self.best_model = None
        self.scaler = None

    def normalize_image(self, img):
        """Convert to uint8 if needed"""
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)
        return img
    
    def normalize_hog(self, descr):
        """Normalize HOG descriptor"""
        norm = np.linalg.norm(descr)
        if norm > 1e-10:
            descr = descr / norm
        return descr

    def get_positive_descriptors(self):
        descriptors = []
        all_files = []
        for dir_path in self.params.dirs_pos_examples:
            all_files.extend(glob.glob(os.path.join(dir_path, '*.jpg')))

        for f in all_files:
            img = cv.imread(f, cv.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv.resize(img, (self.params.dim_window, self.params.dim_window))

            # Compute HOG directly on uint8 image
            descr = hog(img,
                        pixels_per_cell=(self.params.dim_hog_cell, self.params.dim_hog_cell),
                        cells_per_block=(2, 2),
                        feature_vector=True)
            
            if np.all(np.isfinite(descr)):
                descr = self.normalize_hog(descr)
                descriptors.append(descr)

            if self.params.use_flip_images:
                descr_flip = hog(np.fliplr(img),
                                 pixels_per_cell=(self.params.dim_hog_cell, self.params.dim_hog_cell),
                                 cells_per_block=(2, 2),
                                 feature_vector=True)
                if np.all(np.isfinite(descr_flip)):
                    descr_flip = self.normalize_hog(descr_flip)
                    descriptors.append(descr_flip)

        return np.array(descriptors)

    def get_negative_descriptors(self):
        descriptors = []
        files = glob.glob(os.path.join(self.params.dir_neg_examples, '*.jpg'))
        num_images = len(files)
        if num_images == 0:
            return np.array([])

        neg_per_image = max(1, self.params.number_negative_examples // num_images)

        for f in files:
            img = cv.imread(f, cv.IMREAD_GRAYSCALE)
            if img is None:
                continue
            h, w = img.shape
            if h < self.params.dim_window or w < self.params.dim_window:
                continue

            xs = np.random.randint(0, w - self.params.dim_window + 1, neg_per_image)
            ys = np.random.randint(0, h - self.params.dim_window + 1, neg_per_image)

            for x, y in zip(xs, ys):
                patch = img[y:y+self.params.dim_window, x:x+self.params.dim_window]
                if patch.shape != (self.params.dim_window, self.params.dim_window):
                    continue
                
                # Compute HOG directly on uint8 patch
                descr = hog(patch,
                            pixels_per_cell=(self.params.dim_hog_cell, self.params.dim_hog_cell),
                            cells_per_block=(2, 2),
                            feature_vector=True)
                if np.all(np.isfinite(descr)):
                    descr = self.normalize_hog(descr)
                    descriptors.append(descr)

        return np.array(descriptors)

    def train_classifier(self, examples, labels):
        model_path = os.path.join(
            self.params.dir_save_files,
            f'best_linear_model_{self.params.dim_hog_cell}_{self.params.number_negative_examples}.pkl'
        )
        scaler_path = os.path.join(
            self.params.dir_save_files,
            f'scaler_{self.params.dim_hog_cell}_{self.params.number_negative_examples}.pkl'
        )

        if os.path.exists(model_path) and os.path.exists(scaler_path):
            with open(model_path, 'rb') as f:
                self.best_model = pickle.load(f)
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            print(f'Loaded model: {model_path}')
            print(f'Loaded scaler: {scaler_path}')
            return

        # Subsample if too many examples
        if len(examples) > 150000:
            print(f"Subsampling from {len(examples)} examples...")
            pos_mask = labels == 1
            neg_mask = labels == 0
            
            pos_examples = examples[pos_mask]
            neg_examples = examples[neg_mask]
            
            target_negatives = min(len(neg_examples), len(pos_examples) * 3)
            if len(neg_examples) > target_negatives:
                neg_indices = np.random.choice(len(neg_examples), target_negatives, replace=False)
                neg_examples = neg_examples[neg_indices]
            
            examples = np.vstack([pos_examples, neg_examples])
            labels = np.concatenate([np.ones(len(pos_examples)), np.zeros(len(neg_examples))])
            print(f"Subsampled to {len(examples)} examples")

        # Fit scaler
        print("Fitting StandardScaler...")
        self.scaler = StandardScaler()
        examples_scaled = self.scaler.fit_transform(examples)

        X_train, X_val, y_train, y_val = train_test_split(
            examples_scaled, labels,
            test_size=0.15,
            stratify=labels,
            random_state=42
        )

        class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
        class_weight_dict = {0: class_weights[0], 1: class_weights[1]}

        best_val_acc = 0.0
        best_model = None
        best_C = None

        print("\n" + "="*60)
        print("VALIDATION DURING HYPERPARAMETER SEARCH")
        print("="*60)

        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=RuntimeWarning)
            
            for C in [0.001, 0.01, 0.1, 1.0]:
                print(f"  Training C = {C:.3f}...", end=' ', flush=True)
                
                model = LinearSVC(
                    C=C,
                    class_weight=class_weight_dict,
                    max_iter=5000,
                    tol=1e-3,
                    dual='auto',
                    random_state=42
                )

                model.fit(X_train, y_train)
                val_acc = accuracy_score(y_val, model.predict(X_val))
                print(f"validation accuracy = {val_acc:.4f}")

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_model = model
                    best_C = C

        self.best_model = best_model

        with open(model_path, 'wb') as f:
            pickle.dump(best_model, f)
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)

        print("\n" + "="*60)
        print("FINAL RESULTS")
        print("="*60)
        print(f"Best C: {best_C:.3f}")
        print(f"Best validation accuracy: {best_val_acc:.4f}")

        val_preds = best_model.predict(X_val)
        print("\nClassification report:")
        print(classification_report(y_val, val_preds,
                                    target_names=['Negative', 'Positive'],
                                    digits=4))

    def non_maximal_suppression(self, boxes, scores, iou_threshold=0.3):
        if len(boxes) == 0:
            return boxes, scores

        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]
        keep = []

        while order.size > 0:
            idx = order[0]
            keep.append(idx)

            xx1 = np.maximum(x1[idx], x1[order[1:]])
            yy1 = np.maximum(y1[idx], y1[order[1:]])
            xx2 = np.minimum(x2[idx], x2[order[1:]])
            yy2 = np.minimum(y2[idx], y2[order[1:]])

            w = np.maximum(0., xx2 - xx1)
            h = np.maximum(0., yy2 - yy1)
            inter_area = w * h

            union = areas[idx] + areas[order[1:]] - inter_area
            iou = inter_area / (union + 1e-6)

            mask = iou > iou_threshold
            order = order[1:][~mask]

        return boxes[keep], scores[keep]

    def run(self):
        if self.best_model is None:
            raise RuntimeError("Model not trained or loaded!")
        if self.scaler is None:
            raise RuntimeError("Scaler not fitted or loaded!")

        detections, scores, file_names = [], [], []
        test_files = sorted(glob.glob(os.path.join(self.params.dir_test_examples, '*.jpg')))

        scales = [0.4, 0.6, 0.8, 1.0, 1.3, 1.6, 2.0]
        stride = 6

        print(f"\nRunning detector on {len(test_files)} images...")
        print(f"Scales: {scales}")
        print(f"Stride: {stride}")
        print(f"Threshold: {self.params.threshold}\n")

        for i, file in enumerate(test_files):
            if (i + 1) % 10 == 0:
                print(f'Progress: {i+1}/{len(test_files)}')
            
            img = cv.imread(file, cv.IMREAD_GRAYSCALE)
            if img is None:
                continue

            original_h, original_w = img.shape[:2]
            image_dets, image_scores = [], []

            # Pre-compute scaled images and HOG maps for all scales
            scale_data = []
            for scale in scales:
                scaled_w = int(original_w * scale)
                scaled_h = int(original_h * scale)
                if scaled_w < self.params.dim_window or scaled_h < self.params.dim_window:
                    continue

                scaled_img = cv.resize(img, (scaled_w, scaled_h))
                
                # Compute HOG map once per scale
                hog_map = hog(
                    scaled_img,
                    pixels_per_cell=(self.params.dim_hog_cell, self.params.dim_hog_cell),
                    cells_per_block=(2, 2),
                    feature_vector=False
                )
                
                scale_data.append({
                    'scale': scale,
                    'hog_map': hog_map,
                    'scaled_w': scaled_w,
                    'scaled_h': scaled_h
                })

            # Now process all scales together with batch prediction
            all_descriptors = []
            all_positions = []
            all_scale_info = []
            
            n_template = self.params.dim_window // self.params.dim_hog_cell - 1
            
            for scale_info in scale_data:
                hog_map = scale_info['hog_map']
                n_blocks_y, n_blocks_x = hog_map.shape[:2]
                
                for y in range(0, n_blocks_y - n_template, stride):
                    for x in range(0, n_blocks_x - n_template, stride):
                        descr = hog_map[y:y+n_template, x:x+n_template].flatten()
                        descr = self.normalize_hog(descr)
                        
                        all_descriptors.append(descr)
                        all_positions.append((x, y))
                        all_scale_info.append(scale_info)

            if len(all_descriptors) == 0:
                continue

            # Single batch prediction for ALL scales at once!
            all_descriptors = np.array(all_descriptors)
            
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=RuntimeWarning)
                descriptors_scaled = self.scaler.transform(all_descriptors)
                batch_scores = self.best_model.decision_function(descriptors_scaled)

            # Process all detections
            for (x, y), scale_info, score in zip(all_positions, all_scale_info, batch_scores):
                if score > self.params.threshold:
                    x_min = x * self.params.dim_hog_cell
                    y_min = y * self.params.dim_hog_cell
                    x_max = x_min + self.params.dim_window
                    y_max = y_min + self.params.dim_window

                    scale_x = original_w / scale_info['scaled_w']
                    scale_y = original_h / scale_info['scaled_h']
                    x_min = int(x_min * scale_x)
                    x_max = int(x_max * scale_x)
                    y_min = int(y_min * scale_y)
                    y_max = int(y_max * scale_y)

                    image_dets.append([x_min, y_min, x_max, y_max])
                    image_scores.append(score)

            if image_scores:
                image_dets, image_scores = self.non_maximal_suppression(
                    np.array(image_dets),
                    np.array(image_scores),
                    iou_threshold=0.3
                )
                detections.extend(image_dets)
                scores.extend(image_scores)
                file_names.extend([ntpath.basename(file)] * len(image_scores))

        print(f"\nDetection complete! Found {len(detections)} detections total")
        return np.array(detections), np.array(scores), np.array(file_names)

    def eval_detections(self, detections, scores, file_names):
        dets_per_image = defaultdict(list)
        for det, score, fname in zip(detections, scores, file_names):
            dets_per_image[fname].append((det, score))

        for fname in sorted(dets_per_image.keys()):
            print(f"{fname}: {len(dets_per_image[fname])} detections")

        print(f"\nTotal images with detections: {len(dets_per_image)}")
        print(f"Total detections: {len(detections)}")