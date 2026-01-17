import os
import cv2 as cv
import numpy as np
from skimage.feature import hog
from collections import defaultdict


class ProperScoobyExtractor:
    def __init__(self, params):
        self.params = params
        self.characters = ['daphne', 'fred', 'shaggy', 'velma']
        self.patch_size = 64  # FIXED native size

    # ============================================================
    # ANNOTATIONS
    # ============================================================

    def load_all_annotations(self):
        all_annotations = {}
        print("→ Încărcare adnotări")

        for character in self.characters:
            annotation_path = os.path.join(
                self.params.base_dir,
                "antrenare",
                f"{character}_annotations.txt"
            )

            if not os.path.exists(annotation_path):
                print(f"  SKIP: {annotation_path} nu există")
                continue

            char_annotations = {}

            with open(annotation_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue

                    filename = parts[0]
                    bbox = list(map(int, parts[1:5]))
                    char_annotations.setdefault(filename, []).append(bbox)

            all_annotations[character] = char_annotations
            total_boxes = sum(len(v) for v in char_annotations.values())
            print(f"  {character}: {len(char_annotations)} imagini, {total_boxes} fețe")

        return all_annotations

    # ============================================================
    # HOG EXTRACTION — no normalization
    # ============================================================

    def extract_hog_descriptor(self, patch):
        """Extract HOG from a patch that is ALREADY 64x64 — no normalization"""
        if patch.shape[:2] != (self.patch_size, self.patch_size):
            return None

        # Only convert to float32 — no mean/std
        patch = patch.astype(np.float32)

        descriptor = hog(
            patch,
            pixels_per_cell=(self.params.dim_hog_cell, self.params.dim_hog_cell),
            cells_per_block=(2, 2),
            feature_vector=True
        )

        if np.any(np.isnan(descriptor)) or np.any(np.isinf(descriptor)):
            return None

        return descriptor

    @staticmethod
    def compute_iou(box_a, box_b):
        x1 = max(box_a[0], box_b[0])
        y1 = max(box_a[1], box_b[1])
        x2 = min(box_a[2], box_b[2])
        y2 = min(box_a[3], box_b[3])

        if x2 <= x1 or y2 <= y1:
            return 0.0

        inter_area = (x2 - x1) * (y2 - y1)
        area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
        area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])

        return inter_area / (area_a + area_b - inter_area + 1e-6)

    # ============================================================
    # MULTI-SCALE EXTRACTION
    # ============================================================

    def extract_training_examples(self):
        """
        Extract 64x64 patches at MULTIPLE SCALES.
        For each face, find the BEST scale where it's closest to 64x64.
        """
        annotations = self.load_all_annotations()

        positive_features = []
        negative_features = []

        # Scales to cover the annotation distribution (10x20 to 180x180)
        scales = [0.3, 0.4, 0.5, 0.6, 0.7, 0.85, 1.0, 1.2, 1.5, 1.8, 2.2, 2.6]
        
        max_negatives_per_image = 5  # Total negatives per image (not per scale)

        total_images = sum(len(v) for v in annotations.values())
        processed = 0

        print(f"\n→ Extragere patch-uri 64×64 MULTI-SCALE")
        print(f"  Strategy: Best scale per face")
        print(f"  Max negatives per image: {max_negatives_per_image}\n")

        for character, char_annotations in annotations.items():
            images_dir = os.path.join(self.params.base_dir, "antrenare", character)

            print("=" * 60)
            print(f"Procesare {character.upper()} ({len(char_annotations)} imagini)")
            print("=" * 60)

            for img_name, gt_boxes in char_annotations.items():
                processed += 1
                img_path = os.path.join(images_dir, img_name)

                img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
                if img is None:
                    continue

                h, w = img.shape

                # ─── POSITIVES: For each GT box, find best scale ───
                for gt_box in gt_boxes:
                    x1, y1, x2, y2 = gt_box
                    w_box = x2 - x1
                    h_box = y2 - y1
                    
                    # Find the scale where this box is closest to 64x64
                    best_scale = None
                    best_scale_diff = float('inf')
                    
                    for scale in scales:
                        scaled_w_box = w_box * scale
                        scaled_h_box = h_box * scale
                        
                        diff = abs(scaled_w_box - self.patch_size) + abs(scaled_h_box - self.patch_size)
                        
                        if diff < best_scale_diff:
                            best_scale_diff = diff
                            best_scale = scale
                    
                    # Extract at best scale only
                    scaled_w = int(w * best_scale)
                    scaled_h = int(h * best_scale)
                    
                    if scaled_w < self.patch_size or scaled_h < self.patch_size:
                        continue
                    
                    scaled_img = cv.resize(img, (scaled_w, scaled_h))
                    
                    scaled_x1 = int(x1 * best_scale)
                    scaled_y1 = int(y1 * best_scale)
                    scaled_x2 = int(x2 * best_scale)
                    scaled_y2 = int(y2 * best_scale)
                    
                    center_x = (scaled_x1 + scaled_x2) // 2
                    center_y = (scaled_y1 + scaled_y2) // 2
                    
                    patch_x1 = max(0, center_x - self.patch_size // 2)
                    patch_y1 = max(0, center_y - self.patch_size // 2)
                    patch_x2 = min(scaled_w, patch_x1 + self.patch_size)
                    patch_y2 = min(scaled_h, patch_y1 + self.patch_size)
                    
                    if patch_x2 - patch_x1 < self.patch_size:
                        patch_x1 = max(0, patch_x2 - self.patch_size)
                    if patch_y2 - patch_y1 < self.patch_size:
                        patch_y1 = max(0, patch_y2 - self.patch_size)
                    
                    patch = scaled_img[patch_y1:patch_y2, patch_x1:patch_x2]
                    
                    if patch.shape[:2] != (self.patch_size, self.patch_size):
                        continue
                    
                    descriptor = self.extract_hog_descriptor(patch)
                    if descriptor is not None:
                        positive_features.append(descriptor)
                        
                        if self.params.use_flip_images:
                            flipped = np.fliplr(patch)
                            desc_flip = self.extract_hog_descriptor(flipped)
                            if desc_flip is not None:
                                positive_features.append(desc_flip)

                # ─── NEGATIVES: Random 64x64 patches at random scale ───
                neg_count = 0
                attempts = 0
                max_attempts = max_negatives_per_image * 30

                while neg_count < max_negatives_per_image and attempts < max_attempts:
                    attempts += 1
                    
                    scale = np.random.choice(scales)
                    scaled_w = int(w * scale)
                    scaled_h = int(h * scale)
                    
                    if scaled_w < self.patch_size or scaled_h < self.patch_size:
                        continue
                    
                    scaled_img = cv.resize(img, (scaled_w, scaled_h))
                    
                    scaled_gt_boxes = [
                        [int(bx1*scale), int(by1*scale), int(bx2*scale), int(by2*scale)]
                        for bx1, by1, bx2, by2 in gt_boxes
                    ]
                    
                    x = np.random.randint(0, scaled_w - self.patch_size + 1)
                    y = np.random.randint(0, scaled_h - self.patch_size + 1)
                    
                    window_box = [x, y, x + self.patch_size, y + self.patch_size]
                    
                    max_iou = max(
                        self.compute_iou(window_box, gt) 
                        for gt in scaled_gt_boxes
                    ) if scaled_gt_boxes else 0
                    
                    if max_iou >= 0.3:
                        continue
                    
                    patch = scaled_img[y:y+self.patch_size, x:x+self.patch_size]
                    descriptor = self.extract_hog_descriptor(patch)
                    
                    if descriptor is not None:
                        negative_features.append(descriptor)
                        neg_count += 1

                if processed % 20 == 0:
                    print(f"[{processed}/{total_images}] "
                          f"Pozitive: {len(positive_features)} | "
                          f"Negative: {len(negative_features)}")

        print("\n" + "=" * 60)
        print("EXTRAGERE FINALIZATĂ")
        print("=" * 60)
        print(f"Pozitive: {len(positive_features)}")
        print(f"Negative: {len(negative_features)}")
        print(f"Ratio: {len(negative_features)/max(1, len(positive_features)):.1f}:1")
        print("=" * 60)

        return np.array(positive_features), np.array(negative_features)


def extract_scooby_training_data(params):
    extractor = ProperScoobyExtractor(params)
    return extractor.extract_training_examples()