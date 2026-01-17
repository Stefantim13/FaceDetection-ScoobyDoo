import os
import cv2 as cv
import numpy as np
from skimage.feature import hog


class ScoobyDooTrainingExtractor:
    def __init__(self, params):
        self.params = params
        self.characters = ['daphne', 'fred', 'shaggy', 'velma']

    # ============================================================
    # ANNOTATIONS
    # ============================================================

    def load_all_annotations(self):
        """
        Încarcă toate adnotările.
        Returnează:
            dict[character][filename] = list of [x1, y1, x2, y2]
        """
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
    # GEOMETRY
    # ============================================================

    @staticmethod
    def compute_iou(box_a, box_b):
        """
        Intersection over Union pentru două bounding box-uri
        """
        x1 = max(box_a[0], box_b[0])
        y1 = max(box_a[1], box_b[1])
        x2 = min(box_a[2], box_b[2])
        y2 = min(box_a[3], box_b[3])

        if x2 <= x1 or y2 <= y1:
            return 0.0

        inter_area = (x2 - x1) * (y2 - y1)
        area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
        area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])

        return inter_area / (area_a + area_b - inter_area)

    # ============================================================
    # HOG
    # ============================================================

    def extract_hog_descriptor(self, patch):
        """
        Calculează descriptorul HOG pentru un patch 36x36
        """
        patch = patch.astype(np.float32)
        std = patch.std()

        if std > 0:
            patch = (patch - patch.mean()) / std

        descriptor = hog(
            patch,
            pixels_per_cell=(self.params.dim_hog_cell, self.params.dim_hog_cell),
            cells_per_block=(2, 2),
            feature_vector=True
        )

        if np.any(np.isnan(descriptor)) or np.any(np.isinf(descriptor)):
            return None

        return descriptor

    # ============================================================
    # MAIN EXTRACTION
    # ============================================================

    def extract_training_examples(self):
        """
        Extrage exemple pozitive și negative folosind sliding window multi-scale
        și limitează numărul de negative per imagine la 3.
        """
        annotations = self.load_all_annotations()

        positive_features = []
        negative_features = []

        scales = [0.4, 0.5, 0.6, 0.8, 1.0, 1.2, 1.5, 2.0, 2.5]
        scales = [0.7, 1.0, 1.3]
        iou_pos = 0.5
        iou_neg = 0.3  # sub acest IoU considerăm negative
        max_negatives_per_image = 5

        total_images = sum(len(v) for v in annotations.values())
        processed = 0

        print(f"\n→ Extragere din {total_images} imagini")
        print(f"  Scale-uri: {scales}")
        print(f"  IoU pozitiv ≥ {iou_pos}")
        print(f"  IoU negativ < {iou_neg}\n")

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

                negative_candidates = []

                for scale in scales:
                    scaled_img = cv.resize(img, None, fx=scale, fy=scale)
                    h, w = scaled_img.shape

                    if h < self.params.dim_window or w < self.params.dim_window:
                        continue

                    scaled_boxes = [
                        [int(x * scale) for x in box]
                        for box in gt_boxes
                    ]

                    step = int(self.params.dim_window * (1 - self.params.overlap))

                    for y in range(0, h - self.params.dim_window + 1, step):
                        for x in range(0, w - self.params.dim_window + 1, step):
                            window_box = [
                                x, y,
                                x + self.params.dim_window,
                                y + self.params.dim_window
                            ]

                            max_iou = max(
                                self.compute_iou(window_box, gt)
                                for gt in scaled_boxes
                            )

                            patch = scaled_img[
                                y:y + self.params.dim_window,
                                x:x + self.params.dim_window
                            ]

                            descriptor = self.extract_hog_descriptor(patch)
                            if descriptor is None:
                                continue

                            # Pozitive
                            if max_iou >= iou_pos:
                                positive_features.append(descriptor)
                                if self.params.use_flip_images:
                                    flipped = np.fliplr(patch)
                                    desc_flip = self.extract_hog_descriptor(flipped)
                                    if desc_flip is not None:
                                        positive_features.append(desc_flip)
                            # Candidate negative
                            elif max_iou < iou_neg:
                                negative_candidates.append(descriptor)

                # Limităm negativele la 3 per imagine
                if negative_candidates:
                    num_to_sample = min(max_negatives_per_image, len(negative_candidates))
                    chosen_indices = np.random.choice(len(negative_candidates), num_to_sample, replace=False)
                    for idx in chosen_indices:
                        negative_features.append(negative_candidates[idx])

                if processed % 50 == 0:
                    print(f"[{processed}/{total_images}] "
                          f"Pozitive: {len(positive_features)} | "
                          f"Negative: {len(negative_features)}")

        print("\n" + "=" * 60)
        print("EXTRAGERE FINALIZATĂ")
        print("=" * 60)
        print(f"Pozitive: {len(positive_features)}")
        print(f"Negative: {len(negative_features)}")
        print("=" * 60)

        # Salvăm numărul de negative în params
        self.params.number_negative_examples = len(negative_features)

        return np.array(positive_features), np.array(negative_features)


# ============================================================
# HELPER FUNCTION
# ============================================================

def extract_scooby_training_data(params):
    extractor = ScoobyDooTrainingExtractor(params)
    return extractor.extract_training_examples()
