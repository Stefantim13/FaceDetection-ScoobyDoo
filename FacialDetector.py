from Parameters import *
import numpy as np
from sklearn.svm import LinearSVC
import glob
import cv2 as cv
import pickle
import ntpath
from copy import deepcopy
from skimage.feature import hog
import os
from collections import defaultdict

class FacialDetector:
    def __init__(self, params: Parameters):
        self.params = params
        self.best_model = None

    # =========================================================
    # IMAGE NORMALIZATION (safe)
    # =========================================================
    def normalize_image(self, img):
        img = img.astype(np.float32)
        std = img.std()
        if std > 1e-6:
            img = (img - img.mean()) / std
        else:
            img = img - img.mean()
        return img

    # =========================================================
    # POSITIVE DESCRIPTORS
    # =========================================================
    def get_positive_descriptors(self):
        descriptors = []
        all_files = []
        for dir_path in self.params.dirs_pos_examples:
            all_files.extend(glob.glob(os.path.join(dir_path, '*.jpg')))
        print(f'Calculam descriptorii pentru {len(all_files)} imagini pozitive...')

        for i, f in enumerate(all_files):
            print(f'Pozitiv {i}/{len(all_files)}')
            img = cv.imread(f, cv.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv.resize(img, (self.params.dim_window, self.params.dim_window))
            img = self.normalize_image(img)

            descr = hog(img,
                        pixels_per_cell=(self.params.dim_hog_cell, self.params.dim_hog_cell),
                        cells_per_block=(2, 2),
                        feature_vector=True)
            if np.all(np.isfinite(descr)):
                descriptors.append(descr)

            # Flipped image
            if self.params.use_flip_images:
                descr_flip = hog(np.fliplr(img),
                                 pixels_per_cell=(self.params.dim_hog_cell, self.params.dim_hog_cell),
                                 cells_per_block=(2, 2),
                                 feature_vector=True)
                if np.all(np.isfinite(descr_flip)):
                    descriptors.append(descr_flip)

        return np.array(descriptors)

    # =========================================================
    # NEGATIVE DESCRIPTORS
    # =========================================================
    def get_negative_descriptors(self):
        descriptors = []
        files = glob.glob(os.path.join(self.params.dir_neg_examples, '*.jpg'))
        num_images = len(files)
        if num_images == 0:
            return np.array([])

        neg_per_image = max(1, self.params.number_negative_examples // num_images)
        print(f'Calculam descriptorii negativi din {num_images} imagini...')

        for i, f in enumerate(files):
            print(f'Negativ {i}/{num_images}')
            img = cv.imread(f, cv.IMREAD_GRAYSCALE)
            if img is None:
                continue
            h, w = img.shape
            if h < self.params.dim_window or w < self.params.dim_window:
                continue
            img = self.normalize_image(img)

            xs = np.random.randint(0, w - self.params.dim_window, neg_per_image)
            ys = np.random.randint(0, h - self.params.dim_window, neg_per_image)

            for x, y in zip(xs, ys):
                patch = img[y:y+self.params.dim_window, x:x+self.params.dim_window]
                if patch.shape != (self.params.dim_window, self.params.dim_window):
                    continue
                descr = hog(patch,
                            pixels_per_cell=(self.params.dim_hog_cell, self.params.dim_hog_cell),
                            cells_per_block=(2, 2),
                            feature_vector=True)
                if np.all(np.isfinite(descr)):
                    descriptors.append(descr)

        return np.array(descriptors)

    # =========================================================
    # TRAIN SVM CLASSIFIER
    # =========================================================
    def train_classifier(self, examples, labels):
        model_path = os.path.join(
            self.params.dir_save_files,
            f'best_model_{self.params.dim_hog_cell}_{self.params.number_negative_examples}.pkl'
        )
        if os.path.exists(model_path):
            self.best_model = pickle.load(open(model_path, 'rb'))
            print(f'Loaded existing model: {model_path}')
            return

        best_acc = 0
        best_model = None
        for C in [1e-3, 1e-2, 1e-1, 1]:
            print(f'Train SVM C={C}')
            model = LinearSVC(C=C, max_iter=5000, class_weight='balanced')
            model.fit(examples, labels)
            acc = model.score(examples, labels)
            if acc > best_acc:
                best_acc = acc
                best_model = deepcopy(model)

        self.best_model = best_model
        pickle.dump(best_model, open(model_path, 'wb'))
        print(f'SVM antrenat. Accuracy train: {best_acc:.3f}')

    # =========================================================
    # NON-MAXIMAL SUPPRESSION
    # =========================================================
    def non_maximal_suppression(self, boxes, scores, iou_threshold=0.3):
        """
        Standard Non-Maximal Suppression (NMS):
        - Keeps the highest-scoring box among overlapping boxes.
        - Removes boxes with IoU > iou_threshold.
        - Works for boxes of different sizes.
        """

        if len(boxes) == 0:
            return boxes, scores

        # Coordinates of boxes
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)

        # Sort boxes by score (descending)
        order = scores.argsort()[::-1]
        keep = []

        while len(order) > 0:
            i = order[0]  # index of current highest score
            keep.append(i)

            # Compute intersection with remaining boxes
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)
            inter = w * h

            # Compute IoU
            iou = inter / (areas[i] + areas[order[1:]] - inter)

            # Keep boxes with IoU <= threshold (remove highly overlapping boxes)
            inds_to_keep = np.where(iou <= iou_threshold)[0]
            order = order[inds_to_keep + 1]  # +1 because order[0] is current box

        # Return filtered boxes and corresponding scores
        return boxes[keep], scores[keep]


    # =========================================================
    # RUN DETECTION (multi-scale)
    # =========================================================
    def run(self):
        if self.best_model is None:
            raise RuntimeError("Modelul nu a fost antrenat sau incarcat!")

        detections, scores, file_names = [], [], []
        test_files = glob.glob(os.path.join(self.params.dir_test_examples, '*.jpg'))

        # SVM weights
        w = self.best_model.coef_.reshape(-1)
        b = self.best_model.intercept_[0]

        scales = [0.5, 0.7, 1.0, 1.5, 2.0]

        for i, file in enumerate(test_files):
            print(f'Test {i}/{len(test_files)}: {ntpath.basename(file)}')
            img = cv.imread(file, cv.IMREAD_GRAYSCALE)
            if img is None:
                continue

            original_h, original_w = img.shape[:2]
            image_dets, image_scores = [], []

            for scale in scales:
                scaled_w = int(original_w * scale)
                scaled_h = int(original_h * scale)
                if scaled_w < self.params.dim_window or scaled_h < self.params.dim_window:
                    continue

                scaled_img = cv.resize(img, (scaled_w, scaled_h))
                scaled_img = self.normalize_image(scaled_img)

                hog_map = hog(
                    scaled_img,
                    pixels_per_cell=(self.params.dim_hog_cell, self.params.dim_hog_cell),
                    cells_per_block=(2, 2),
                    feature_vector=False
                )

                n_blocks_y, n_blocks_x = hog_map.shape[:2]
                n_template = self.params.dim_window // self.params.dim_hog_cell - 1

                for y in range(n_blocks_y - n_template):
                    for x in range(n_blocks_x - n_template):
                        descr = hog_map[y:y+n_template, x:x+n_template].flatten()
                        score = np.dot(descr, w) + b

                        if score > self.params.threshold:
                            # Box in scaled image
                            x_min = x * self.params.dim_hog_cell
                            y_min = y * self.params.dim_hog_cell
                            x_max = x_min + self.params.dim_window
                            y_max = y_min + self.params.dim_window

                            # Scale back to original
                            scale_x = original_w / scaled_w
                            scale_y = original_h / scaled_h
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

        return np.array(detections), np.array(scores), np.array(file_names)

    # =========================================================
    # EVALUATE DETECTIONS
    # =========================================================
    def eval_detections(self, detections, scores, file_names):
        print("\n→ EVALUARE DETECTII")
        dets_per_image = defaultdict(list)
        for det, score, fname in zip(detections, scores, file_names):
            dets_per_image[fname].append((det, score))

        for fname in sorted(dets_per_image.keys()):
            dets = dets_per_image[fname]
            print(f"{fname}: {len(dets)} detections")

        print("\n→ EVALUARE FINALIZATA")
        print(f"Total imagini detectate: {len(dets_per_image)}")
        print(f"Total detectii: {len(detections)}")
