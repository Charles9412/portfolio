# SafeDriveAI/src/data_utils.py

import os
from pathlib import Path
from PIL import Image
import cv2
import mediapipe as mp
import random
import shutil
import numpy as np

quad_size = 364  # still used for cropping quadrants


def crop_face_from_pil(pil_img, padding=0.2):
    """
    Detects and crops face from a PIL image using Mediapipe.
    Returns cropped face image (PIL.Image) or None if no face found.
    """
    img = np.array(pil_img.convert("RGB"))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    h, w, _ = img.shape

    mp_face_detection = mp.solutions.face_detection
    with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detector:
        results = face_detector.process(img)

        if results.detections:
            box = results.detections[0].location_data.relative_bounding_box
            xmin = int(box.xmin * w)
            ymin = int(box.ymin * h)
            box_w = int(box.width * w)
            box_h = int(box.height * h)

            # Padding
            pad_w = int(box_w * padding)
            pad_h = int(box_h * padding)

            x1 = max(xmin - pad_w, 0)
            y1 = max(ymin - pad_h, 0)
            x2 = min(xmin + box_w + pad_w, w)
            y2 = min(ymin + box_h + pad_h, h)

            face_crop = img[y1:y2, x1:x2]
            return Image.fromarray(face_crop)
        else:
            return None


def split_and_label_images(raw_dir, sober_dir, drunk_dir, face_size=(224, 224)):
    raw_dir = Path(raw_dir)
    sober_dir = Path(sober_dir)
    drunk_dir = Path(drunk_dir)

    image_files = [f for f in raw_dir.glob(
        "*") if f.is_file() and f.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]]

    if not image_files:
        print("No images found in raw directory.")
        return

    sober_dir.mkdir(parents=True, exist_ok=True)
    drunk_dir.mkdir(parents=True, exist_ok=True)

    for img_path in image_files:
        full_img = Image.open(img_path).convert("RGB")
        base_name = img_path.stem

        # Extract quadrants
        quadrants = {
            # Top-left
            "sober": full_img.crop((0, 0, quad_size, quad_size)),
            # Top-right
            "drunk_1": full_img.crop((quad_size, 0, 728, quad_size)),
            # Bottom-left
            "drunk_2": full_img.crop((0, quad_size, quad_size, 728)),
            # Bottom-right
            "drunk_3": full_img.crop((quad_size, quad_size, 728, 728))
        }

        # Face crop each quadrant
        for key, quadrant in quadrants.items():
            face = crop_face_from_pil(quadrant)
            if face:
                face = face.resize(face_size)
                if key == "sober":
                    face.save(sober_dir / f"{base_name}_sober.jpg")
                else:
                    level = key.split("_")[1]
                    face.save(drunk_dir / f"{base_name}_drunk{level}.jpg")
            else:
                print(
                    f"⚠️ No face detected in {img_path.name} [{key}] — Skipping.")

        print(f"✅ Processed {img_path.name}")


def process_human_sober_faces(input_dir, output_dir, face_size=(224, 224)):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_files = [f for f in input_dir.glob(
        "*") if f.is_file() and f.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]]

    if not image_files:
        print("⚠️ No images found.")
        return

    for img_path in image_files:
        try:
            pil_img = Image.open(img_path).convert("RGB")
            face = crop_face_from_pil(pil_img)  # same helper as before

            if face:
                face = face.resize(face_size)
                face.save(output_dir / f"{img_path.stem}_face.jpg")
            else:
                print(f"❌ No face found in {img_path.name}")
        except Exception as e:
            print(f"❌ Error processing {img_path.name}: {e}")

    print(f"✅ Finished processing {len(image_files)} images.")


def sample_sober_subset(input_dir, output_dir, n_samples):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_images = list(input_dir.glob("*"))
    if len(all_images) < n_samples:
        raise ValueError(
            f"Not enough images to sample: requested {n_samples}, found {len(all_images)}")

    sampled = random.sample(all_images, n_samples)

    for img in sampled:
        shutil.copy(img, output_dir / img.name)

    print(f"✅ Sampled {n_samples} images to {output_dir}")
