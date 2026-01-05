import cv2
import os
import numpy as np
from retinaface import RetinaFace
from nudenet import NudeDetector


MIN_RESOLUTION = 360          # SOP minimum
MIN_FACE_SIZE = 120           # px
BLUR_REJECT = 35              # Laplacian variance threshold

SUPPORTED_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".webp", ".gif", ".avif"
}

# NudeNet detector (loaded once)
nsfw_detector = NudeDetector()


# UTILITY FUNCTIONS

def reject(reason, checks):
    return {
        "stage": 1,
        "result": "REJECT",
        "reason": reason,
        "checks": checks
    }

def pass_stage(checks):
    return {
        "stage": 1,
        "result": "PASS",
        "reason": None,
        "checks": checks
    }

def is_supported_format(image_path):
    ext = os.path.splitext(image_path.lower())[1]
    return ext in SUPPORTED_EXTENSIONS

def is_resolution_ok(img):
    h, w = img.shape[:2]
    return min(h, w) >= MIN_RESOLUTION

def blur_score(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def is_orientation_ok(landmarks):
    """
    Fast orientation sanity:
    Eyes must be above nose
    """
    le_y = landmarks["left_eye"][1]
    re_y = landmarks["right_eye"][1]
    nose_y = landmarks["nose"][1]

    if le_y > nose_y or re_y > nose_y:
        return False
    return True

def is_face_covered(landmarks):
    """
    If mouth landmarks missing → likely mask / full cover
    """
    return (
        "mouth_left" not in landmarks or
        "mouth_right" not in landmarks
    )


# NSFW / BARE BODY (STAGE-1)

def check_nsfw_stage1(image_path):
    """
    Stage-1 NSFW policy:
    - ANY nudity / bare body → REJECT
    - No suspend here (per requirement)
    """

    disallowed_classes = {
        # Explicit nudity
        "FEMALE_GENITALIA_EXPOSED",
        "MALE_GENITALIA_EXPOSED",
        "ANUS_EXPOSED",
        "FEMALE_BREAST_EXPOSED",

        # Bare / semi-nude / inappropriate
        "MALE_BREAST_EXPOSED",
        "FEMALE_BREAST_COVERED",
        "BELLY_EXPOSED",
        "BUTTOCKS_EXPOSED",
        "BUTTOCKS_COVERED",
        "UNDERWEAR",
        "SWIMWEAR"
    }

    detections = nsfw_detector.detect(image_path)

    for d in detections:
        if d["class"] in disallowed_classes and d["score"] > 0.6:
            return False, f"Disallowed content detected ({d['class']})"

    return True, None

# STAGE-1 MAIN VALIDATOR

def stage1_validate(image_path, photo_type="PRIMARY"):
    """
    photo_type: PRIMARY | SECONDARY
    """

    checks = {}

    # ---------------- IMAGE READ ----------------
    img = cv2.imread(image_path)
    if img is None:
        return reject("Invalid or unreadable image", checks)
    checks["image_read"] = "PASS"

    # ---------------- FORMAT ----------------
    if not is_supported_format(image_path):
        return reject("Unsupported image format", checks)
    checks["format"] = "PASS"

    # ---------------- RESOLUTION ----------------
    if not is_resolution_ok(img):
        return reject("Low resolution image", checks)
    checks["resolution"] = "PASS"

    # ---------------- FACE DETECTION ----------------
    faces = RetinaFace.detect_faces(image_path)
    if not faces:
        return reject("No face detected", checks)

    if photo_type == "PRIMARY" and len(faces) > 1:
        return reject("Group photo not allowed as primary photo", checks)

    checks["face_count"] = "PASS"

    # Pick first face (Stage-1 does not rank faces)
    face = list(faces.values())[0]
    area = face["facial_area"]
    landmarks = face["landmarks"]

    fw = area[2] - area[0]
    fh = area[3] - area[1]

    if min(fw, fh) < MIN_FACE_SIZE:
        return reject("Face too small or unclear", checks)

    checks["face_size"] = "PASS"

    # ---------------- BLUR ----------------
    blur = blur_score(img)
    if blur < BLUR_REJECT:
        return reject("Image is too blurry", checks)

    checks["blur"] = "PASS"

    # ---------------- ORIENTATION ----------------
    if not is_orientation_ok(landmarks):
        return reject("Improper image orientation", checks)

    checks["orientation"] = "PASS"

    # ---------------- MASK / FACE COVER ----------------
    if is_face_covered(landmarks):
        return reject("Face is covered or wearing a mask", checks)

    checks["face_cover"] = "PASS"

    # ---------------- NSFW / BARE BODY ----------------
    nsfw_ok, nsfw_reason = check_nsfw_stage1(image_path)
    if not nsfw_ok:
        return reject(nsfw_reason, checks)

    checks["nsfw"] = "PASS"

    # ---------------- FINAL ----------------
    return pass_stage(checks)

if __name__ == "__main__":
    result = stage1_validate("Fullface.jpeg", photo_type="PRIMARY")
    print(result)
