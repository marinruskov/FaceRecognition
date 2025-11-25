import cv2
import os
import numpy as np
import pickle
import customtkinter as ctk
from PIL import Image

MODEL_PATH = "model.yml"
LABELS_PATH = "labels.pkl"
DATASET_DIR = "DATASET"
FACE_SIZE = (200, 200)
CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
confidence_threshold = 80

# Train LBPH model
def train():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

    faces = []
    labels = []

    for filename in os.listdir(DATASET_DIR):
        if not filename.lower().endswith(".jpg"):
            continue

        path = os.path.join(DATASET_DIR, filename)
        img = cv2.imread(path)

        if img is None:
            continue

        # Extract label from filename
        # Split by "_" and take second part before "."
        try:
            label = int(filename.split("_")[1].split(".")[0])
        except:
            print("Invalid filename:", filename)
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Try to detect a face
        rects = face_cascade.detectMultiScale(gray, 1.1, 4)

        # If face detected, extract it
        if len(rects) > 0:
            for (x, y, w, h) in rects:
                face = gray[y:y+h, x:x+w]
                face = cv2.resize(face, FACE_SIZE)
                faces.append(face)
                labels.append(label)
        else:
            # Use entire image if no face detected
            face = cv2.resize(gray, FACE_SIZE)
            faces.append(face)
            labels.append(label)

    if len(faces) < 2:
        raise ValueError("Not enough data to train LBPH.")

    recognizer.train(np.array(faces), np.array(labels))
    recognizer.write(MODEL_PATH)

    # Save mapping (label → person)
    with open(LABELS_PATH, "wb") as f:
        pickle.dump({"ids": list(set(labels))}, f)

    print("Training complete. Samples:", len(faces))

# Recognize from camera
def recognize_camera(parent_window):
    # Load model
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(MODEL_PATH)

    # Load labels
    with open(LABELS_PATH, "rb") as f:
        label_ids = pickle.load(f)

    # Clean label mappings
    clean_label_ids = {}
    for k, v in label_ids.items():
        clean_label_ids[k] = int(v[0] if isinstance(v, list) else v)

    id_to_name = {v: k for k, v in clean_label_ids.items()}

    # Load cascade
    face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

    # Start webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened() and not cap.read()[0]:
        print("Error: Could not open webcam.")
        return

    # Create a CTK window for video
    win = ctk.CTkToplevel(parent_window)
    win.title("Live Face Recognition")
    win.geometry("800x600")

    video_label = ctk.CTkLabel(win, text="")
    video_label.pack(pady=10)

    info_frame = ctk.CTkFrame(win)
    info_frame.pack(pady=5)

    def update_frame():
        ret, frame = cap.read()
        if not ret:
            win.after(10, update_frame)
            return

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        face_infos = []

        # Recognize faces
        for (x, y, w, h) in faces:
            face_roi = gray[y:y+h, x:x+w]
            face_roi = cv2.resize(face_roi, FACE_SIZE)

            label_id, confidence = recognizer.predict(face_roi)

            if confidence > confidence_threshold:
                name = "Unknown"
            else:
                name = id_to_name.get(label_id, f"Person_{label_id}")

            face_infos.append((name, confidence))

            # Draw rectangle + label
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
            cv2.putText(frame, f"{name} ({int(confidence)})", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # Convert to CTkImage
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        pil_img = pil_img.resize((760, 540))

        tk_img = ctk.CTkImage(light_image=pil_img, dark_image=pil_img, size=(760, 540))
        video_label.configure(image=tk_img)
        video_label.image = tk_img

        # Update face info labels
        for widget in info_frame.winfo_children():
            widget.destroy()
        for i, (name, conf) in enumerate(face_infos):
            lbl = ctk.CTkLabel(info_frame, text=f"Face {i+1}: {name} (Confidence: {(100 - (conf / 80) * 100)}%")
            lbl.pack()

        win.after(10, update_frame)

    update_frame()

# Recognize from image file
def recognize_image(image_path):
    # Load model
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(MODEL_PATH)

    # Load labels
    with open(LABELS_PATH, "rb") as f:
        label_ids = pickle.load(f)

    # Clean labels
    clean_label_ids = {}
    for k, v in label_ids.items():
        if isinstance(v, list) and len(v) > 0:
            clean_label_ids[k] = int(v[0])
        else:
            clean_label_ids[k] = int(v)

    # Invert mapping id -> name
    id_to_name = {v: k for k, v in clean_label_ids.items()}

    # Load cascade
    face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

    # Load image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Could not read image: " + image_path)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect ALL faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    if len(faces) == 0:
        return {"success": False, "error": "No face detected"}

    results = []

    # Recognize each face
    for (x, y, w, h) in faces:
        face_roi = gray[y:y+h, x:x+w]
        face_roi = cv2.resize(face_roi, FACE_SIZE)

        label_id, confidence = recognizer.predict(face_roi)
        print(label_id, confidence)

        if confidence > confidence_threshold:
            name = "Unknown"
        else:
            name = id_to_name.get(label_id, f"Person_{label_id}")

        results.append({
            "name": name,
            "confidence": (100 - (confidence / 80) * 100),
            "box": [int(x), int(y), int(w), int(h)]
        })

    return {
        "success": True,
        "results": results,
        "detections": faces.tolist()
    }

# Test functions
if __name__ == "__main__":
    print("1 - Train")
    print("2 - Recognize Camera")
    print("3 - Recognize from Image")

    option = input("Enter option: ")

    if option == "1":
        train()
    elif option == "2":
        recognize_camera()
    elif option == "3":
        print(recognize_image("test.jpg"))
    else:
        print("Invalid option.")
