import customtkinter as ctk
from tkinter import filedialog
from PIL import Image
import FaceRecognition
import cv2

def upload_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.png")])
    if not file_path:
        return

    image = cv2.imread(file_path)

    result = FaceRecognition.recognize_image(file_path)

    if not result.get("success", False):
        print("Error:", result.get("error", "Unknown error"))
        return

    # Multi-face results format
    detections = result.get("detections", [])
    face_results = result.get("results", [])

    # Extract names and confidences for each face
    names = [fr["name"] for fr in face_results]
    confidences = [fr["confidence"] for fr in face_results]

    # Display all data
    display_image(image, detections, names, confidences)

def display_image(image, detections, names, confidences):
    # Open a new window
    win = ctk.CTkToplevel()
    win.title("Image Preview")
    win.geometry("600x600")

    # Convert BGR → RGB
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Draw rectangles
    for (x, y, w, h) in detections:
        cv2.rectangle(img_rgb, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Convert to PIL image
    pil_img = Image.fromarray(img_rgb)

    # Create CTkImage
    ctk_img = ctk.CTkImage(
        light_image=pil_img,
        dark_image=pil_img,
        size=(450, 450)
    )

    # Put image in label
    lbl = ctk.CTkLabel(win, image=ctk_img, text="")
    lbl.image = ctk_img
    lbl.pack(pady=10)


    ctk_label = ctk.CTkLabel(win, text="Detected Face:")
    ctk_label.pack(pady=5)
    for i, (name, conf) in enumerate(zip(names, confidences)):
        face_info = f"Face {i + 1}: {name} (Confidence: {conf:.2f})"
        face_label = ctk.CTkLabel(win, text=face_info)
        face_label.pack()


# GUI setup
app = ctk.CTk()
app.title("Face Recognition App")
app.geometry("200x400")

label = ctk.CTkLabel(app, text="Face Recognition")
label.pack(pady=20)

upload_button = ctk.CTkButton(app, text="Upload Image", command=lambda: upload_image())
upload_button.pack(pady=10)

webcam_button = ctk.CTkButton(app, text="Use Webcam", command=lambda: FaceRecognition.recognize_camera(app))
webcam_button.pack(pady=10)

train_button = ctk.CTkButton(app, text="Train Model", command=lambda: FaceRecognition.train())
train_button.pack(pady=10)

app.mainloop()