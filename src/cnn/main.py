import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import threading
from cnn import predict_and_annotate


class ImageVideoProcessor(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("Image and Video Processor")
        self.create_widgets()

        self.display_width = 640  # Set fixed display width
        self.display_height = 480  # Set fixed display height
        self.current_image = None
        self.current_video_path = None
        self.video_thread = None

    def create_widgets(self):
        # Buttons for uploading image and video
        upload_image_button = tk.Button(
            self, text="Upload Image", command=self.upload_image
        )
        upload_image_button.pack()

        upload_video_button = tk.Button(
            self, text="Upload Video", command=self.upload_video
        )
        upload_video_button.pack()

        # Label for displaying images and videos
        self.image_label = tk.Label(self)
        self.image_label.pack()

        # Buttons for processing image and video
        process_image_button = tk.Button(
            self, text="Process Image", command=self.process_image
        )
        process_image_button.pack()

        process_video_button = tk.Button(
            self, text="Process Video", command=self.process_video
        )
        process_video_button.pack()

    def upload_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")]
        )
        if file_path:
            self.current_image = Image.open(file_path)
            self.display_image(self.current_image)

    def upload_video(self):
        self.current_video_path = filedialog.askopenfilename(
            filetypes=[("Video Files", "*.mp4;*.avi")]
        )

    def process_image(self):
        if self.current_image:
            # Image processing logic here
            processed_image = predict_and_annotate(
                self.current_image
            )  # Replace with actual processing
            self.display_image(
                Image.fromarray(
                    cv2.cvtColor(processed_image, cv2.COLOR_RGB2BGR)
                ).convert("CMYK")
            )
        else:
            messagebox.showinfo("No Image", "Please upload an image first.")

    def process_video(self):
        if self.current_video_path:
            self.play_video()
            # Video processing logic here
            messagebox.showinfo(
                "Process Video", "Video processing not implemented yet."
            )
        else:
            messagebox.showinfo("No Video", "Please upload a video first.")

    def display_image(self, image):
        # Resize image to fixed display size
        image = image.resize(
            (self.display_width, self.display_height), Image.Resampling.LANCZOS
        )
        photo = ImageTk.PhotoImage(image)
        self.image_label.config(image=photo)
        self.image_label.image = photo  # Keep a reference.

    def play_video(self):
        if self.video_thread and self.video_thread.is_alive():
            return  # Avoid multiple threads for video playback

        self.video_thread = threading.Thread(target=self.video_loop)
        self.video_thread.daemon = True
        self.video_thread.start()

    def video_loop(self):
        cap = cv2.VideoCapture(self.current_video_path)
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                frame = predict_and_annotate(frame)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = Image.fromarray(frame)
                frame = frame.resize(
                    (self.display_width, self.display_height), Image.Resampling.LANCZOS
                )
                photo = ImageTk.PhotoImage(image=frame)

                self.image_label.config(image=photo)
                self.image_label.image = photo
            else:
                break
            self.image_label.update()

        cap.release()


app = ImageVideoProcessor()
app.mainloop()
