# Human Detection System 👁️📧

## 📌 About the Project

This is a simple real-time human detection project that I created using Python.
The system uses my laptop webcam to detect people and draw a green box around them.

I also added an extra feature where the system sends an email with a screenshot whenever a person is detected (every 30 seconds).

---

## 🚀 Features

* Detects humans in real-time
* Uses laptop webcam
* Draws bounding box around detected person
* Shows number of people detected
* Filters low-confidence detections
* Sends email with image when a person is detected

---

## 🛠️ Technologies Used

* Python
* OpenCV
* YOLOv5
* PyTorch
* SMTP (for sending email)

---

## ▶️ How to Run

1. Install all required libraries:

   ```
   pip install -r requirements.txt
   ```

2. Run the program:

   ```
   python main.py
   ```

---

## ⚙️ Email Setup

Before running, update your email details in the code:

```id="setup"
sender_email = "your_email@gmail.com"
receiver_email = "your_email@gmail.com"
password = "your_app_password"
```

Note: Use Gmail App Password (not your normal password).

---

## 📷 Output

* Webcam will open
* Humans will be detected with a green box
* People count will be shown
* Email will be sent with screenshot when a person is detected

---

## 📚 What I Learned

* Basics of computer vision
* How YOLO works for object detection
* Real-time video processing using OpenCV
* Sending emails using Python

---

## 🔮 Future Improvements

* Add sound alert
* Improve detection accuracy
* Convert into web application

---

## 👤 Author

Mohammed Rafeek
