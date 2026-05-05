import torch
import cv2
import smtplib
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
import time

# ---------------- EMAIL FUNCTION ----------------
def send_email_with_image(image):
    sender_email = "mohammedrafeek21993@gmail.com"
    receiver_email = "mr0512796@gmail.com"
    password = "dxvb ttvk bohp omiy"

    msg = MIMEMultipart()
    msg["Subject"] = "Alert: Human Detected"
    msg["From"] = sender_email
    msg["To"] = receiver_email

    body = MIMEText("Person detected. See attached image.")
    msg.attach(body)

    # Convert image to bytes
    _, img_encoded = cv2.imencode('.jpg', image)
    img_bytes = img_encoded.tobytes()

    image_part = MIMEImage(img_bytes, name="capture.jpg")
    msg.attach(image_part)

    try:
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(sender_email, password)
        server.send_message(msg)
        server.quit()
        print("Email with image sent!")
    except Exception as e:
        print("Email error:", e)

# ------------------------------------------------

# Load YOLO model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

cap = cv2.VideoCapture(0)

last_email_time = 0  # track last sent time

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)

    person_detected = False
    person_count = 0

    for *box, conf, cls in results.xyxy[0]:
        label = model.names[int(cls)]

        if label == 'person' and conf > 0.5:
            person_detected = True
            person_count += 1

            x1, y1, x2, y2 = map(int, box)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'Person {conf:.2f}', (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    # Show count
    cv2.putText(frame, f'People Count: {person_count}', (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # 📧 Send screenshot every 30 seconds if person detected
    current_time = time.time()

    if person_detected and (current_time - last_email_time > 30):
        send_email_with_image(frame)
        last_email_time = current_time

    cv2.imshow("Human Detection System", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
