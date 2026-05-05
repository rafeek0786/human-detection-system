import torch
import cv2
import smtplib
from email.mime.text import MIMEText

# ---------------- EMAIL FUNCTION ----------------
def send_email():
    sender_email = "mohammedrafeek21993@gmail.com"
    receiver_email = "mr0512796@gmail.com"
    password = "dxvb ttvk bohp omiy"

    subject = "Alert: Human Detected"
    body = "A person has been detected by your Human Detection System."

    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = sender_email
    msg["To"] = receiver_email

    try:
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(sender_email, password)
        server.send_message(msg)
        server.quit()
        print("Email sent!")
    except Exception as e:
        print("Email error:", e)

# ------------------------------------------------

# Load YOLO model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

cap = cv2.VideoCapture(0)

email_sent = False  # prevent spam

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run detection
    results = model(frame)

    person_count = 0

    for *box, conf, cls in results.xyxy[0]:
        label = model.names[int(cls)]

        # Confidence filter
        if label == 'person' and conf > 0.5:
            person_count += 1

            x1, y1, x2, y2 = map(int, box)

            # Draw box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Label
            cv2.putText(frame, f'Person {conf:.2f}', (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

            # Send email only once
            if not email_sent:
                send_email()
                email_sent = True

    # Show count
    cv2.putText(frame, f'People Count: {person_count}', (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Human Detection System", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
