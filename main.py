import torch
import cv2
import winsound  # for alert sound (Windows)

# Load YOLO model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)

    person_count = 0

    for *box, conf, cls in results.xyxy[0]:
        label = model.names[int(cls)]

        if label == 'person':
            person_count += 1

            x1, y1, x2, y2 = map(int, box)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'Person {conf:.2f}', (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    # Show count
    cv2.putText(frame, f'People Count: {person_count}', (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # 🚨 ALERT condition
    if person_count > 1:
        cv2.putText(frame, "ALERT: Multiple People!", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

        # Beep sound
        winsound.Beep(1000, 500)  # frequency, duration

    cv2.imshow("Human Detection System", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
