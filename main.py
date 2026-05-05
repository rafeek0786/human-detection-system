import torch
import cv2

# Load YOLO model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run detection
    results = model(frame)

    person_count = 0

    for *box, conf, cls in results.xyxy[0]:
        label = model.names[int(cls)]

        # ✅ Confidence filter added here
        if label == 'person' and conf > 0.5:
            person_count += 1

            x1, y1, x2, y2 = map(int, box)

            # Draw box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Show label + confidence
            cv2.putText(frame, f'Person {conf:.2f}', (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    # Show people count
    cv2.putText(frame, f'People Count: {person_count}', (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Human Detection System", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
