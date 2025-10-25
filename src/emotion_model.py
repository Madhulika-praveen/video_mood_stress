from fer import FER
import cv2

detector = FER(mtcnn=False)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    result = detector.detect_emotions(frame)
    if result:
        emotions = result[0]["emotions"]
        dominant = max(emotions, key=emotions.get)
        stress = (emotions["angry"] + emotions["sad"] + emotions["fear"]) * 100 / 3  # Simple stress formula
        cv2.putText(frame, f"Emotion: {dominant}", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        cv2.putText(frame, f"Stress: {int(stress)}%", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
    
        print(f"Detected: {dominant}, Stress: {int(stress)}%, Raw: {emotions}")

    cv2.imshow("Mood & Stress Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
