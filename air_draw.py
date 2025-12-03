import cv2
import mediapipe as mp
import numpy as np

print("Air Draw start ho raha hai ✅")

# Camera open
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    print("❌ Camera open nahi hua.")
    exit()

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_draw = mp.solutions.drawing_utils

canvas = None
drawing = False
prev_point = None  # last fingertip point

while True:
    ret, frame = cap.read()
    if not ret:
        print("Frame nahi mil raha, exit.")
        break

    frame = cv2.flip(frame, 1)
    h, w, c = frame.shape

    if canvas is None:
        canvas = np.zeros_like(frame)

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            lm = handLms.landmark

            # Index fingertip (8) & thumb tip (4)
            x_index = int(lm[8].x * w)
            y_index = int(lm[8].y * h)
            x_thumb = int(lm[4].x * w)
            y_thumb = int(lm[4].y * h)

            # Distance between thumb & index
            dist = ((x_index - x_thumb)**2 + (y_index - y_thumb)**2)**0.5

            # Pinch = draw mode
            if dist < 40:
                drawing = True
            else:
                drawing = False
                prev_point = None  # line break

            # Fingertip ko screen pe highlight karo
            cv2.circle(frame, (x_index, y_index), 10, (0, 255, 0), -1)

            if drawing:
                if prev_point is not None:
                    # Canvas pe line draw
                    cv2.line(canvas, prev_point, (x_index, y_index), (255, 255, 255), 6)
                prev_point = (x_index, y_index)

            mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)

    # Canvas + live camera merge
    gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)

    frame_bg = cv2.bitwise_and(frame, frame, mask=mask_inv)
    canvas_fg = cv2.bitwise_and(canvas, canvas, mask=mask)

    out = cv2.add(frame_bg, canvas_fg)

    cv2.putText(out, "Pinch (thumb+index) to draw | C=clear | Q=quit",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imshow("Air Draw", out)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        print("Q dabaya, exit.")
        break
    elif key == ord('c'):
        canvas = np.zeros_like(frame)
        prev_point = None
        print("Canvas clear.")

cap.release()
hands.close()
cv2.destroyAllWindows()
print("Program khatam ✅")
