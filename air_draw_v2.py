import cv2
import mediapipe as mp
import numpy as np
import time
import random

print("AirDraw v2 is starting... ✅")

# Open camera
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    print(" Error: Could not open camera.")
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

particles = []  # particle list: each = [x, y, vx, vy, life]

def spawn_particles(x, y, count=8):
    """Create glowing particles around the fingertip."""
    for _ in range(count):
        angle = random.uniform(0, 2 * np.pi)
        speed = random.uniform(1, 3)
        vx = np.cos(angle) * speed
        vy = np.sin(angle) * speed
        life = random.randint(15, 30)  # frames
        particles.append([float(x), float(y), vx, vy, life])

while True:
    ret, frame = cap.read()
    if not ret:
        print(" Error: Frame not received. Exiting.")
        break

    frame = cv2.flip(frame, 1)
    h, w, c = frame.shape

    if canvas is None:
        canvas = np.zeros_like(frame)

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    fingertip = None  # (x, y)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            lm = handLms.landmark

            # Index fingertip (8) & thumb tip (4)
            x_index = int(lm[8].x * w)
            y_index = int(lm[8].y * h)
            x_thumb = int(lm[4].x * w)
            y_thumb = int(lm[4].y * h)

            fingertip = (x_index, y_index)

            # Distance between thumb & index
            dist = ((x_index - x_thumb)**2 + (y_index - y_thumb)**2)**0.5

            # Pinch = draw mode
            if dist < 40:
                drawing = True
            else:
                drawing = False
                prev_point = None  # line break

            if drawing:
                # Draw on canvas
                if prev_point is not None:
                    cv2.line(canvas, prev_point, (x_index, y_index),
                             (255, 255, 255), 6)
                prev_point = (x_index, y_index)

                # Create particle effects
                spawn_particles(x_index, y_index, count=6)

            mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)

    # Merge canvas with live camera feed
    gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)

    frame_bg = cv2.bitwise_and(frame, frame, mask=mask_inv)
    canvas_fg = cv2.bitwise_and(canvas, canvas, mask=mask)

    out = cv2.add(frame_bg, canvas_fg)

    # LASER EFFECT on fingertip
    if fingertip is not None:
        x_f, y_f = fingertip
        # inner bright core
        cv2.circle(out, (x_f, y_f), 10, (0, 255, 255), -1)
        # outer glow rings
        cv2.circle(out, (x_f, y_f), 18, (0, 200, 255), 2)
        cv2.circle(out, (x_f, y_f), 26, (0, 150, 255), 2)

    # PARTICLE UPDATE + DRAW
    new_particles = []
    for p in particles:
        x, y, vx, vy, life = p
        x += vx
        y += vy
        life -= 1

        if life > 0:
            new_particles.append([x, y, vx, vy, life])
            alpha = int(80 + 175 * (life / 30.0))
            alpha = max(50, min(255, alpha))
            color = (0, alpha, 255)  # glowing orange-pink
            cv2.circle(out, (int(x), int(y)), 4, color, -1)

    particles = new_particles

    cv2.putText(out, "Pinch=draw | C=clear | S=save | Q=quit",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 255), 2)

    cv2.imshow("AirDraw v2", out)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        print("Q pressed. Exiting program.")
        break
    elif key == ord('c'):
        canvas = np.zeros_like(frame)
        particles = []
        prev_point = None
        print("Canvas cleared.")
    elif key == ord('s'):
        filename = f"drawing_{int(time.time())}.png"
        cv2.imwrite(filename, canvas)
        print(f"Image saved as: {filename}")

cap.release()
hands.close()
cv2.destroyAllWindows()
print("Program finished. ")
