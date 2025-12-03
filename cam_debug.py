import cv2
import time

print("Script started ✅")

# Try both camera indexes
for cam_index in [0, 1]:
    print(f"Trying camera index {cam_index}...")
    cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)

    # Give the camera some time to initialize
    time.sleep(1)

    if not cap.isOpened():
        print(f" Camera {cam_index} could not be opened.")
        cap.release()
        continue

    print(f"✅ Camera {cam_index} opened successfully! Window should appear now.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Frame not received, stopping...")
            break

        frame = cv2.flip(frame, 1)
        cv2.imshow(f"Webcam Test (cam {cam_index})", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Q pressed. Exiting.")
            cap.release()
            cv2.destroyAllWindows()
            exit(0)

    cap.release()

print("😕 None of the camera indexes worked.")
cv2.destroyAllWindows()
