import cv2
import time

print("Script start ho gaya ✅")

# Try camera index 0 and 1
for cam_index in [0, 1]:
    print(f"Camera index {cam_index} try kar rahe hain...")
    cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)

    # Thoda wait karo camera open hone ke liye
    time.sleep(1)

    if not cap.isOpened():
        print(f"❌ Camera {cam_index} open nahi hua.")
        cap.release()
        continue

    print(f"✅ Camera {cam_index} open ho gaya! Window ab open honi chahiye.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Frame nahi mil raha, breaking...")
            break

        frame = cv2.flip(frame, 1)
        cv2.imshow(f"Webcam Test (cam {cam_index})", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Q dabaya, exit ho rahe hain.")
            cap.release()
            cv2.destroyAllWindows()
            exit(0)

    cap.release()

print("😕 Koi bhi camera index kaam nahi kar raha.")
cv2.destroyAllWindows()
