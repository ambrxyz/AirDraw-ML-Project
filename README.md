# ✨ AirDraw ML Project  
### Real-time Hand-Tracking + Air Drawing using Python, OpenCV, and MediaPipe  

This project lets you draw **in the air using your fingers** — just like Iron Man HUD style.  
It uses **real-time hand landmark detection** from MediaPipe and overlays glowing lines, particles, and laser effects using OpenCV.

---

## 🚀 Features  
### **AirDraw v1**
- Real-time hand tracking  
- Pinch gesture (thumb + index) for drawing  
- Simple white brush  
- Smooth movement tracking  

### **AirDraw v2**
- Laser beam fingertip effect  
- Particle trail animations  
- Save drawing as PNG (`S` key)  
- Improved line smoothing  

### **AirDraw v3**
- Multiple brush colors  
  - `1` → White  
  - `2` → Blue  
  - `3` → Green  
  - `4` → Pink  
- Particle colors match brush  
- Eraser mode (`E` key)  
- Clear canvas (`C` key)  
- Save canvas (`S` key)  
- Better UI + text overlay  

---

## 🎮 Controls  
| Action | Key / Gesture |
|--------|----------------|
| Start Drawing | Pinch (Thumb + Index Finger) |
| Switch Brush Color | 1 / 2 / 3 / 4 |
| Toggle Eraser | E |
| Clear Canvas | C |
| Save Drawing | S |
| Quit Program | Q |

---

## 🧠 Tech Stack  
- **Python 3.10+**  
- **OpenCV** – real-time video & drawing  
- **MediaPipe Hands** – hand landmark detection  
- **NumPy** – arrays & processing  

---

## 🛠 Installation

Clone this repository:
```bash
git clone https://github.com/ambrxyz/AirDraw-ML-Project.git
```

Install required libraries:
```bash
pip install opencv-python mediapipe numpy
```

Run any version:
```bash
python air_draw.py
python air_draw_v2.py
python air_draw_v3.py
```

---

## 📸 Screenshots  
(You can add images here later)
```
![Demo](demo.png)
```

---

## 💡 Future Improvements  
- Background removal (Green screen mode)  
- Add gesture-based undo  
- Web version using Mediapipe JS  
- Export video timelapse of drawing  
- Add face + hand combined AR effects  

---

## 👤 Author  
**Amber (ambrxyz)**  
Real-time AI/ML & AR enthusiast 🔥  

---

## ⭐ Show Some Love  
If you like this project, please ⭐ the repo!  
