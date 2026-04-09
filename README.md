# 👁️ Eye Tracking Based Fatigue Detection System

## 📌 Overview
This project is a real-time fatigue detection system that uses eye tracking to monitor signs of drowsiness. It leverages computer vision and facial landmark detection to analyze eye movements and determine fatigue levels.

## 🚀 Features
- Real-time face and eye detection  
- Eye Aspect Ratio (EAR) based fatigue detection  
- Alerts for drowsiness detection  
- Lightweight and efficient implementation  

## 🛠️ Technologies Used
- Python  
- OpenCV  
- MediaPipe  
- NumPy  
- Pandas  

## 📂 Project Structure
```
├── main.py
├── face_landmarker.task
├── requirements.txt
```

## ⚙️ Installation
```bash
git clone <your-repo-link>
cd Eye-Tracking-Based-Fatigue-Detection-System
pip install -r requirements.txt
```

## ▶️ Usage
```bash
python main.py
```

## 🧠 How It Works
The system calculates the Eye Aspect Ratio (EAR) using facial landmarks detected by MediaPipe. If the EAR falls below a certain threshold for a sustained period, it indicates fatigue or drowsiness.

## 📌 Requirements
- Python 3.8+  
- Webcam  

## ⚠️ Limitations
- Works best in good lighting conditions  
- May be less accurate with glasses or occlusions  

## 📈 Future Improvements
- Add sound alerts  
- Improve accuracy with deep learning models  
- Mobile app integration  

## 🤝 Contributing
Feel free to fork this repository and contribute.

