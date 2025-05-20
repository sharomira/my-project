# Neural Vision-Based Road Safety System

## Overview

The **Neural Vision-Based Road Safety System** is an AI-powered driver monitoring application that enhances road safety by detecting driver fatigue, distraction, and risky behaviors in real-time using computer vision and deep learning techniques.

This system integrates multiple components into a unified GUI:

* **Drowsiness Detection** using Eye Aspect Ratio (EAR) and MobileNetV2
* **Yawning Detection** using MobileNetV2 and Mediapipe
* **Phone Usage Detection** using YOLOv8
* **Head Pose Estimation** (Upcoming Feature)

## Features

* ✅ Real-time drowsiness detection via webcam
* ✅ Yawning detection using pre-trained CNN and face landmarks
* ✅ Phone usage detection with YOLOv8 object detection
* 🛠️ Planned support for head pose estimation for driver distraction analysis
* 🖥️ Simple and intuitive Tkinter-based GUI

## Tech Stack

* **Languages:** Python 3.x
* **Libraries:** OpenCV, Mediapipe, TensorFlow/Keras, Ultralytics YOLOv8, Tkinter
* **Hardware:** CPU (Google Colab used for model training)
* **Data Sources:**

  * Drowsiness: Custom eye state dataset
  * Yawning: YawDD Dataset
  * Phone Usage: State Farm Distracted Driver Detection Dataset
  * Head Pose: NTHU-DDD and AFLW (planned)

## Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/your-username/neural-vision-road-safety.git
   cd neural-vision-road-safety
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Download pretrained models and place them in the `models/` directory**

   * MobileNetV2 for drowsiness & yawning detection
   * YOLOv8 weights for phone usage detection

4. **Prepare datasets**

   * Mount Google Drive if using Colab
   * Ensure your datasets follow the correct folder structure (details in `data/README.md`)

## Usage

Run the main GUI-based application:

```bash
python main.py
```

Use the interface to start real-time driver monitoring and get alerts when fatigue or distraction is detected.

## Folder Structure

```
neural-vision-road-safety/
├── models/                # Pretrained models
├── datasets/              # Processed datasets
├── utils/                 # Utility functions
├── gui/                   # GUI components (Tkinter)
├── yolo/                  # YOLOv8 configuration and scripts
├── main.py                # Entry point for GUI
├── requirements.txt
└── README.md
```

## License

This project is open-source and licensed under the MIT License. See `LICENSE` for more details.

## Acknowledgments

* [Kaggle - State Farm Distracted Driver Detection](https://www.kaggle.com/competitions/state-farm-distracted-driver-detection)
* [YawDD Dataset](https://www.idiap.ch/en/dataset/yawdd)
* [NTHU Drowsy Driver Detection Dataset](https://nthu-en.web.nthu.edu.tw)
* Ultralytics YOLOv8: [https://github.com/ultralytics/ultralytics](https://github.com/ultralytics/ultralytics)
