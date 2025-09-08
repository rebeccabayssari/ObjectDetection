# Android Application for Object Localization and Recognition in a Video Stream

##  Description
This project implements a **real-time object detection system** using **YOLOv8** and integrates it into an **Android application**.  
The AI model is trained on a custom dataset and exported in a mobile-friendly format (`.tflite`) to run efficiently on smartphones.  
The Android app processes live video streams from the camera and displays bounding boxes around detected objects.

---

##  Project Structure
- **`Object_detection_android/Object_detection/`** : Android application source code (Kotlin).  
- **`train_export_yolov8_model.ipynb`** : Jupyter notebook for training and exporting the YOLOv8 model.  
- **`data.yaml`** : Dataset configuration file (paths, classes, structure).  
- **`deepL report (1).pdf`** : Project report (detailed documentation and methodology).  

---

##  Workflow
1. **Dataset preparation**  
   - Annotated dataset in YOLO format (images + labels).  
   - Configuration defined in `data.yaml`.

2. **Model training (YOLOv8)**  
   - Train YOLOv8 on a custom dataset.  
   - Export trained model to `.pt` format.  
   - Convert `.pt` → `.tflite` for mobile deployment.  

3. **Android integration**  
   - Load `.tflite` model in the Android app.  
   - Process video frames in real-time.  
   - Draw bounding boxes and class labels on the camera feed.  

---

##  Requirements
- **Python (for training)**  
  - `ultralytics` (YOLOv8)  
  - `torch`, `torchvision`  
  - `opencv-python`  
  - `pyyaml`  

- **Android (for deployment)**  
  - Android Studio  
  - Kotlin  
  - TensorFlow Lite  

---

##  Results
- Model trained on a **13-class flower dataset** (3,343 images).  
- Converted successfully to `.tflite` format.  
- Real-time detection achieved on Android with bounding boxes and labels.  

---

##  Future Improvements
- Add saving of detections.  
- Allow confidence threshold adjustments.  
- Extend dataset with more object categories for better generalization.  

---

## 👩‍💻 Author
Developed as part of a project on **AI and Mobile Deployment for Object Detection**.
