# 🏎️ AI-Powered Traffic Vision & Speed Analytics
### Real-time Vehicle Detection & Multi-Lane Tracking

This project is a comprehensive **Computer Vision** solution developed in **Python 3.13**[cite: 3, 5]. It leverages **YOLO** architecture to detect, classify, and track vehicles across multiple highway lanes while estimating their real-time speed.

## 📺 Project Demo
<p align="center">
  <media/video15_annotated-ezgif.com-video-to-gif-converter.gif>
  <img src="media/output_demo.gif" width="800" alt="Traffic Analysis Demo">
</p>

## 🎯 Core Functionalities
* **Object Detection:** Real-time identification of Cars, Vans, and Trucks using `best.pt` weights[cite: 5].
* **Speed Estimation:** Mathematical calculation of vehicle velocity based on frame displacement.
* **Lane Analysis:** Statistical distribution of speeds per lane using **Pandas** & **Matplotlib**[cite: 6].
* **Data Logging:** Every detection is logged into `final_results.csv` for post-traffic engineering analysis[cite: 6].

## ⚙️ Tech Stack
* **Language:** Python 3.13[cite: 5].
* **AI Model:** YOLO (Ultralytics)[cite: 5].
* **Libraries:** OpenCV, NumPy, Pandas, Matplotlib[cite: 6].
* **Environment:** Structured and developed within **PyCharm**[cite: 4, 6].

## 📊 Speed Distribution Analysis
The system visualizes traffic flow patterns per lane:
<p align="center">
  <img src="media/speed_plot.png" width="600" alt="Speed Distribution Plot">
</p>

## 📂 Project Structure
* `video_tracking_speed_lane.py`: Core processing engine.
* `plot_speed_distribution_by_lane.py`: Analytics & Visualization script[cite: 6].
* `best.pt`: Trained model weights[cite: 5].
* `Screenshot (34).jpg`: Development environment setup[cite: 6].

---
*Developed as a high-performance case study for Computer Vision & Data Science.*
