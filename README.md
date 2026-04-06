🧠 Student Engagement Detection System
A deep learning–based computer vision system to classify student engagement levels (e.g., engaged, bored, distracted) using facial and behavioral cues from video data.

🚀 Overview

This project leverages the DAiSEE Dataset to train a CNN-based model for multi-class engagement classification. It includes end-to-end processing from data preprocessing to deployment via a web application.

🛠️ Tech Stack
Languages: Python
Frameworks/Libraries: PyTorch, OpenCV, NumPy, Pandas
Model: EfficientNetB2 (CNN)
Deployment: Streamlit

📊 Dataset
Dataset: DAiSEE (Dataset for Affective States in E-Environments)
Type: Video-based, multi-class classification
Classes: Engaged, Bored, Confused, Frustrated

⚙️ Methodology
Extracted frames from video data for training
Applied preprocessing techniques (normalization, resizing)
Handled class imbalance using appropriate sampling/weighting
Trained an EfficientNetB2-based CNN model for classification
Evaluated model performance on validation and test datasets

📈 Results
Validation Accuracy: 74.75%
Test Accuracy: 72.63%

These results are consistent with realistic performance benchmarks for the DAiSEE dataset.

🌐 Deployment

Developed an interactive Streamlit web application to:

Upload input (image/video frame)
Predict engagement level in real-time
Display model output with a simple UI

🧩 Project Structure
├── src/
│   ├── dataset.py
│   ├── model_v2.py
│   ├── train_v2.py
├── app.py              # Streamlit app
├── requirements.txt
├── README.md

▶️ How to Run
1. Clone the repository
    git clone https://github.com/shivani51-06/your-repo-name.git
    cd your-repo-name
2. Install dependencies
    pip install -r requirements.txt
3. Run Streamlit app
    streamlit run app.py

📌 Future Improvements
Add confusion matrix and class-wise precision/recall
Improve performance with hyperparameter tuning
Extend to real-time webcam-based engagement detection

🙌 Key Takeaways
Built an end-to-end deep learning + deployment pipeline
Gained hands-on experience in computer vision and model evaluation
Demonstrated ability to translate ML models into usable applications
