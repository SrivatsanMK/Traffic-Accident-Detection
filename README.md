# Traffic Accident Detection 🚦🚗

This project is a machine learning/computer vision application designed to detect traffic accidents from video feeds or images. It includes a Jupyter Notebook for training the classification model and a Python application to run the detection system.

## 📂 Project Structure

* `app.py`: The main application script to run the accident detection.
* `accident-classification.ipynb`: Jupyter Notebook containing the code for data preprocessing, model building, and training.
* `model/`: Directory containing the saved, pre-trained model files (e.g., `model.json`).
* `videos/`: Directory to store sample videos for testing the model.
* `requirements.txt`: List of Python dependencies required to run the project.

---

## 🚀 Step-by-Step Installation

**1. Clone the repository**
```bash
git clone https://github.com/SrivatsanMK/Traffic-Accident-Detection.git
cd Traffic-Accident-Detection
```

**2. Set up a Virtual Environment**
```bash
# Create the virtual environment
python -m venv env

# Activate the virtual environment (Windows)
env\Scripts\activate

# Activate the virtual environment (Mac/Linux)
source env/bin/activate

# Install Requirements
pip install -r requirements.txt
```

---

## 🗄️ Dataset Preparation
Link
```bash
https://www.kaggle.com/datasets/srivatsanmk2004/accident-detection-dataset
```

---

## 🧠 How to Train the Model

If you want to retrain the model from scratch or tweak the architecture, follow these steps:

1. Ensure your dataset is properly extracted in the project folder.
2. Open the Jupyter Notebook:
   ```bash
   jupyter notebook accident-classification.ipynb
   ```

3. Run through the cells sequentially. The notebook will:

* Load and preprocess the image data.
* Train the deep learning model.
* Evaluate the model's accuracy.
* Save the newly trained model weights and structure into the `model/` folder.

---

## 💻 How to Run the Application
Once your environment is set up and the model is ready (either pre-trained or newly trained), you can run the main application.
```bash
python app.py
```
