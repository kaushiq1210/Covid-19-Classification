# 🦠 COVID-19 Detection using CNN and PyQt5

A deep learning-based desktop application for detecting **COVID-19 infection from chest X-ray images** using **Convolutional Neural Networks (CNN)**.  
The application is built using **Python**, **TensorFlow/Keras**, and **PyQt5** for an intuitive and interactive GUI.

---

## 🚀 Features

- 🧠 **Train a CNN model** on chest X-ray datasets directly from the GUI.
- 📁 **Load and display images** using an interactive file browser.
- 🔍 **Classify X-ray images** as _COVID_ or _Non-COVID_.
- 📊 **Automatic evaluation** with confusion matrix & classification report.
- 💾 **Save and reload model** (`.json` + `.h5`) for reuse.
- ⚖️ **Handles class imbalance** using computed class weights.
- 🧩 **Early stopping** to prevent overfitting.

---

## 🧩 Project Structure

COVID-19-Detection-GUI/
│
├── main.py # Main application script
├── model.json # Saved CNN model architecture
├── model.weights.h5 # Trained model weights
├── class_indices.json # Class label mapping
│
├── TrainingDataset/ # Training data (organized by class folders)
│ ├── COVID/
│ └── Non-COVID/
│
├── TestingDataset/ # Testing data (organized by class folders)
│ ├── COVID/
│ └── Non-COVID/
│
└── README.md # Project documentation

yaml
Copy code

---

## 🧠 Model Architecture

The CNN model consists of:

- **Conv2D + MaxPooling2D + BatchNormalization layers** for feature extraction
- **Dropout layers** for regularization
- **Flatten + Dense layers** for classification
- **Output layer:** Softmax activation for 2-class output

**Optimizer:** Adam  
**Loss:** Categorical Crossentropy  
**Metrics:** Accuracy

---

## ⚙️ Requirements

Install all dependencies using:

```bash
pip install tensorflow numpy scikit-learn pyqt5
Optional for GPU support:

bash
Copy code
pip install tensorflow-gpu
🖥️ How to Run
1️⃣ Clone the Repository
bash
Copy code
git clone https://github.com/<your-username>/COVID-19-Detection-GUI.git
cd COVID-19-Detection-GUI
2️⃣ Prepare Dataset
Organize your dataset as shown below:

Copy code
TrainingDataset/
 ├── COVID/
 └── Non-COVID/

TestingDataset/
 ├── COVID/
 └── Non-COVID/
Each folder should contain relevant chest X-ray images (.jpg, .png, .jpeg).

3️⃣ Run the Application
bash
Copy code
python main.py
🧭 GUI Overview
Component	Description
🖼️ Image Display	Displays the selected X-ray image
📂 Browse Image	Opens file browser to choose an image
⚙️ Training	Trains CNN model on given dataset
🔍 Classify	Predicts whether image is COVID or Non-COVID
📊 Result Box	Shows prediction label & confidence score

Results & Screenshots

![Confusion Matrix](Covid-19-Classification/assets/Confusion Matrix.png)

![Normal Classification](Covid-19-Classification/assets/Normal.png)

![Covid Classfication]
(Covid-19-Classification/assets/Covid.png)

```
