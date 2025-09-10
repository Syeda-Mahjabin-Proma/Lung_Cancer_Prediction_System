# 🫁 Lung Cancer Prediction System

## 📌 Overview

This project predicts **lung cancer risk** based on patient details such as **age, gender, smoking history, exposure to harmful substances, COPD diagnosis and family history**.
It uses **Random Forest**, **Multi-Layer Perceptron (MLP)**, and **1D CNN** models to classify patients as high-risk or low-risk, with probabilities for each prediction.

---

## 🗂️ Dataset Description

The dataset `lung_cancer_dataset.csv` contains **50,000 rows** and the following columns:

| Column                          | Type    | Description                                              |
| ------------------------------- | ------- | -------------------------------------------------------- |
| **patient\_id**                 | int64   | Unique patient identifier (dropped during preprocessing) |
| **age**                         | int64   | Age of the patient                                       |
| **gender**                      | object  | Male/Female → encoded as 1/0                             |
| **pack\_years**                 | float64 | Smoking history (packs/day × years)                      |
| **radon\_exposure**             | object  | Low/Medium/High → encoded as 1/2/3                       |
| **asbestos\_exposure**          | object  | Yes/No → encoded as 1/0                                  |
| **secondhand\_smoke\_exposure** | object  | Yes/No → encoded as 1/0                                  |
| **copd\_diagnosis**             | object  | Yes/No → encoded as 1/0                                  |
| **alcohol\_consumption**        | object  | Dropped after preprocessing                              |
| **family\_history**             | object  | Yes/No → encoded as 1/0                                  |
| **lung\_cancer**                | object  | Yes/No → encoded as 1/0 (target variable)                |

### Data Preprocessing

* Removed unnecessary columns (`patient_id`, `alcohol_consumption`)
* Encoded categorical features to numeric values
* Converted string exposures to numeric scales (e.g., `radon_exposure`)
* Checked for missing values and cleaned dataset

---

## 📊 Exploratory Data Analysis (EDA)

* Histograms for all features to understand distributions
* Boxplots to detect outliers and visualize spread
* Correlation heatmap to inspect feature relationships

---

## Modeling

### 1️⃣ Random Forest

* Trained using **RandomForestClassifier**
* Evaluated with **Accuracy, Precision, Recall, F1-Score**
* Confusion matrix plotted to visualize predictions

### 2️⃣ Multi-Layer Perceptron (MLP)

* Built using **TensorFlow Keras**
* Architecture: Dense(64) → Dropout(0.3) → Dense(32) → Dense(1, Sigmoid)
* Trained with **binary crossentropy** loss and **Adam optimizer**
* Evaluated on test set with metrics and confusion matrix

### 3️⃣ 1D CNN

* Built using **Conv1D layers + GlobalMaxPooling + Dense layers**
* Trained to learn feature interactions for better performance
* Evaluated with **Accuracy, Precision, Recall, F1-Score, ROC-AUC**

---

## Model Performance

| Model            | Accuracy | Precision | Recall | F1 Score |
| ---------------- | -------- | --------- | ------ | -------- |
| Random Forest    | 0.7006   | 0.688     | 0.7006 | 0.693    |
| MLP              | 0.7321   | 0.715     | 0.7321 | 0.713    |
| 1D CNN           | 0.7316   | 0.714     | 0.7316 | 0.704    |

> Metrics are calculated on the **train set** to evaluate model performance.

---

## User Input Prediction

Users can input patient details:

* Age
* Gender (0=Female, 1=Male)
* Pack Years (Smoking history)
* Radon Exposure (1=Low, 2=Medium, 3=High)
* Asbestos Exposure (0=No, 1=Yes)
* Secondhand Smoke Exposure (0=No, 1=Yes)
* COPD Diagnosis (0=No, 1=Yes)
* Family History (0=No, 1=Yes)

The system predicts:

* **Lung Cancer Risk**: Yes/No

---

## Technologies Used

* Python
* Pandas & NumPy
* Matplotlib & Seaborn
* Scikit-learn
* TensorFlow & Keras
* MLxtend (Confusion Matrix Plotting)

---

## ▶️ How to Run

1. Clone the repository:

```bash
git clone https://github.com/Syeda-Mahjabin-Proma/Lung_Cancer_Prediction_System.git
```

2. Install dependencies:

```bash
pip install -r requirements.txt
pip install mlxtend
```

3. Run the Python script:

```bash
python main.py
```

4. Enter patient details when prompted to get lung cancer risk prediction

---

## 🌟 Future Improvements

* Include additional patient lifestyle features (diet, physical activity)
* Hyperparameter tuning for MLP and CNN models 
* Develop a **web or mobile application** for real-time risk assessment 
* Integrate explainable AI (SHAP/LIME) to show **feature importance**

---

## 📄 License

This project is licensed under the [MIT License](https://opensource.org/licenses/MIT)
