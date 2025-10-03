# Maternal Pregnancy Risk Prediction using Semi-Supervised Learning  

This project is also featured on my **Data Science Portfolio**:

🔗 [Maternal Pregnancy Risk Prediction on Portfolio](https://www.datascienceportfol.io/ZeMichaelKassahun/projects/1)

This project focuses on predicting **maternal pregnancy risks** using **machine learning** and **semi-supervised learning techniques**.  
The goal is to identify high-risk pregnancies based on maternal health indicators and provide meaningful visualizations to support data-driven healthcare insights.  

You can access the project repository here: [GitHub Link](https://github.com/Michael-Warrior-Angel/maternity-pregnancy-risk-prediction)

---

## Key Features
- **Data Preprocessing & Cleaning**
  - Handled missing values and dropped redundant columns.
  - Feature engineering (BMI calculation, Age Group creation).  

- **Exploratory Data Analysis (EDA)**
  - Distribution plots: boxplots, violin plots, stacked bar charts.  
  - Pairwise relationship plots and correlation heatmaps.  
  - PCA visualization for dimensionality reduction.  
  - Time-series analysis of blood pressure across gestational time.  

- **Patient Similarity Network**
  - Built a **BMI-based similarity network** using NetworkX.  
  - Patients (nodes) connected if BMI difference < 2.0.  

- **Machine Learning (Random Forest Classifier)**
  - Hyperparameter tuning with **GridSearchCV**.  
  - Balanced training to handle class imbalance.  
  - Evaluated with accuracy, cross-validation, classification report, and confusion matrix.  

- **Semi-Supervised Learning**
  - Simulated partially labeled data (30% masked as unlabeled).  
  - Combined labeled and unlabeled data for robust model training.  

---

## Tech Stack
- **Python** 3.x  
- **Libraries**:  
  - `pandas`, `numpy`, `scipy`, `scikit-learn`  
  - `matplotlib`, `seaborn`  
  - `networkx`, `openpyxl`  

---

## Installation
Clone the repository and install dependencies:

```bash
git clone https://github.com/Michael-Warrior-Angel/maternity-pregnancy-risk-prediction.git
cd maternity-pregnancy-risk-prediction
pip install -r requirements.txt
