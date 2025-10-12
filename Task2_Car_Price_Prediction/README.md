# 🚗 Car Price Prediction 💰

[![Python](https://img.shields.io/badge/python-3.12-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

---

## Table of Contents
1. [Project Overview](#project-overview)  
2. [Project Workflow](#project-workflow)  
3. [Project Structure](#project-structure)  
4. [Tech Stack](#tech-stack)  
5. [Dataset](#dataset)  
6. [Installation](#installation)  
7. [Usage](#usage)  
   - [Example Inputs](#example-inputs)  
8. [Results](#results)  
9. [Screenshots](#screenshots)  
10. [Future Improvements](#future-improvements)  
11. [License](#license)  
12. [Author](#author)  

---

## Project Overview
This project predicts the **selling price of used cars** using machine learning.  
It demonstrates a complete **end-to-end ML workflow**: data preprocessing, EDA, feature engineering, model training, evaluation, and deployment via Streamlit.

**Key Features:**
- Predict car prices using inputs like brand, year, kms driven, fuel type, transmission, and ownership.
- Preprocessing includes `Car_Age` calculation and label encoding of categorical variables.
- Handles invalid/unseen inputs gracefully (optional improvement).

---

## Project Workflow
The complete workflow of the project is as follows:

1. **Data Collection** → Load dataset containing car details (price, year, kms, fuel, etc.)  
2. **Data Cleaning & Preprocessing** → Handle missing values, encode categorical variables, and feature engineering (Car Age).  
3. **Exploratory Data Analysis (EDA)** → Visualize distributions, correlations, and feature importance.  
4. **Feature Selection** → Select key predictors such as Present Price, Kms Driven, Fuel Type, Transmission, Car Age.  
5. **Model Training** → Train multiple ML models (Linear Regression, Decision Tree, Random Forest, etc.).  
6. **Model Evaluation** → Compare models using R², RMSE, and select the best one.  
7. **Model Saving** → Store the trained model in `models/` using joblib/pickle.  
8. **Deployment** → Build interactive web app using **Streamlit** for real-time predictions.  

---

## Project Structure

car-price-prediction/
│
├── app.py # Streamlit web application
├── requirements.txt # Dependencies
├── README.md # Documentation
├── models/ # Trained models & encoders
│ ├── model.pkl
│ └── encoders.pkl
├── data/ # Datasets
│ ├── raw/
│ └── processed/
├── notebooks/ # EDA & model notebooks
│ └── car_price_prediction.ipynb
│ 
├── src/ # Python scripts
│ ├── __init__.py
│ ├── data_preprocessing.py
│ ├── model.py
│ ├── feature_engineering.py
│ └── predict.py
├── reports/ # EDA plots & evaluation
│ ├── eda_plots/
│ └── correlation_heatmap.png
└── logs/ # Application logs

---


---

## Tech Stack
- **Language:** Python  
- **Libraries:** Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn  
- **Deployment:** Streamlit  
- **Serialization:** Pickle  
- **Logging:** Python logging module  

---

## Dataset
- Source: Sample car dataset
- Features: `Car_Name`, `Present_Price`, `Driven_kms`, `Fuel_Type`, `Selling_type`, `Transmission`, `Owner`, `Year`  
- Preprocessing:
  - Added `Car_Age`  
  - Encoded categorical variables using LabelEncoder  

---

## Installation
   ```bash
   # Clone repo
   git clone <repo_url>
   cd car-price-prediction

   # Create virtual environment
   python -m venv venv
   # Activate
   # Windows
   venv\Scripts\activate
   # Linux/Mac
   source venv/bin/activate

   # Install dependencies
   pip install -r requirements.txt

   # Run pipeline (optional)
   python main.py

   # Run Streamlit app
   streamlit run app.py 
   ``` 

---

## Usage

1. Open the Streamlit app.
2.  Enter car details in the input form:
   - Only valid options from training data are allowed for categorical fields. 
   - Years must be within training data range.
3. Click Predict.
4. View predicted selling price.

---

### Example Inputs:

Car Name: Bajaj Avenger 220
Fuel Type: Diesel
Selling Type: Dealer
Transmission: Manual
Present Price: 1.25 Lakhs
Driven KMs: 10000
Owner: 2
Year of Purchase: 2001

---

## Results

- Best Model: LinearRegression (R² ≈ 0.92)
- Key Features: Present Price, Car Age, Driven KMs
- Example Predictions:
   - Bajaj Avenger 220 → ₹ 0.07 Lakhs
   - Activa 4g → ₹ 1.34 Lakhs

---

## Screenshots

App Interface


Prediction Output


EDA & Correlation

---

## Future Improvements

- Restrict user inputs to training dataset ranges to prevent invalid entries.
- Use advanced ML models (XGBoost, Gradient Boosting).
- Add interactive visual analytics.
- Deploy as a full web application with authentication.

---

## License

This project is licensed under the MIT License.
See the LICENSE
 file for details.

---

## Author

Ayesha Banu

M.Sc. Computer Science |  Gold Medalist

Data Scientist | Data Analyst | Full-Stack Python Developer | GenAI Enthusiast

Email: ayesha24banu@gmail.com

Linkedin: https://www.linkedin.com/in/ayesha_banu_cs