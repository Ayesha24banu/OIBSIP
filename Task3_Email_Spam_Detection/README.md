# 📧 Email Spam Detection using Machine Learning

An interactive **Machine Learning web application** built with **Streamlit** that detects whether an email/message is **Spam** or **Ham**.  
The project demonstrates **end-to-end Data Science workflow**: data preprocessing, model training, evaluation, deployment, and user interaction.  

---

## Table of Contents

1. [Project Overview](#-project-overview)  
2. [Key Features](#-key-features)  
3. [Business Objective](#-business-objective)  
4. [Project Demo](#-project-demo)  
5. [Dataset Overview](#-dataset-overview)  
6. [Tools and Technologies](#-tools-and-technologies)  
7. [Project Structure](#-project-structure)  
8. [Project Workflow](#-project-workflow)  
9. [Example Results](#-example-results)  
10. [Setup Instructions](#-setup-instructions)  
11. [Usage Guide](#-usage-guide)  
    - [Single Message Prediction](#-single-message-prediction)  
    - [Batch Prediction](#-batch-prediction)  
12. [Demo](#-demo)  
13. [Logging](#-logging)  
14. [Sample Dataset](#-sample-dataset)  
15. [Conclusion](#-conclusion)  
16. [Future Enhancements](#-future-enhancements)  
17. [Deliverables](#-deliverables)  
18. [Acknowledgment](#-acknowledgment)  
19. [License](#-license)  
20. [Author](#-author)  

---

## 🚀 Project Overview

Spam emails are a major cybersecurity and productivity concern. This project uses **Natural Language Processing (NLP)** and **Machine Learning** to classify messages as **Spam** or **Ham (Not Spam)**.  

- Trained on the **SMS Spam Collection Dataset**.  
- Uses **TF-IDF Vectorization** for text preprocessing.  
- **SVM Classifier** for robust classification.  
- Provides both **Single Message** and **Batch Prediction** via CSV uploads.  

---

## ✨ Key Features

- 🔹 **Real-time Prediction**: Classify a single message instantly.  
- 🔹 **Batch Prediction**: Upload CSV files and get predictions for multiple messages.  
- 🔹 **User-Friendly Web UI**: Powered by **Streamlit**.  
- 🔹 **Logging**: All actions and predictions are saved in `logs/app.log`.  
- 🔹 **Downloadable Results**: Batch predictions can be exported as CSV.  

---

## 🎯 Business Objective

The goal of this project is to **detect whether an email is Spam or Not Spam (Ham)** using **Machine Learning**.  
This solution helps business email systems, organizations, and spam filters to improve **security, productivity, and user experience**.  

**Key Outcomes:**
- Build a reliable spam classifier using NLP features (TF-IDF)
- Deploy model for single & batch prediction
- Provide an interactive **Streamlit web app**
- Maintain logs for monitoring and debugging

---

## 📸 Project Demo

### Streamlit Web App:

<img width="1916" height="966" alt="Single_Email_Ham_Detection" src="https://github.com/user-attachments/assets/45f2ff7b-13b0-49de-96d6-98341efb9b65" />


### Example Predictions:
**📝 Single Message Prediction**

<img width="1916" height="966" alt="Single_Email_Spam_Detection" src="https://github.com/user-attachments/assets/30250ebd-d02e-4a4a-b2e9-f21820d0f009" />

**📊 Batch Prediction**

<img width="1914" height="1452" alt="Batch_Email_Spam_Detection" src="https://github.com/user-attachments/assets/2b0c5781-e37b-477b-b4c9-d1dc70db987a" />

---

## 📦 Dataset Overview

- **Source:** [Kaggle – SMS Spam Collection Dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)  
- **Size:** 5,572 messages (spam/ham)  

| Feature   | Description                       |
|-----------|-----------------------------------|
| `label`   | Spam or Ham (target variable)     |
| `message` | Email/SMS message text            |

---

## 🧰 Tools and Technologies

| Layer            | Technology                                      |
|------------------|-------------------------------------------------|
| Language         | Python 3.10+                                    |
| Data Handling    | Pandas, NumPy                                   |
| ML Algorithms    | SVM (Scikit-learn)                              |
| Feature Extractor| TF-IDF Vectorizer (Scikit-learn)                |
| Deployment       | Streamlit                                       |
| Logging          | Python `logging` (logs saved in `logs/app.log`) |
| Notebook         | Jupyter Notebook (EDA + training)               |

---

## 🧱 Project Structure

email_spam_detection/
├── data/ # Raw dataset
│ ├──spam.csv
│ └── processed_spam.csv
│
├── config/
| ├── config.yaml
│ └── logging.conf
|
├── notebooks/ # Experiments & training
│ └── spam_spam_detection.ipynb
│
├── models/ # Saved ML models
│ ├── svm_spam_classifier.pkl
│ └── tfidf_vectorizer.pkl
│
├── src/ # Modular scripts
│ ├── init.py
│ ├── data_preprocessing.py
│ ├── feature_extraction.py
│ ├── model_train_evaluate.py
│ └── predict.py
│
├── app.py # Streamlit app
│
├── logs/ # Log files
│ ├── app.log
│ └── project.log
│
├── report/ # Project screenshots
│ ├── app_demo.png
│ ├── prediction_example.png
│ └── confusion_matrix.png
│
├── requirements.txt
├── .gitignore
└── README.md

---

## 🔍 Project Workflow

### 📌 Step 1: Data Preprocessing
- Clean text: remove punctuation, stopwords, lowercase  
- Convert text → numerical vectors using **TF-IDF**  

### 📌 Step 2: Model Training
- Algorithm: **Support Vector Machine (SVM)**  
- Save trained model → `svm_spam_classifier.pkl`  
- Save vectorizer → `tfidf_vectorizer.pkl`  

### 📌 Step 3: Model Evaluation
- Metrics: Accuracy, Precision, Recall, F1-score  
- Confusion Matrix visualization (see image above 👆)  

### 📌 Step 4: Streamlit Web App
- Single message prediction  
- Batch CSV upload prediction  
- Logging → `logs/app.log`  

---

## 📈 Example Results

| Metric       | Value |
|--------------|-------|
| Accuracy     | 97%   |
| Precision    | 96%   |
| Recall       | 95%   |
| F1-Score     | 95%   |

---

## ⚙️ Setup Instructions 

### Prerequisites
- Python 3.10+
- pip installed

1. Clone the Repository
    
```bash
git clone https://github.com/your-username/email_spam_detection.git
cd email_spam_detection
```

2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows
```

3. Install Dependencies

```bash
pip install -r requirements.txt
```

4. Run the Application

```bash
streamlit run app.py
```

---

## 📖 Usage Guide

**📝 Single Message Prediction**

- Enter a message in the text box.
- Click Predict Message.
- The app will display Spam ⚠️ or Ham ✅.

**📊 Batch Prediction**

- Upload a CSV file containing a text column.
- The app predicts spam/ham for all rows.
- Results are shown in a table and available for download as CSV.

---

## 🎥 Demo

📌 Watch the working demo here:


https://github.com/user-attachments/assets/230dd81d-cccd-4078-b383-318c8d2272ca


---

## 📝 Logging

- All activities are logged in:

    logs/app.log

- Includes predictions, errors, and batch operations.

---

## 📂 Sample Dataset

Example demo CSV (sample.csv):

text
Congratulations! You won a free lottery ticket. Claim now!
Hi John, are we still meeting tomorrow?
Get cheap loans now!!! Limited offer.
Don't forget to submit the project by tonight.

---

## 📝 Conclusion

- Achieved 97% accuracy in spam detection
- Deployed Streamlit app for real-time predictions
- Supports single & batch predictions
- Logs maintained in logs/app.log for monitoring

---

## 🔄 Future Enhancements

- Deploy REST API with FastAPI/Flask
- Deep Learning models (LSTM, BERT)
- Multi-language spam detection
- Integration with live email servers
- Implement user authentication.
- Integrate with email clients for real-time detection.
- Deploy to Cloud / Docker for production use.

---

## 📎Deliverables

- svm_spam_classifier.pkl – Trained model
- tfidf_vectorizer.pkl – TF-IDF transformer
- app.py – Streamlit web app
- app.log – Application logs
- predictions.csv – Batch results

---

## 🙏 Acknowledgment

Dataset credits: UCI/Kaggle SMS Spam Collection dataset.
 
---

## 📄 License

This project is licensed under the MIT License.
See the LICENSE
 file for details.
 
---

## 👤 Author

Ayesha Banu

M.Sc. Computer Science |  Gold Medalist

Data Scientist | Data Analyst | Full-Stack Python Developer | GenAI Enthusiast

Email: ayesha24banu@gmail.com

Linkedin: https://www.linkedin.com/in/ayesha_banu_cs
