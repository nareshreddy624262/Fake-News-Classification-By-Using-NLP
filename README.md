# Fake-News-Classification-By-Using-NLP
```markdown
# 📰 Fake News Classification 🕵️‍♂️

This repository contains a **Fake News Classification** project, which aims to detect whether a given news article is **real** or **fake** using Natural Language Processing (NLP) and Machine Learning techniques.

---

## 📌 Project Overview

Fake news is a growing problem in today's digital world, spreading misinformation across social media and online platforms. This project builds a **machine learning model** that classifies news articles as **Fake** or **Real** based on their textual content.

---

## 📂 Repository Structure

```
📂 Fake-News-Classification  
│── 📁 data/             # Dataset used for training & testing  
│── 📁 notebooks/        # Jupyter notebooks with step-by-step explanations  
│── 📁 models/          # Saved trained models  
│── 📁 scripts/         # Python scripts for training and inference  
│── README.md           # Project documentation  
│── requirements.txt    # Dependencies  
│── app.py              # Flask app for web-based predictions  
```

---

## 📌 Dataset

This project uses the **Fake News Dataset** from Kaggle:
- [Fake News Dataset](https://www.kaggle.com/c/fake-news/)
- It contains **title**, **text**, **subject**, and **label** columns.
- Labels:  
  - **1** → Fake News  
  - **0** → Real News  

---

## 🏗️ Techniques Used

### 1️⃣ **Data Preprocessing**
- Removing stopwords, punctuations, and special characters.
- Tokenization and Lemmatization.
- Lowercasing and whitespace removal.

### 2️⃣ **Feature Engineering**
- **TF-IDF Vectorization** (Term Frequency - Inverse Document Frequency)
- **Count Vectorization**
- **Word Embeddings (Word2Vec, GloVe, FastText)**

### 3️⃣ **Machine Learning Models**
- Logistic Regression
- Naïve Bayes Classifier
- Support Vector Machine (SVM)
- Random Forest
- Gradient Boosting (XGBoost, LightGBM)

### 4️⃣ **Deep Learning Models**
- Recurrent Neural Networks (RNNs)
- Long Short-Term Memory (LSTM)
- Bidirectional LSTM (Bi-LSTM)

---

## 🚀 Installation and Setup

### Clone the Repository
```bash
git clone https://github.com/your-username/Fake-News-Classification.git
cd Fake-News-Classification
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 📢 How to Run

### Run Jupyter Notebooks:
Open and execute the notebooks inside the `notebooks/` folder:
```bash
jupyter notebook
```

### Run Python Scripts:
Train the model using:
```bash
python scripts/train_model.py
```
Test the model using:
```bash
python scripts/test_model.py
```

### Run Web App:
A **Flask Web Application** is included for easy user interaction.
```bash
python app.py
```
Open your browser and go to `http://127.0.0.1:5000/` to test the model.

---

## 📊 Example Usage

```python
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Load dataset
df = pd.read_csv('data/fake_news.csv')

# Vectorization
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['text'])

# Train model
model = LogisticRegression()
model.fit(X, df['label'])

# Predict
sample_news = ["Breaking: Scientists discover a new planet."]
sample_vector = vectorizer.transform(sample_news)
prediction = model.predict(sample_vector)

print("Fake News" if prediction[0] == 1 else "Real News")
```

---

## 🔍 Performance Metrics

The trained models are evaluated based on:
- **Accuracy**
- **Precision**
- **Recall**
- **F1 Score**
- **Confusion Matrix**
- **ROC-AUC Score**

---

## 📜 Contribution

Contributions are welcome! If you find any improvements, feel free to:
1. **Fork** the repository.
2. **Create a new branch**:  
   ```bash
   git checkout -b feature-branch
   ```
3. **Commit changes**:  
   ```bash
   git commit -m "Added a new feature"
   ```
4. **Push the branch**:  
   ```bash
   git push origin feature-branch
   ```
5. **Open a Pull Request (PR)**.

---

## 🏆 License

This project is licensed under the **MIT License**. Feel free to use and modify it.

---

## 📞 Contact

For any queries or suggestions, connect with me on:

📧 Email: your-email@example.com  
💼 LinkedIn: [Your Profile](https://www.linkedin.com/in/your-profile)  

---

🚀 **Happy Coding!** 🚀
```
