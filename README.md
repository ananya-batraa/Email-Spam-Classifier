# 📧 Email Spam Classifier: Project Overview

This project builds a simple yet effective system to automatically check if an email is spam or legitimate (ham). It's composed of a smart machine learning algorithm (the brain) and an easy-to-use web interface (the face).

Think of the entire system as a digital bouncer that instantly sorts incoming emails.

## 🧠 The Brain: How the Model Learns (train.py)

The machine learning model learns to spot spam by analyzing thousands of pre-labeled examples.

| Step | Process | Simple Explanation | Technical Term |
|------|---------|-------------------|----------------|
| 1. Text Prep | The raw email text is cleaned and standardized. | We make all text lowercase, remove common but unhelpful words (stopwords), and reduce words to their base form (stemming). | Lowercasing, Stopword Removal, Stemming |
| 2. Word to Number | Computers only understand numbers, so we convert the cleaned text into a numerical format. | This tool scores each word based on how important it is in the email and the overall dataset (it looks for words unique to spam). | TfidfVectorizer |
| 3. Training | The numbers from thousands of labeled emails (spam/not spam) are fed into the algorithm. | The algorithm looks for patterns that reliably distinguish spam from legitimate emails. | Naive-Bayes Algorithm |
| 4. Save | The trained "knowledge" (the model) and the word-to-number tool are permanently saved. | We take a photo of the model's brain and the converter so we can use them later. | .pkl file |

## 🤖 The Face: How It Works in Real-Time (api.py)

When a user submits a new email for checking, the API follows these steps:

1. **Load Knowledge**: It instantly loads the saved model and the TfidfVectorizer from the .pkl files.
2. **Convert New Email**: It uses the loaded TfidfVectorizer to convert the new email into the exact same numerical format the model understands.
3. **Predict**: The numbers are fed into the model, which returns its classification.
4. **Result**: It sends back the classification ("spam" or "not spam") along with a confidence score (e.g., 90% sure).

The `/predict` endpoint is the specific API address that accepts new email text via POST requests and returns the classification.

## 🚀 Setup and Run

Follow these steps to set up and run the spam classifier locally.

### 1. Install Dependencies
Install all required libraries (FastAPI, sklearn, etc.):
```bash
pip install -r requirements.txt
```

### 2. Train the Model
This step trains the machine learning model and creates the necessary saved files (.pkl files). Run this first.
```bash
python3 train.py
```

### 3. Start the API Server
This starts the server that handles email classification requests:
```bash
python3 api.py
```

### 4. Access the Web Interface
Use a web server tool like the "Live Server" extension in VS Code.
- Right-click index.html and select "Open with Live Server".
- You can now type or paste emails into the website to check them!

## 💡 Key Takeaway

This project successfully teaches a computer to recognize spam by learning from examples, then uses that saved knowledge to classify new, unseen emails quickly and accurately via a simple web application.
