🚀 News Text Classification API — Scikit-Learn + FastAPI
A complete end-to-end NLP workflow: model training → evaluation → deployment.

📌 Project Overview
This project demonstrates an end-to-end Natural Language Processing (NLP) workflow, starting from dataset preparation, TF-IDF vectorisation, model experimentation, and evaluation—followed by deployment of the trained model through a FastAPI microservice.
The API provides a /predict endpoint that accepts raw text and returns a predicted news category (e.g., sci.space or rec.sport.baseball).

🎯 Key Features
🔹 Machine Learning (Model Training)
    Uses 20 Newsgroups dataset (two categories for simplicity).
    Converts text to numerical features using TF–IDF vectorisation.
    Trains and compares:
      Linear Support Vector Classifier (LinearSVC)
      Logistic Regression
      Multinomial Naive Bayes
    Selects the best model based on accuracy & F1-score.
    Wraps the best model + TF-IDF into a Scikit-Learn Pipeline.
    Saves the final trained Pipeline using joblib.
  
🔹Model Evaluation
    Accuracy and F1-score metrics
    Classification report (precision, recall, F1)
    Baseline vs advanced model comparison

🔹 API Deployment (FastAPI)
    Loads the saved ML model (news_text_classifier.joblib)
    Exposes a /predict endpoint for real-time predictions
    Provides interactive documentation via Swagger UI at
    👉 http://127.0.0.1:8000/docs

🛠️ Tech Stack

| Component          | Technology        |
| ------------------ | ----------------- |
| Language           | Python            |
| ML Framework       | Scikit-Learn      |
| Feature Extraction | TF–IDF            |
| Model Persistence  | Joblib            |
| API Framework      | FastAPI           |
| Web Server         | Uvicorn           |
| Notebook           | Jupyter / VS Code |

📂 Project Structure
.
├── model.ipynb                 # Model training & evaluation notebook
├── app.py                      # FastAPI app exposing /predict endpoint
├── news_text_classifier.joblib # Saved ML pipeline (TF-IDF + best classifier)
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation

📥 Installation
  1. Clone the repository
    git clone https://github.com/your-username/news-text-classification-api.git
    cd news-text-classification-api

  2. Install dependencies
    pip install -r requirements.txt

🧠 Training the Model
  Open the Jupyter Notebook:
    jupyter notebook model.ipynb

Run all cells to:
  Load and inspect the dataset
  Train several ML models
  Evaluate and compare performance
  Build the final ML pipeline
  Save the trained model using joblib

The trained model is stored as:
  news_text_classifier.joblib

🚀 Running the FastAPI Service
Start the API server using:
  python -m uvicorn app:app --reload

Open:
  Swagger UI: http://127.0.0.1:8000/docs

📝 Example API Request (via Swagger UI)
Request Body:
{
  "text": "NASA discovered a new exoplanet today."
}

Sample Response:
{
  "predicted_label": 1,
  "predicted_category": "sci.space"
}

<img width="1787" height="635" alt="image" src="https://github.com/user-attachments/assets/60cddca5-8f08-433c-9782-f9343cc64749" />

<img width="1788" height="866" alt="image" src="https://github.com/user-attachments/assets/5f06935b-3403-4abf-9037-6f7f88c8b721" />

📌 Future Enhancements
  Include more categories from the 20 Newsgroups dataset
  Integrate transformer-based models (e.g., BERT, DistilBERT)

👤 Author
Omkar Joshi
📧 Email: omkar.joshi.nz@gmail.com
