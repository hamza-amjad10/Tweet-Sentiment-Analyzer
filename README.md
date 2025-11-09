"# Tweet-Sentiment-Analyzer" 
This project is a **Tweet Sentiment Analysis** web application built with **Python**, **Scikit-learn**, and **Streamlit**. 
It predicts whether a tweet is **positive**, **negative**, or **neutral**, and also shows the **confidence score** of the prediction.  

---

## Folder Structure

project-root/
├── model/
│ ├── model.pkl # Trained Random Forest classifier
│ └── tfidf_vectorizer.pkl # Saved TF-IDF vectorizer
├── model.py # Script to train and save the model
├── web_app.py # Streamlit web app
├── test.csv # Dataset for training/testing
├── requirements.txt # Project dependencies
└── README.md # Project documentation



## Features

- Preprocess and clean tweet text automatically.
- Predict sentiment as **positive**, **negative**, or **neutral**.
- Show confidence score of predictions.
- Interactive **Streamlit web interface**.
- Visual feedback for predictions using colors and emojis:
  - 🙂 Positive
  - 😶 Neutral
  - 😞 Negative

---

## Installation

1. Clone the repository:

```bash
git clone <https://github.com/hamza-amjad10/Tweet-Sentiment-Analyzer.git>
Create a virtual environment (recommended):


python -m venv env
source env/bin/activate      # On Windows: env\Scripts\activate
Install dependencies:


pip install -r requirements.txt
Usage
Train the model (optional, if you want to retrain):


python model.py
Run the Streamlit web app:


streamlit run web_app.py
Enter your tweet in the text box and click Predict to see the sentiment and confidence.

Example
Tweet:

I just got a promotion at work! Feeling ecstatic and motivated to achieve even more!
Output:

Model predicts the tweet is: positive 🙂
Confidence: 67%

Pre-trained Models
The pre-trained models are stored in the model/ folder:

model.pkl → Random Forest classifier

tfidf_vectorizer.pkl → TF-IDF vectorizer for feature extraction

Dependencies
See requirements.txt for the complete list of Python packages.

License
This project is open-source and available under the MIT License.

pgsql
Copy code

---

I can also **add a section to show how to include screenshots in the README** if you want it to look more visual and professional on GitHub.  

Do you want me to do that too?






