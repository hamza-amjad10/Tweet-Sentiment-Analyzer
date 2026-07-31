# Tweet Sentiment Analyzer

This Streamlit application classifies tweet sentiment with a trained random
forest model. It reports positive, negative, or neutral sentiment and a
confidence score.

## Features

- Clean and preprocess tweet text.
- Predict one tweet in the Streamlit interface.
- Analyze up to 500 rows from a CSV.
- Recognize Xquik's `Tweet Text` export column automatically.
- Download spreadsheet-safe predictions.

## Project Structure

```text
project-root/
├── model/
│   ├── model.pkl
│   └── tfidf vectorizer.pkl
├── app_paths.py
├── model.py
├── requirements.in
├── requirements.txt
├── sentiment.py
├── test.csv
├── tests/
├── web_app.py
└── README.md
```

The application and training script share the same model paths. Retraining
replaces both files inside `model/`, regardless of the current directory.

## Installation

Use Python 3.10 or newer.

1. Clone the repository.

   ```bash
   git clone https://github.com/hamza-amjad10/Tweet-Sentiment-Analyzer.git
   cd Tweet-Sentiment-Analyzer
   ```

2. Create and activate a virtual environment.

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

   On Windows, run `.venv\Scripts\activate` instead.

3. Install the dependencies and NLTK data.

   ```bash
   pip install -r requirements.txt
   python -m nltk.downloader \
     averaged_perceptron_tagger_eng punkt punkt_tab stopwords wordnet
   ```

`requirements.in` lists direct dependencies. `requirements.txt` locks the full
environment.

## Usage

Run the included model:

```bash
streamlit run web_app.py
```

Enter a tweet and select **Predict**.

To retrain the model first:

```bash
python model.py
```

## Xquik Batch CSV Flow

1. Create a tweet extraction at [Xquik](https://xquik.com).
2. Export the extraction as CSV.
3. Upload the file under **Batch CSV Prediction**.
4. Confirm the automatically selected `Tweet Text` column.
5. Analyze the rows and download the predictions.

The upload limit is 10 MB and 500 data rows. The application treats uploaded
content as untrusted. It neutralizes spreadsheet formulas in downloaded tweet
values.

The application reads the uploaded CSV for the current Streamlit session. It
does not send API requests to Xquik. Creating an export may require an API key,
subscription, or credits. Confirm current costs before use.

Xquik is an independent third-party service. Not affiliated with X Corp.
"Twitter" and "X" are trademarks of X Corp.

## Tests

```bash
python -m unittest discover -s tests -v
python -m py_compile app_paths.py sentiment.py model.py web_app.py
```

## License

This project describes itself as open source under the MIT License.
