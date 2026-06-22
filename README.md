# Fake News Detector

AI-Powered News Authenticity Analysis using Machine Learning.

## Project Overview

This project analyzes news articles and predicts whether they are **Real** or **Fake** using Natural Language Processing (NLP) and Machine Learning.

##  Features

- Detects Fake and Real news articles
- Text preprocessing and cleaning
- TF-IDF vectorization
- Machine Learning classification
- Interactive Streamlit web application
- Confidence score visualization

## Technologies Used

- Python
- Scikit-Learn
- Pandas
- NumPy
- NLTK
- Streamlit

##  Dataset

- Total Articles: 44,898
- Fake Articles: 23,481
- Real Articles: 21,417

##  Installation

Clone the repository:

```bash
git clone https://github.com/abinjosemanavalan-source/Fake-News-Detector.git
cd Fake-News-Detector
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

## Project Structure

```
Fake-News-Detector/
│
├── app.py
├── train_model.py
├── fake_news_model.pkl
├── tfidf_vectorizer.pkl
├── requirements.txt
└── README.md
```

##  Model Performance

- Accuracy: 80.93%

## Disclaimer

This project is intended for educational and research purposes. Predictions may not always reflect real-world facts and should not be used as the sole source for verifying news authenticity.

##  Author

Abin Jose
