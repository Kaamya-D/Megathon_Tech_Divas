# Save intensity_scorer.py
# %%writefile intensity_scorer.py

from transformers import pipeline
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from keybert import KeyBERT

emotion_model = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", return_all_scores=True)
sentiment_analyzer = SentimentIntensityAnalyzer()
kw_model = KeyBERT()

def detect_emotions(text):
    emotions = emotion_model(text)[0]
    return {emotion['label']: emotion['score'] for emotion in emotions}

def extract_keywords(text, top_n=5):
    keywords = kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 2), top_n=top_n)
    return [keyword[0] for keyword in keywords]

def score_sentiment(text):
    return sentiment_analyzer.polarity_scores(text)

def analyze_text(text):
    emotions = detect_emotions(text)
    keywords = extract_keywords(text)
    sentiment = score_sentiment(text)
    return {
        "emotions": emotions,
        "keywords": keywords,
        "sentiment": sentiment
    }