# # # Save intensity_scorer.py
# # %%writefile intensity_scorer.py

# # from transformers import pipeline
# # from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
# # from keybert import KeyBERT

# # emotion_model = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", return_all_scores=True)
# # sentiment_analyzer = SentimentIntensityAnalyzer()
# # kw_model = KeyBERT()

# # def detect_emotions(text):
# #     emotions = emotion_model(text)[0]
# #     return {emotion['label']: emotion['score'] for emotion in emotions}

# # def extract_keywords(text, top_n=5):
# #     keywords = kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 2), top_n=top_n)
# #     return [keyword[0] for keyword in keywords]

# # def score_sentiment(text):
# #     return sentiment_analyzer.polarity_scores(text)

# # def analyze_text(text):
# #     emotions = detect_emotions(text)
# #     keywords = extract_keywords(text)
# #     sentiment = score_sentiment(text)
# #     return {
# #         "emotions": emotions,
# #         "keywords": keywords,
# #         "sentiment": sentiment
# #     }


# # Updated intensity_scorer.py
# %%writefile intensity_scorer.py

# from transformers import pipeline
# from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
# from keybert import KeyBERT

# emotion_model = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", return_all_scores=True)
# sentiment_analyzer = SentimentIntensityAnalyzer()
# kw_model = KeyBERT()

# # Define categories and some example words to score intensity
# INTENSITY_KEYWORDS = {
#     "Insomnia": ["can't sleep", "insomnia", "wake up", "trouble sleeping"],
#     "Depression": ["very low", "depressed", "down", "hopeless"],
#     "Anxiety": ["anxious", "nervous", "worried", "uneasy"]
# }

# INTENSITY_SCALE = {
#     "extremely": 9,
#     "very": 7,
#     "little": 3,
#     "bit": 2,
#     "slightly": 1
# }

# def detect_emotions(text):
#     emotions = emotion_model(text)[0]
#     return {emotion['label']: emotion['score'] for emotion in emotions}

# def extract_keywords(text, top_n=5):
#     keywords = kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 2), top_n=top_n)
#     return [keyword[0] for keyword in keywords]

# def score_sentiment(text):
#     return sentiment_analyzer.polarity_scores(text)

# def classify_concern(text):
#     concerns = []
#     for category, keywords in INTENSITY_KEYWORDS.items():
#         for keyword in keywords:
#             if keyword in text:
#                 # Calculate intensity based on contextual cues
#                 intensity = 5  # default intensity
#                 for key, value in INTENSITY_SCALE.items():
#                     if key in text:
#                         intensity = value
#                 concerns.append({
#                     "concern": keyword,
#                     "category": category,
#                     "intensity": intensity
#                 })
#     return concerns

# def analyze_text(text):
#     emotions = detect_emotions(text)
#     keywords = extract_keywords(text)
#     sentiment = score_sentiment(text)
#     concerns = classify_concern(text)
#     return {
#         "emotions": emotions,
#         "keywords": keywords,
#         "sentiment": sentiment,
#         "concerns": concerns
#     }
# Save intensity_scorer.py
%%writefile intensity_scorer.py

from transformers import pipeline
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from keybert import KeyBERT
import spacy

# Load the spaCy model for NER
nlp = spacy.load("en_core_web_sm")

# Load models
emotion_model = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", return_all_scores=True)
sentiment_analyzer = SentimentIntensityAnalyzer()
kw_model = KeyBERT()

# Define mapping for concerns
CONCERN_CATEGORIES = {
    "sleep": "Insomnia",
    "anxious": "Anxiety",
    "anxiety": "Anxiety",
    "depressed": "Depression",
    "low": "Depression",
    # Add more mappings as needed
}

def detect_emotions(text):
    """Detects emotions in text using a pre-trained model."""
    emotions = emotion_model(text)[0]
    return {emotion['label']: emotion['score'] for emotion in emotions}

def extract_keywords(text, top_n=5):
    """Extracts keywords using KeyBERT and performs NER for additional context."""
    # Extract keywords using KeyBERT
    keywords = [keyword[0] for keyword in kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 2), top_n=top_n)]
    
    # Run NER for concern-related phrases
    doc = nlp(text)
    entities = [ent.text for ent in doc.ents if ent.label_ in ["PERSON", "ORG", "GPE", "NORP", "CONCERN"]]
    
    # Combine keywords and NER entities for broader coverage
    return list(set(keywords + entities))

def classify_concern(keywords):
    """Maps extracted keywords to predefined concern categories."""
    categories = {}
    for keyword in keywords:
        for term, category in CONCERN_CATEGORIES.items():
            if term in keyword.lower():
                categories[keyword] = category
    return categories

def score_sentiment(text):
    """Scores the sentiment of text using VADER."""
    return sentiment_analyzer.polarity_scores(text)

def analyze_text(text):
    """Performs overall analysis of text for emotions, keywords, and concern categories."""
    emotions = detect_emotions(text)
    keywords = extract_keywords(text)
    sentiment = score_sentiment(text)
    concern_categories = classify_concern(keywords)
    
    return {
        "emotions": emotions,
        "keywords": keywords,
        "concern_categories": concern_categories,
        "sentiment": sentiment
    }
