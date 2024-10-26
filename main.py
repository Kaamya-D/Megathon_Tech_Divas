# Save main.py
#%%writefile main.py

from intensity_scorer import analyze_text
from utils import clean_text

def main():
    text = input("Enter some text: ")
    cleaned_text = clean_text(text)
    analysis = analyze_text(cleaned_text)
    
    print("\nText Analysis Results:")
    print("Emotions:", analysis["emotions"])
    print("Keywords:", analysis["keywords"])
    print("Sentiment:", analysis["sentiment"])

if _name_ == "_main_":
    main()