
# Mental Health Concern Classification Using NLP

## Project Overview

This project uses Natural Language Processing (NLP) to automatically analyze and classify mental health concerns from user input. It aims to provide meaningful insights into users' mental health by detecting emotional polarity, extracting keywords, classifying concerns, scoring intensity, and tracking sentiment progression over time.

The solution consists of five primary components:
1. **Polarity Detection**: Identifies the sentiment of user input and detects shifts over time.
2. **Keyword Extraction (NER)**: Extracts key phrases related to mental health concerns.
3. **Concern Classification**: Classifies extracted phrases into predefined mental health categories.
4. **Intensity Scoring**: Assigns a severity score (1–10) based on linguistic cues.
5. **Timeline-Based Sentiment Shift Analysis**: Tracks and analyzes shifts in user sentiment over multiple inputs.

The bot is designed to provide insights into users' mental health journeys, which may help support mental health care or personal reflection.
**Kindly view all branches for other parts of the code.
**## Goals

The bot should:
1. Detect the polarity of user input.
2. Extract mental health-related keywords.
3. Classify the concern into a specific category.
4. Score the intensity of the concern on a scale from 1 to 10.
5. Track the timeline of sentiment shifts over multiple inputs.

Example:

```
Input (Day 1): "I can’t sleep well and I feel very low."
Output:
  - Polarity: Negative
  - Concern 1: "can’t sleep well" → Category: "Insomnia" → Intensity: 6/10
  - Concern 2: "feel very low" → Category: "Depression" → Intensity: 7/10

Input (Day 7): "I feel a bit better but still anxious."
Output:
  - Polarity: Neutral
  - Concern: "still anxious" → Category: "Anxiety" → Intensity: 4/10
  - Timeline Shift: Signs of improvement from Depression to Anxiety.
```

## Model Details

1. **Polarity Finder**: Uses sentiment analysis techniques to determine the emotional tone of the input.
2. **Keyword Extractor (NER)**: Identifies mental health-related keywords using Named Entity Recognition.
3. **Concern Classifier**: Classifies keywords into predefined categories using a supervised model.
4. **Intensity Scorer**: Scores the severity of concern based on linguistic intensity indicators.
5. **Timeline Analyzer**: Aggregates and tracks the sentiment over multiple inputs.


## Future Improvements

- Improve context sensitivity in the intensity scorer by integrating contextual embeddings.
- Enhance timeline-based analysis with more advanced sentiment tracking techniques.
- Expand concern categories for a more detailed classification.

## Progress report-
Worked on the intensity scorer functionality, building a bot with the memory stored in the system and creating the dataset for training the deep learning chatbot.

---

