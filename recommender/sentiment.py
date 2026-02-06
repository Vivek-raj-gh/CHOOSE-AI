from nltk.sentiment.vader import SentimentIntensityAnalyzer

sia = SentimentIntensityAnalyzer()

def analyze_review(review):
    score = sia.polarity_scores(review)
    return score["compound"]

print(analyze_review("Amazing battery life and camera"))
