from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

sent = "Rescuers heoically help beached garbage back into ocean"
analyzer = SentimentIntensityAnalyzer()

senti = analyzer.polarity_scores(sent)

print(senti)