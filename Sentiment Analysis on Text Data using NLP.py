import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from textblob import TextBlob
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score
data = {
    'text': [
        "I love this product!",
        "This is the worst experience ever.",
        "It was okay, not too bad.",
        "Absolutely fantastic!",
        "I hate it. Waste of money.",
        "Meh, it's average.",
        "Great quality and service.",
        "Very disappointing...",
        "It’s decent for the price."
    ]
}
df = pd.DataFrame(data)
def get_sentiment(text):
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    if polarity > 0.1:
        return 'Positive'
    elif polarity < -0.1:
        return 'Negative'
    else:
        return 'Neutral'
df['sentiment'] = df['text'].apply(get_sentiment)
print(df)
sns.countplot(data=df, x='sentiment')
plt.title("Sentiment Distribution")
plt.show()
df['label'] = df['sentiment'].map({'Positive': 1, 'Neutral': 0, 'Negative': -1})
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['text'])
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = MultinomialNB()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
