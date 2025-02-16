

pip install praw

import praw
import re
import pandas as pd
from textblob import TextBlob
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import nltk
nltk.download('stopwords')
stemmer = nltk.SnowballStemmer("english")
from nltk.corpus import stopwords
import string
stopword=set(stopwords.words('english'))

# Function to initialize the Reddit API client
def initialize_reddit_client():
    return praw.Reddit(
        client_id="wjF9jS9lR9f4tTItCZ-XQA",
        client_secret="k0Vb7WJxY8xvtWM-m2N-j1ZihFmkdA",
        user_agent="Harshavardhanv1.0ImplementNearby3261 " ,
        check_for_async=False
    )

# Function to scrape posts from a subreddit
def scrape_reddit_subreddit(reddit, subreddit_name, limit=1500):
    subreddit = reddit.subreddit(subreddit_name)

    # List to hold scraped data
    posts_data = []

    for post in subreddit.new(limit=limit):
        posts_data.append({
            "title": post.title,
            "score": post.score,
            "id": post.id,
            "url": post.url,
            "created_utc": post.created_utc,
            "num_comments": post.num_comments,
            "body": post.selftext,
        })

    # Convert the list of dictionaries into a DataFrame
    posts_df = pd.DataFrame(posts_data)
    return posts_df

def clean_text(text):
    text = str(text).lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    text = [word for word in text.split(' ') if word not in stopword]
    text=" ".join(text)
    text = [stemmer.stem(word) for word in text.split(' ')]
    text=" ".join(text)
    return text

def analyze_sentiment(text):
    analysis = TextBlob(text)
    return 1 if analysis.sentiment.polarity > 0 else 0

def check_stock_sentiment(stock_name, data):
      relevant_posts = data[data['title'].str.contains(stock_name, case=False, na=False)]

      if relevant_posts.empty:
          return f"No relevant posts found for stock: {stock_name}"

      positive_posts = relevant_posts[relevant_posts['sentiment'] == 1].shape[0]
      negative_posts = relevant_posts[relevant_posts['sentiment'] == 0].shape[0]

      if positive_posts > negative_posts:
          return f"Sentiment for {stock_name}: Increasing (Positive sentiment: {positive_posts}, Negative sentiment: {negative_posts})"
      elif positive_posts < negative_posts:
          return f"Sentiment for {stock_name}: Decreasing (Positive sentiment: {positive_posts}, Negative sentiment: {negative_posts})"
      else:
          return f"Sentiment for {stock_name}: Neutral (Positive sentiment: {positive_posts}, Negative sentiment: {negative_posts})"

if __name__ == "__main__":
    # Initialize Reddit client
    reddit = initialize_reddit_client()

    subreddit_name = "wallstreetbets"
    limit = 1500

    # Scrape subreddit data
    posts_df = scrape_reddit_subreddit(reddit, subreddit_name, limit)

    # Save the data to a CSV file
    output_file = f"{subreddit_name}_posts.csv"
    posts_df.to_csv(output_file, index=False)
    print(f"Scraped data saved to {output_file}")

    # Load the data back for sentiment analysis and modeling
    data = pd.read_csv(output_file)
    #
    data["title"] = data["title"].apply(clean_text)
    # Add a sentiment column based on post titles
    data["sentiment"] = data["title"].apply(analyze_sentiment)


    # Prepare data for machine learning
    X = data["title"]  # Features (post titles)
    y = data["sentiment"]  # Labels (sentiment)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Vectorize text data
    vectorizer = TfidfVectorizer()
    X_train_vect = vectorizer.fit_transform(X_train)
    X_test_vect = vectorizer.transform(X_test)

    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_vect, y_train)

    # Make predictions
    y_pred = model.predict(X_test_vect)




data.head(10)

stock_name = input("Enter your favorite stock to analyze sentiment: ")
sentiment_result = check_stock_sentiment(stock_name, data)
print(sentiment_result)

accuracy = accuracy_score(y_test, y_pred)
 report = classification_report(y_test, y_pred)

 print(f"Accuracy: {accuracy}")
 print("Classification Report:\n", report)