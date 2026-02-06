import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_csv("data/electronics.csv")

tfidf = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf.fit_transform(df["features"])

def recommend(product_name, top_n=5):
    idx = df[df["name"] == product_name].index[0]
    similarity = cosine_similarity(tfidf_matrix[idx], tfidf_matrix)[0]
    similar_products = similarity.argsort()[::-1][1:top_n+1]
    return df.iloc[similar_products][["name", "price", "rating"]]

print(recommend("iPhone 14"))
