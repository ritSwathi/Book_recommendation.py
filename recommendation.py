import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import numpy as np
books = pd.read_csv(r"C:\Users\SWATHI\Downloads\python\books.csv")
ratings = pd.read_csv(r"C:\Users\SWATHI\Downloads\python\ratings.csv")
tags = pd.read_csv(r"C:\Users\SWATHI\Downloads\python\tags.csv")
book_tags = pd.read_csv(r"C:\Users\SWATHI\Downloads\python\book_tags.csv")

# Merge book_tags with tags to get readable tag_name
book_tags = book_tags.merge(tags, on='tag_id', how='left')

# Fill missing tag names safely
book_tags['tag_name'] = book_tags['tag_name'].fillna('')

# Aggregate tags per book
book_tags_grouped = book_tags.groupby('goodreads_book_id')['tag_name'].apply(lambda x: " ".join(x)).reset_index()

# Merge aggregated tags with books
books = books.merge(book_tags_grouped, left_on='book_id', right_on='goodreads_book_id', how='left')

# Fill missing tags with empty string
books['tag_name'] = books['tag_name'].fillna('')

# Keep only relevant columns
books = books[['book_id', 'title', 'authors', 'tag_name']]

# Create a combined features column for content-based filtering
books['combined_features'] = books['title'] + " " + books['authors'] + " " + books['tag_name']
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(books['combined_features'])

# Compute cosine similarity
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# Create a mapping from book_id to index
book_indices = pd.Series(books.index, index=books['title']).drop_duplicates()
def recommend_books(title, num_recommendations=5):
    if title not in book_indices:
        print("Book not found in dataset.")
        return
    
    idx = book_indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:num_recommendations+1]
    
    book_indices_list = [i[0] for i in sim_scores]
    recommended_books = books.iloc[book_indices_list][['title', 'authors']]
    return recommended_books
book_to_search = "Atomic Habits"  # change to any book in your dataset
recommend_books(book_to_search, 5)
