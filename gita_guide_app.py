import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

# Load the dataset
@st.cache_data
def load_data():
    df = pd.read_csv("gita_dataset.csv")
    df.dropna(inplace=True)
    return df

df = load_data()

# TF-IDF setup
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['Themes'])

# Recommendation logic
def recommend(query):
    query_vec = tfidf.transform([query])
    similarity = cosine_similarity(query_vec, tfidf_matrix).flatten()
    top_index = similarity.argmax()
    verse = df.iloc[top_index]
    return {
        'chapter': verse['Chapter'],
        'verse_no': verse['Verse'],
        'sanskrit': verse['Sanskrit'],
        'translation': verse['English Translation'],
        'summary': verse['Summary']
    }

# Chatbot handler
def chatbot_response(user_input):
    pattern = r"(I feel|I'm facing|What should I do|Tell me about|How to deal with)\s(.+)"
    match = re.search(pattern, user_input, re.IGNORECASE)

    if match:
        query = match.group(2)
        return recommend(query)
    else:
        return {
            'chapter': "",
            'verse_no': "",
            'sanskrit': "",
            'translation': "",
            'summary': "Try: 'I feel anxious' or 'How to deal with failure?'"
        }

# Streamlit UI
st.set_page_config(page_title="🕉️ GitaGuide", page_icon="📖")
st.title("🕉️ GitaGuide: Spiritual Chatbot Based on the Bhagavad Gita")
st.markdown("Ask your life questions and get divine wisdom from the Gita ✨")

user_input = st.text_input("🧘 What’s on your mind?", placeholder="e.g., I feel lost")

if st.button("Get Gita Wisdom"):
    result = chatbot_response(user_input)

    if result['chapter']:
        st.markdown(f"### 📖 Chapter {result['chapter']}, Verse {result['verse_no']}")
        st.markdown(f"**🗣️ Sanskrit:** {result['sanskrit']}")
        st.markdown(f"**🌍 Translation:** {result['translation']}")
        st.markdown(f"**💡 Insight:** {result['summary']}")
    else:
        st.info(result['summary'])
