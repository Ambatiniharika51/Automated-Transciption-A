import streamlit as st

st.title("Podcast Dashboard 🚀")

# Variables 
block_texts = ["Segment 1", "Segment 2", "Segment 3"]
keywords = ["AI", "Podcast", "Technology"]
sentiment = "Positive"

# Segments
st.header("Segments")
for s in block_texts:
    st.write(s)

# Keywords
st.header("Keywords")
for k in keywords:
    st.write(k)

# Sentiment
st.header("Sentiment")
st.write(sentiment)