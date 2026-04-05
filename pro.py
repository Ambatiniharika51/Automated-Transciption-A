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
import streamlit as st
import pandas as pd
import json
import speech_recognition as sr
from textblob import TextBlob
import numpy as np
from io import BytesIO
import soundfile as sf
from collections import Counter
import re
from nltk.tokenize import sent_tokenize
import nltk

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')

# Try to import pyaudio, but don't fail if it's not available
try:
    import pyaudio
    import wave
    PYAUDIO_AVAILABLE = True
except ImportError:
    PYAUDIO_AVAILABLE = False

st.set_page_config(page_title="Podcast Dashboard", layout="wide")

st.title("🎙️ Podcast Dashboard")
st.write("Real-time podcast analysis with voice input, speech-to-text, and sentiment detection")

# Load segments data
try:
    with open('segments.json', 'r') as f:
        segments_data = json.load(f)
except:
    segments_data = []

# Load results data
try:
    with open('results.json', 'r') as f:
        results_data = json.load(f)
except:
    results_data = {"segments": [], "keywords": [], "sentiment": "N/A"}

# Function to analyze sentiment
def analyze_sentiment(text):
    """Analyze sentiment using TextBlob"""
    if not text:
        return "Neutral", 0
    
    analysis = TextBlob(text)
    polarity = analysis.sentiment.polarity
    
    if polarity > 0.1:
        sentiment = "Positive"
    elif polarity < -0.1:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"
    
    score = (polarity + 1) / 2 * 100  # Convert to 0-100 scale
    return sentiment, score

# Function to divide text into topics
def divide_into_topics(text, num_topics=3):
    """Divide podcast text into topics based on sentence clusters"""
    sentences = sent_tokenize(text)
    if len(sentences) <= num_topics:
        return [{"title": f"Topic {i+1}", "content": sent} for i, sent in enumerate(sentences)]
    
    sentences_per_topic = len(sentences) // num_topics
    topics = []
    for i in range(num_topics):
        start_idx = i * sentences_per_topic
        end_idx = (i + 1) * sentences_per_topic if i < num_topics - 1 else len(sentences)
        topic_content = " ".join(sentences[start_idx:end_idx])
        topics.append({
            "title": f"Topic {i+1}",
            "content": topic_content
        })
    return topics

# Function to extract keywords from text
def extract_keywords(text, num_keywords=5):
    """Extract top keywords from text"""
    words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
    stop_words = {'that', 'this', 'with', 'from', 'have', 'were', 'been', 'would', 'could', 'should', 'their', 'about', 'also'}
    filtered_words = [w for w in words if w not in stop_words]
    word_freq = Counter(filtered_words)
    return word_freq.most_common(num_keywords)

# Function to generate summary
def generate_summary(text, num_sentences=2):
    """Generate summary from text using sentence scoring"""
    sentences = sent_tokenize(text)
    if len(sentences) <= num_sentences:
        return text
    
    words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
    word_freq = Counter(words)
    
    sentence_scores = {}
    for i, sent in enumerate(sentences):
        sent_words = re.findall(r'\b[a-zA-Z]{4,}\b', sent.lower())
        sentence_scores[i] = sum(word_freq[w] for w in sent_words)
    
    top_sentences = sorted(sentence_scores, key=sentence_scores.get, reverse=True)[:num_sentences]
    top_sentences = sorted(top_sentences)
    summary = " ".join([sentences[i] for i in top_sentences])
    return summary

# Create tabs
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["📊 Overview", "🎬 Segments", "🔑 Keywords", "😊 Sentiment", "🎤 Voice Input", "📚 Topics & Summaries"])

# Tab 1: Overview
with tab1:
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Segments", len(segments_data))
    with col2:
        st.metric("Keywords Found", len(results_data.get("keywords", [])))
    with col3:
        st.metric("Overall Sentiment", results_data.get("sentiment", "N/A"))

# Tab 2: Segments
with tab2:
    st.subheader("Podcast Segments")
    if segments_data:
        for segment in segments_data:
            with st.expander(f"Segment {segment.get('id')} - {segment.get('start')} to {segment.get('end')}"):
                st.write(f"**Text:** {segment.get('text')}")
                st.write(f"**Summary:** {segment.get('summary')}")
                if segment.get('keywords'):
                    st.write(f"**Keywords:** {', '.join(segment.get('keywords'))}")
    else:
        st.info("No segments data available")

# Tab 3: Keywords
with tab3:
    st.subheader("Top Keywords")
    keywords = results_data.get("keywords", [])
    if keywords:
        # Create a dataframe for better visualization
        keyword_df = pd.DataFrame({
            'Keyword': keywords,
            'Frequency': [len(k) for k in keywords]  # Simple frequency based on keyword length
        })
        st.bar_chart(keyword_df.set_index('Keyword'))
        st.dataframe(keyword_df, width='stretch')
    else:
        st.info("No keywords data available")

# Tab 4: Sentiment
with tab4:
    st.subheader("Sentiment Analysis")
    sentiment = results_data.get("sentiment", "Unknown")
    
    # Show sentiment with emoji
    sentiment_emoji = {
        "Positive": "😊",
        "Negative": "😞",
        "Neutral": "😐"
    }
    
    st.metric("Overall Sentiment", f"{sentiment_emoji.get(sentiment, '❓')} {sentiment}")
    
    # Add a simple sentiment breakdown
    sentiment_data = pd.DataFrame({
        'Sentiment': ['Positive', 'Neutral', 'Negative'],
        'Score': [70, 20, 10]  # Example data
    })
    st.bar_chart(sentiment_data.set_index('Sentiment'))

# Tab 5: Voice Input & Real-time Sentiment
with tab5:
    st.subheader("🎤 Voice & Text Analysis")
    st.write("Choose how you'd like to analyze your text or uploaded audio:")
    
    col_upload, col_text = st.columns(2)
    
    # ===== AUDIO FILE UPLOAD =====
    with col_upload:
        st.markdown("### 📁 Upload Audio")
        st.write("Upload an existing audio file")
        
        audio_file = st.file_uploader("Select audio file", type=["wav", "mp3", "m4a"], key="voice_upload")
        
        if audio_file is not None:
            st.audio(audio_file)
            
            if st.button("🎙️ Transcribe & Analyze", key="transcribe_btn"):
                with st.spinner("Processing audio..."):
                    try:
                        audio_bytes = audio_file.read()
                        audio_path = "temp_audio.wav"
                        with open(audio_path, "wb") as f:
                            f.write(audio_bytes)
                        
                        recognizer = sr.Recognizer()
                        with sr.AudioFile(audio_path) as source:
                            audio = recognizer.record(source)
                        
                        try:
                            transcript = recognizer.recognize_google(audio)
                            st.success("✅ Transcription Complete!")
                            st.write(f"**You said:** {transcript}")
                            
                            sentiment, score = analyze_sentiment(transcript)
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Sentiment", sentiment)
                            with col2:
                                st.metric("Confidence", f"{score:.1f}%")
                            with col3:
                                emoji = "😊" if sentiment == "Positive" else "😞" if sentiment == "Negative" else "😐"
                                st.metric("Emotion", emoji)
                            
                            st.write("**Sentiment Breakdown:**")
                            sentiment_chart = pd.DataFrame({
                                'Type': ['Positive', 'Neutral', 'Negative'],
                                'Score': [
                                    max(0, score),
                                    50 - abs(score - 50),
                                    max(0, 100 - score)
                                ]
                            })
                            st.bar_chart(sentiment_chart.set_index('Type'))
                            
                            # Topics, Summary, Keywords
                            st.subheader("📚 Topics & Summary")
                            topics = divide_into_topics(transcript, 3)
                            for topic in topics:
                                st.write(f"**{topic['title']}:** {topic['content']}")
                            summary = generate_summary(transcript, 2)
                            st.write(f"**Summary:** {summary}")
                            keywords = extract_keywords(transcript, 5)
                            st.write(f"**Keywords:** {', '.join([word for word, count in keywords])}")
                            
                        except sr.UnknownValueError:
                            st.error("❌ Could not understand the audio. Speak more clearly.")
                        except sr.RequestError:
                            st.error("❌ Speech service error. Check your internet connection.")
                            
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
    
    # ===== TEXT INPUT =====
    with col_text:
        st.markdown("### ⌨️ Type Text")
        st.write("Type or paste text to analyze")
        
        user_text = st.text_area("Enter your text here:", placeholder="Type something and click Analyze...", key="text_sentiment", height=100)
        
        if user_text:
            if st.button("✅ Analyze Sentiment", key="analyze_text"):
                sentiment, score = analyze_sentiment(user_text)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Sentiment", sentiment)
                with col2:
                    st.metric("Score", f"{score:.1f}%")
                with col3:
                    emoji = "😊" if sentiment == "Positive" else "😞" if sentiment == "Negative" else "😐"
                    st.metric("Mood", emoji)
                
                st.progress(int(score) / 100)
                
                # Show sentiment breakdown
                st.write("**Breakdown:**")
                breakdown_data = pd.DataFrame({
                    'Type': ['Positive', 'Neutral', 'Negative'],
                    'Score': [60, 30, 10]
                })
                st.bar_chart(breakdown_data.set_index('Type'))
                
                # Topics, Summary, Keywords
                st.subheader("📚 Topics & Summary")
                topics = divide_into_topics(user_text, 3)
                for topic in topics:
                    st.write(f"**{topic['title']}:** {topic['content']}")
                summary = generate_summary(user_text, 2)
                st.write(f"**Summary:** {summary}")
                keywords = extract_keywords(user_text, 5)
                st.write(f"**Keywords:** {', '.join([word for word, count in keywords])}")

# Tab 6: Topics & Summaries
with tab6:
    st.subheader("📚 Topics & Summaries")
    st.write("Upload a podcast audio file to analyze topics, summaries, and keywords.")
    
    uploaded_file = st.file_uploader("Upload Podcast Audio", type=["mp3", "wav", "ogg", "m4a"], key="podcast_upload")
    num_topics = st.slider("Number of Topics", 2, 5, 3, key="num_topics")
    num_keywords = st.slider("Top Keywords to Show", 3, 10, 5, key="num_keywords")
    
    if uploaded_file is not None:
        st.success("✅ File uploaded successfully!")
        
        # Display audio player
        st.subheader("🎵 Audio Player")
        st.audio(uploaded_file, format="audio/wav")
        
        if st.button("🔄 Analyze Podcast", key="analyze_podcast"):
            with st.spinner("🔄 Converting audio to text..."):
                try:
                    recognizer = sr.Recognizer()
                    with sr.AudioFile(uploaded_file) as source:
                        audio = recognizer.record(source)
                    
                    text = recognizer.recognize_google(audio)
                    st.success("✅ Audio processed successfully!")
                    
                    # Display full transcript
                    with st.expander("📝 Full Transcript", expanded=True):
                        st.write(text)
                    
                    # Topics and Analysis
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.subheader("📚 Topics & Summaries")
                        topics = divide_into_topics(text, num_topics)
                        
                        for i, topic in enumerate(topics):
                            with st.container():
                                st.markdown(f"### {topic['title']}")
                                
                                # Summary
                                summary = generate_summary(topic['content'], num_sentences=2)
                                st.markdown(f"**Summary:** {summary}")
                                
                                # Keywords for this topic
                                topic_keywords = extract_keywords(topic['content'], num_keywords)
                                keywords_text = ", ".join([f"{word}({count})" for word, count in topic_keywords])
                                st.markdown(f"**Keywords:** {keywords_text}")
                                
                                st.divider()
                    
                    with col2:
                        st.subheader("😊 Overall Analysis")
                        
                        # Sentiment
                        sentiment, polarity = analyze_sentiment(text)
                        sentiment_emoji = "😊" if sentiment == "Positive" else "😞" if sentiment == "Negative" else "😐"
                        st.metric("Sentiment", f"{sentiment_emoji} {sentiment}", f"{polarity:.2f}")
                        
                        # Overall keywords
                        st.markdown("**Overall Top Keywords:**")
                        overall_keywords = extract_keywords(text, num_keywords)
                        for word, count in overall_keywords:
                            st.write(f"- {word} ({count})")
                        
                        # Stats
                        st.markdown("---")
                        st.metric("Total Words", len(text.split()))
                        st.metric("Total Sentences", len(sent_tokenize(text)))
                
                except Exception as e:
                    st.error(f"❌ Error processing audio: {str(e)}")
                    st.info("💡 Please ensure the audio is clear and in a supported format (MP3, WAV, OGG, M4A)")
    
    else:
        st.info("👈 Please upload a podcast audio file to get started!") 