import streamlit as st
import pandas as pd
from textblob import TextBlob
import cleantext
import plotly.express as px
from PIL import Image
import cv2
import numpy as np
import tempfile
from fer import FER

# Set page config
st.set_page_config(page_title="Sentiment Analysis App", layout="wide")

# Custom CSS with background image
st.markdown("""...""", unsafe_allow_html=True)  # Your existing CSS here

# Placeholder function for image sentiment analysis
def analyze_image(image_np):
    detector = FER()
    emotions = detector.detect_emotions(image_np)
    
    # Get the dominant emotion
    if emotions:
        dominant_emotion = max(emotions[0]['emotions'], key=emotions[0]['emotions'].get)
        return dominant_emotion
    return "No face detected"

# Placeholder function for video sentiment analysis
def analyze_video(video_path):
    # Initialize the FER detector
    detector = FER()

    # Initialize a dictionary to hold emotion counts
    emotion_counts = {emotion: 0 for emotion in ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']}

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    # Frame count for averaging results
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Analyze the emotions in the frame
        emotions = detector.detect_emotions(frame)
        
        # If emotions are detected, update counts
        if emotions:
            dominant_emotion = max(emotions[0]['emotions'], key=emotions[0]['emotions'].get)
            emotion_counts[dominant_emotion] += 1
        
        frame_count += 1

    cap.release()

    # Determine the overall sentiment based on counts
    if frame_count == 0:
        return "No faces detected"
    
    overall_emotion = max(emotion_counts, key=emotion_counts.get)
    return overall_emotion
# Header
st.title("✨ Sentiment Analysis Dashboard")

# Sidebar
# Sidebar Design
st.sidebar.header("About")
st.sidebar.info(
    "This app performs sentiment analysis on text, CSV data, images, and videos. "
    "Use the tabs below to analyze individual text, upload a CSV file, analyze an image, or a video."
)


# Optional: Add a footer or additional info
st.sidebar.markdown("---")
st.sidebar.markdown("Created with ❤️ using Streamlit")


# Main content
tab1, tab2, tab3, tab4 = st.tabs(["📝 Analyze Text", "📊 Analyze CSV", "🖼️ Analyze Image", "🎥 Analyze Video"])

# Text Analysis Tab
with tab1:
    st.header("Text Analysis")
    
    # Text input
    text = st.text_area("Enter text to analyze:", height=150)
    
    if text:
        with st.spinner("Analyzing..."):
            # Sentiment analysis
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity
            subjectivity = blob.sentiment.subjectivity
            
            # Display results
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Polarity", f"{polarity:.2f}")
                st.progress((polarity + 1) / 2)  # Normalize to 0-1 range
            with col2:
                st.metric("Subjectivity", f"{subjectivity:.2f}")
                st.progress(subjectivity)
            
            # Interpretation
            sentiment = "Positive" if polarity > 0 else "Negative" if polarity < 0 else "Neutral"
            st.info(f"The text sentiment is: **{sentiment}**")
    
    # Text cleaning
    st.subheader("Text Cleaning")
    pre = st.text_input("Enter text to clean:")
    if pre:
        cleaned = cleantext.clean(pre, clean_all=False, extra_spaces=True,
                                  stopwords=True, lowercase=True, numbers=True, punct=True)
        st.code(cleaned)

# CSV Analysis Tab
with tab2:
    st.header("CSV Analysis")
    
    upl = st.file_uploader("Upload CSV file", type=["csv", "xlsx"])
    
    if upl:
        try:
            # Load data
            if upl.name.endswith('.csv'):
                df = pd.read_csv(upl)
            else:
                df = pd.read_excel(upl)
            
            # Remove unnamed columns
            df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
            
            # Perform sentiment analysis
            @st.cache_data
            def analyze_sentiment(df):
                df['score'] = df['tweet'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
                df['sentiment'] = df['score'].apply(lambda x: 'Positive' if x > 0 else ('Negative' if x < 0 else 'Neutral'))
                return df
            
            df = analyze_sentiment(df)
            
            # Display results
            st.subheader("Data Preview")
            st.dataframe(df.head(10), use_container_width=True)
            
            # Sentiment distribution
            st.subheader("Sentiment Distribution")
            fig = px.pie(df, names='sentiment', title='Sentiment Distribution')
            st.plotly_chart(fig, use_container_width=True)
            
            # Download option
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download analyzed data as CSV",
                data=csv,
                file_name='sentiment_analyzed.csv',
                mime='text/csv',
            )
        
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.error("Please make sure your CSV file contains a 'tweets' column.")

# Image Analysis Tab



# Image Analysis Tab
with tab3:
    st.header("Image Sentiment Analysis")
    
    # Image uploader
    image_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    
    if image_file is not None:
        # Open and convert image
        image = Image.open(image_file)
        
        # Convert image to NumPy array
        image_np = np.array(image)
        
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Analyze image sentiment
        with st.spinner("Analyzing image..."):
            sentiment = analyze_image(image_np)
            st.success(f"The detected sentiment is: **{sentiment}**")

# Video Analysis Tab
# Video Analysis Tab
with tab4:
    st.header("Video Sentiment Analysis")
    
    # Video uploader
    video_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])
    
    if video_file is not None:
        # Create a temporary file to save the uploaded video
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(video_file.read())
            video_path = tmp_file.name
        
        # Display the uploaded video
        st.video(video_path)
        
        # Analyze video sentiment
        with st.spinner("Analyzing video..."):
            sentiment = analyze_video(video_path)
            st.success(f"The detected sentiment is: **{sentiment}**")


# Footer
st.markdown("""<div style="color: black; text-align: center;"> --- Created with ❤️ using Streamlit </div>""", unsafe_allow_html=True)
