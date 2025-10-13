# app.py
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time
import io
from PIL import Image
import base64
import requests
from streamlit_lottie import st_lottie
import json

# Page configuration
st.set_page_config(
    page_title="AI Model Hub",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for animations and styling
def local_css():
    st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        background: linear-gradient(90deg, #FF4B4B, #FF9F43);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem;
        animation: fadeIn 2s;
    }
    
    @keyframes fadeIn {
        0% { opacity: 0; }
        100% { opacity: 1; }
    }
    
    .stButton>button {
        background: linear-gradient(45deg, #FF4B4B, #FF9F43);
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 10px;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: scale(1.05);
        box-shadow: 0 5px 15px rgba(255, 75, 75, 0.4);
    }
    
    .upload-box {
        border: 2px dashed #FF4B4B;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        background-color: rgba(255, 75, 75, 0.05);
        margin: 1rem 0;
        transition: all 0.3s ease;
    }
    
    .upload-box:hover {
        background-color: rgba(255, 75, 75, 0.1);
        transform: translateY(-5px);
    }
    
    .result-box {
        border: 2px solid #4CAF50;
        border-radius: 10px;
        padding: 1.5rem;
        background-color: rgba(76, 175, 80, 0.05);
        margin: 1rem 0;
        animation: slideIn 0.5s ease-out;
    }
    
    @keyframes slideIn {
        0% { transform: translateY(20px); opacity: 0; }
        100% { transform: translateY(0); opacity: 1; }
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        padding: 1rem;
        color: white;
        text-align: center;
        margin: 0.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 15px rgba(0, 0, 0, 0.2);
    }
    
    .loading-animation {
        display: flex;
        justify-content: center;
        align-items: center;
        height: 200px;
    }
    </style>
    """, unsafe_allow_html=True)

def load_lottie_url(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

def load_lottie_file(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)

# Load animations
def load_animations():
    lottie_ai = load_lottie_url("https://assets1.lottiefiles.com/packages/lf20_gn0tojcq.json")
    lottie_upload = load_lottie_url("https://assets1.lottiefiles.com/packages/lf20_kxsd2ytq.json")
    lottie_processing = load_lottie_url("https://assets1.lottiefiles.com/packages/lf20_tno6cg2g.json")
    lottie_results = load_lottie_url("https://assets1.lottiefiles.com/packages/lf20_ukaaBc.json")
    
    return {
        "ai": lottie_ai,
        "upload": lottie_upload,
        "processing": lottie_processing,
        "results": lottie_results
    }

# Header section
def header_section():
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("<h1 class='main-header'>🤖 AI Model Hub</h1>", unsafe_allow_html=True)
        st.markdown("### Upload your data and get AI-powered insights in seconds")

# Sidebar
def sidebar_section(animations):
    with st.sidebar:
        st_lottie(animations["ai"], height=150, key="sidebar_ai")
        st.markdown("## Model Selection")
        
        model_option = st.selectbox(
            "Choose your AI model:",
            ["Image Classification", "Object Detection", "Text Analysis", "Data Prediction"]
        )
        
        st.markdown("---")
        st.markdown("## Settings")
        
        confidence_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.05
        )
        
        batch_size = st.slider(
            "Batch Size",
            min_value=1,
            max_value=10,
            value=1,
            step=1
        )
        
        st.markdown("---")
        st.markdown("### About")
        st.info("This platform allows you to easily deploy and test AI models with a modern, interactive interface.")
        
        return model_option, confidence_threshold, batch_size

# Upload section
def upload_section(animations):
    st.markdown("## 📤 Upload Your Data")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("<div class='upload-box'>", unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Drag and drop your file here",
            type=['jpg', 'jpeg', 'png', 'txt', 'csv', 'json'],
            help="Supported formats: Images (JPG, PNG), Text files, CSV, JSON"
        )
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        if animations["upload"]:
            st_lottie(animations["upload"], height=200, key="upload_animation")
    
    return uploaded_file

# Processing simulation
def process_data(uploaded_file, model_option):
    # Simulate processing time
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i in range(100):
        progress_bar.progress(i + 1)
        status_text.text(f"Processing... {i+1}%")
        time.sleep(0.02)
    
    status_text.text("Processing complete!")
    
    # Simulate model results based on file type and model
    if uploaded_file.type.startswith('image'):
        return simulate_image_results()
    elif uploaded_file.type == 'text/plain':
        return simulate_text_results()
    elif uploaded_file.name.endswith('.csv'):
        return simulate_csv_results()
    else:
        return simulate_general_results()

# Simulated results for different file types
def simulate_image_results():
    # Generate sample image classification results
    classes = ['Cat', 'Dog', 'Bird', 'Car', 'Person', 'Building']
    probabilities = np.random.dirichlet(np.ones(6), size=1)[0]
    
    results = {
        "type": "image_classification",
        "predictions": [
            {"class": cls, "probability": float(prob)} 
            for cls, prob in zip(classes, probabilities)
        ],
        "top_prediction": classes[np.argmax(probabilities)],
        "confidence": float(np.max(probabilities))
    }
    
    return results

def simulate_text_results():
    # Generate sample text analysis results
    sentiment = np.random.choice(['Positive', 'Negative', 'Neutral'])
    entities = [
        {"entity": "Apple", "type": "ORG", "confidence": 0.95},
        {"entity": "Tim Cook", "type": "PERSON", "confidence": 0.89},
        {"entity": "California", "type": "GPE", "confidence": 0.92}
    ]
    
    results = {
        "type": "text_analysis",
        "sentiment": sentiment,
        "entities": entities,
        "key_phrases": ["artificial intelligence", "machine learning", "neural networks"]
    }
    
    return results

def simulate_csv_results():
    # Generate sample data prediction results
    dates = pd.date_range(start='2023-01-01', periods=30, freq='D')
    values = 100 + np.cumsum(np.random.normal(0, 5, 30))
    
    results = {
        "type": "data_prediction",
        "historical_data": {
            "dates": dates.strftime('%Y-%m-%d').tolist(),
            "values": values.tolist()
        },
        "predictions": {
            "dates": pd.date_range(start='2023-01-31', periods=7, freq='D').strftime('%Y-%m-%d').tolist(),
            "values": (values[-1] + np.cumsum(np.random.normal(0, 3, 7))).tolist()
        }
    }
    
    return results

def simulate_general_results():
    return {
        "type": "general_analysis",
        "confidence": 0.87,
        "processing_time": "2.3 seconds",
        "model_used": "Custom Deep Learning Model"
    }

# Results display
def display_results(results, animations):
    st.markdown("## 📊 Results")
    
    if animations["results"]:
        st_lottie(animations["results"], height=150, key="results_animation")
    
    st.markdown("<div class='result-box'>", unsafe_allow_html=True)
    
    if results["type"] == "image_classification":
        display_image_results(results)
    elif results["type"] == "text_analysis":
        display_text_results(results)
    elif results["type"] == "data_prediction":
        display_data_results(results)
    else:
        display_general_results(results)
    
    st.markdown("</div>", unsafe_allow_html=True)

def display_image_results(results):
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Top Prediction")
        st.markdown(f"**Class:** {results['top_prediction']}")
        st.markdown(f"**Confidence:** {results['confidence']:.2%}")
        
        # Confidence meter
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = results['confidence'] * 100,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Confidence Level"},
            gauge = {
                'axis': {'range': [0, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 50], 'color': "lightgray"},
                    {'range': [50, 75], 'color': "gray"},
                    {'range': [75, 100], 'color': "lightblue"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### All Predictions")
        
        # Create a bar chart of probabilities
        classes = [pred['class'] for pred in results['predictions']]
        probabilities = [pred['probability'] for pred in results['predictions']]
        
        fig = px.bar(
            x=probabilities, 
            y=classes, 
            orientation='h',
            labels={'x': 'Probability', 'y': 'Class'},
            color=probabilities,
            color_continuous_scale='Viridis'
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

def display_text_results(results):
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Sentiment Analysis")
        
        # Sentiment gauge
        sentiment_values = {'Positive': 1, 'Neutral': 0, 'Negative': -1}
        sentiment_value = sentiment_values[results['sentiment']]
        
        fig = go.Figure(go.Indicator(
            mode = "number+gauge",
            value = sentiment_value,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Sentiment"},
            gauge = {
                'shape': "bullet",
                'axis': {'range': [-1, 1]},
                'threshold': {
                    'line': {'color': "black", 'width': 2},
                    'thickness': 0.75,
                    'value': sentiment_value
                },
                'steps': [
                    {'range': [-1, -0.33], 'color': "lightcoral"},
                    {'range': [-0.33, 0.33], 'color': "lightyellow"},
                    {'range': [0.33, 1], 'color': "lightgreen"}
                ],
                'bar': {'color': "black"}
            }
        ))
        fig.update_layout(height=250)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### Named Entities")
        
        entities_df = pd.DataFrame(results['entities'])
        st.dataframe(entities_df, use_container_width=True)
        
        st.markdown("### Key Phrases")
        for phrase in results['key_phrases']:
            st.markdown(f"- {phrase}")

def display_data_results(results):
    st.markdown("### Data Analysis & Prediction")
    
    # Create a time series plot with historical data and predictions
    fig = make_subplots(rows=1, cols=1)
    
    # Historical data
    fig.add_trace(
        go.Scatter(
            x=results['historical_data']['dates'],
            y=results['historical_data']['values'],
            mode='lines',
            name='Historical Data',
            line=dict(color='blue')
        )
    )
    
    # Predictions
    fig.add_trace(
        go.Scatter(
            x=results['predictions']['dates'],
            y=results['predictions']['values'],
            mode='lines+markers',
            name='Predictions',
            line=dict(color='red', dash='dash')
        )
    )
    
    fig.update_layout(
        title="Historical Data and Future Predictions",
        xaxis_title="Date",
        yaxis_title="Value",
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)

def display_general_results(results):
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.metric(
            label="Confidence Score",
            value=f"{results['confidence']:.2%}",
            delta="High" if results['confidence'] > 0.8 else "Medium"
        )
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.metric(
            label="Processing Time",
            value=results['processing_time'],
            delta="Fast" if "2." in results['processing_time'] else "Standard"
        )
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col3:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.metric(
            label="Model Used",
            value=results['model_used'],
            delta="Custom"
        )
        st.markdown("</div>", unsafe_allow_html=True)

# Footer
def footer_section():
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    with col2:
        st.markdown(
            "<p style='text-align: center; color: gray;'>"
            "AI Model Hub • Powered by Streamlit • "
            "<a href='https://github.com/your-repo' target='_blank'>GitHub</a>"
            "</p>", 
            unsafe_allow_html=True
        )

# Main app
def main():
    # Apply custom CSS
    local_css()
    
    # Load animations
    animations = load_animations()
    
    # Header
    header_section()
    
    # Sidebar
    model_option, confidence_threshold, batch_size = sidebar_section(animations)
    
    # Upload section
    uploaded_file = upload_section(animations)
    
    # Process and display results if file is uploaded
    if uploaded_file is not None:
        if st.button("🚀 Process with AI", use_container_width=True):
            # Show processing animation
            with st.spinner(''):
                if animations["processing"]:
                    st_lottie(animations["processing"], height=200, key="processing_animation")
            
            # Process the data
            results = process_data(uploaded_file, model_option)
            
            # Display results
            display_results(results, animations)
    
    # Footer
    footer_section()

if __name__ == "__main__":
    main()
