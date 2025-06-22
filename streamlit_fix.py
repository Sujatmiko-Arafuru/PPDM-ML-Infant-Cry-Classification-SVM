"""
Streamlit App untuk Klasifikasi Tangisan Bayi
Deployment Real-time menggunakan SVM

Aplikasi ini memungkinkan user untuk:
1. Upload file audio (.wav/.mp3)
2. Play/preview audio yang diupload
3. Mendapatkan prediksi jenis tangisan bayi
4. Melihat statistik dan confidence score
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import librosa
import soundfile as sf
from pydub import AudioSegment
import tempfile
import os
import sys
import time
import io

# Import modules yang diperlukan
sys.path.append('.')
from deploy_preprocess import AudioPreprocessor, get_label_name, validate_audio_file
from contoh_deploy import BabyCryClassifier
from preprocess_config import *
from svm_classes import KernelSVM, OneVsRestSVM

# Konfigurasi halaman
st.set_page_config(
    page_title="Baby Cry Classifier",
    page_icon="👶",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS untuk styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #ff7f0e;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .prediction-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #28a745;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #ffc107;
        margin: 1rem 0;
    }
    .error-box {
        background-color: #f8d7da;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #dc3545;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load model dan scaler (cached untuk performance)"""
    try:
        classifier = BabyCryClassifier(
            model_path="best_svm_model.pkl",
            scaler_path="dataset_preprocessed/stage3/scaler.joblib"
        )
        return classifier, None
    except Exception as e:
        return None, str(e)

def convert_mp3_to_wav(mp3_file):
    """Convert MP3 file ke WAV format"""
    try:
        # Load MP3 menggunakan pydub
        audio = AudioSegment.from_mp3(mp3_file)
        
        # Convert ke WAV
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            audio.export(tmp_file.name, format="wav")
            return tmp_file.name
    except Exception as e:
        st.error(f"Error converting MP3 to WAV: {str(e)}")
        return None

def save_uploaded_file(uploaded_file):
    """Simpan file yang diupload ke temporary file"""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            return tmp_file.name
    except Exception as e:
        st.error(f"Error saving uploaded file: {str(e)}")
        return None

def get_audio_info(audio_path):
    """Mendapatkan informasi audio file"""
    try:
        audio, sr = librosa.load(audio_path, sr=None)
        duration = len(audio) / sr
        
        info = {
            "Duration": f"{duration:.2f} seconds",
            "Sample Rate": f"{sr} Hz",
            "Channels": "Mono" if len(audio.shape) == 1 else "Stereo",
            "Samples": len(audio),
            "File Size": f"{os.path.getsize(audio_path) / 1024:.1f} KB"
        }
        return info, audio, sr
    except Exception as e:
        return None, None, None

def create_waveform_plot(audio, sr, title="Audio Waveform"):
    """Membuat plot waveform audio"""
    time_axis = np.linspace(0, len(audio) / sr, len(audio))
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=time_axis,
        y=audio,
        mode='lines',
        name='Waveform',
        line=dict(color='#1f77b4', width=1)
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Time (seconds)",
        yaxis_title="Amplitude",
        height=300,
        showlegend=False
    )
    
    return fig

def create_feature_plot(features, feature_names):
    """Membuat plot untuk fitur yang diekstrak"""
    # Bagi fitur menjadi kategori
    time_features = features[:4]
    freq_features = features[4:8]
    mfcc_features = features[8:]
    
    time_names = feature_names[:4]
    freq_names = feature_names[4:8]
    mfcc_names = feature_names[8:]
    
    # Buat subplot
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Time Domain Features', 'Frequency Domain Features', 
                       'MFCC Features (Mean)', 'MFCC Features (Std)'),
        specs=[[{"type": "bar"}, {"type": "bar"}],
               [{"type": "bar"}, {"type": "bar"}]]
    )
    
    # Time domain features
    fig.add_trace(go.Bar(
        x=time_names,
        y=time_features,
        name='Time Domain',
        marker_color='#1f77b4'
    ), row=1, col=1)
    
    # Frequency domain features
    fig.add_trace(go.Bar(
        x=freq_names,
        y=freq_features,
        name='Frequency Domain',
        marker_color='#ff7f0e'
    ), row=1, col=2)
    
    # MFCC features (split mean and std)
    mfcc_mean = mfcc_features[::2]  # Every even index (mean)
    mfcc_std = mfcc_features[1::2]  # Every odd index (std)
    mfcc_indices = list(range(1, 14))  # MFCC 1-13
    
    fig.add_trace(go.Bar(
        x=mfcc_indices,
        y=mfcc_mean,
        name='MFCC Mean',
        marker_color='#2ca02c'
    ), row=2, col=1)
    
    fig.add_trace(go.Bar(
        x=mfcc_indices,
        y=mfcc_std,
        name='MFCC Std',
        marker_color='#d62728'
    ), row=2, col=2)
    
    fig.update_layout(
        height=600,
        showlegend=False,
        title_text="Extracted Audio Features"
    )
    
    return fig

def create_probability_plot(probabilities):
    """Membuat plot probabilitas prediksi"""
    labels = list(probabilities.keys())
    values = list(probabilities.values())
    
    # Sort berdasarkan probabilitas
    sorted_data = sorted(zip(labels, values), key=lambda x: x[1], reverse=True)
    labels, values = zip(*sorted_data)
    
    # Buat color mapping
    colors = ['#1f77b4' if i == 0 else '#ff7f0e' for i in range(len(labels))]
    
    fig = go.Figure(data=[
        go.Bar(
            x=labels,
            y=values,
            marker_color=colors,
            text=[f'{v:.1%}' for v in values],
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title="Prediction Probabilities",
        xaxis_title="Cry Type",
        yaxis_title="Probability",
        height=400,
        yaxis=dict(tickformat='.0%')
    )
    
    return fig

def main():
    """Main function untuk Streamlit app"""
    
    # Header
    st.markdown('<div class="main-header">👶 Baby Cry Classifier</div>', unsafe_allow_html=True)
    st.markdown("**Klasifikasi Tangisan Bayi menggunakan Support Vector Machine (SVM)**")
    
    # Sidebar untuk informasi
    with st.sidebar:
        st.header("📋 Informasi Model")
        st.markdown("""
        **Model**: SVM with RBF Kernel  
        **Accuracy**: 82.33%  
        **Classes**: 5 jenis tangisan  
        **Features**: 34 fitur audio  
        
        **Jenis Tangisan:**
        - 🤕 Belly Pain (Sakit Perut)
        - 🍼 Burping (Bersendawa)  
        - 😣 Discomfort (Tidak Nyaman)
        - 🍼 Hungry (Lapar)
        - 😴 Tired (Lelah)
        """)
        
        st.header("📊 Parameter Preprocessing")
        st.markdown(f"""
        - **Sample Rate**: {TARGET_SAMPLE_RATE} Hz
        - **Duration**: {SEGMENT_DURATION} detik  
        - **Frame Length**: {FRAME_LENGTH}
        - **MFCC Coefficients**: {N_MFCC}
        - **Total Features**: {N_TOTAL_FEATURES}
        """)
    
    # Load model
    with st.spinner("Loading model..."):
        classifier, error = load_model()
    
    if error:
        st.markdown(f'<div class="error-box">❌ <strong>Error loading model:</strong> {error}</div>', 
                   unsafe_allow_html=True)
        st.stop()
    
    st.markdown('<div class="success-box">✅ <strong>Model berhasil dimuat!</strong></div>', 
               unsafe_allow_html=True)
    
    # File upload
    st.markdown('<div class="sub-header">📤 Upload Audio File</div>', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Choose an audio file",
        type=['wav', 'mp3'],
        help="Upload file audio tangisan bayi dalam format WAV atau MP3"
    )
    
    if uploaded_file is not None:
        # Simpan file yang diupload
        file_path = save_uploaded_file(uploaded_file)
        
        if file_path is None:
            st.error("Failed to save uploaded file")
            st.stop()
        
        # Convert MP3 ke WAV jika diperlukan
        if uploaded_file.name.lower().endswith('.mp3'):
            st.info("🔄 Converting MP3 to WAV...")
            wav_path = convert_mp3_to_wav(file_path)
            if wav_path:
                file_path = wav_path
                st.success("✅ Conversion completed!")
            else:
                st.error("❌ Failed to convert MP3 to WAV")
                st.stop()
        
        # Validasi file audio
        is_valid, message = validate_audio_file(file_path)
        if not is_valid:
            st.markdown(f'<div class="error-box">❌ <strong>File tidak valid:</strong> {message}</div>', 
                       unsafe_allow_html=True)
            st.stop()
        
        # Tampilkan informasi file
        st.markdown('<div class="sub-header">📊 Informasi Audio</div>', unsafe_allow_html=True)
        
        audio_info, audio_data, sample_rate = get_audio_info(file_path)
        if audio_info:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**File Information:**")
                for key, value in audio_info.items():
                    st.write(f"- **{key}**: {value}")
            
            with col2:
                st.markdown("**Audio Player:**")
                st.audio(uploaded_file.getvalue(), format='audio/wav')
        
        # Tampilkan waveform
        if audio_data is not None:
            st.markdown('<div class="sub-header">🌊 Audio Waveform</div>', unsafe_allow_html=True)
            waveform_fig = create_waveform_plot(audio_data, sample_rate)
            st.plotly_chart(waveform_fig, use_container_width=True)
        
        # Tombol prediksi
        st.markdown('<div class="sub-header">🔮 Prediksi</div>', unsafe_allow_html=True)
        
        if st.button("🚀 Classify Baby Cry", type="primary"):
            with st.spinner("Analyzing audio..."):
                try:
                    # Preprocessing dan prediksi
                    start_time = time.time()
                    result = classifier.predict(file_path, return_probabilities=True)
                    processing_time = time.time() - start_time
                    
                    # Tampilkan hasil prediksi
                    st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric(
                            label="🎯 Predicted Cry Type",
                            value=result['prediction_label'].replace('_', ' ').title(),
                            delta=f"Index: {result['prediction_index']}"
                        )
                    
                    with col2:
                        max_prob = max(result['probabilities'].values())
                        st.metric(
                            label="🎯 Confidence",
                            value=f"{max_prob:.1%}",
                            delta=f"Processing: {processing_time:.2f}s"
                        )
                    
                    with col3:
                        st.metric(
                            label="📊 Features Extracted",
                            value=f"{result['features_shape'][0]}",
                            delta="Features"
                        )
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Tampilkan probabilitas semua kelas
                    st.markdown('<div class="sub-header">📊 Probabilitas Semua Kelas</div>', unsafe_allow_html=True)
                    
                    prob_fig = create_probability_plot(result['probabilities'])
                    st.plotly_chart(prob_fig, use_container_width=True)
                    
                    # Tabel probabilitas
                    prob_df = pd.DataFrame([
                        {
                            "Cry Type": key.replace('_', ' ').title(),
                            "Probability": f"{value:.4f}",
                            "Percentage": f"{value:.1%}",
                            "Confidence": "High" if value > 0.5 else "Medium" if value > 0.3 else "Low"
                        }
                        for key, value in sorted(result['probabilities'].items(), 
                                               key=lambda x: x[1], reverse=True)
                    ])
                    
                    st.dataframe(prob_df, use_container_width=True)
                    
                    # Ekstrak dan tampilkan fitur
                    st.markdown('<div class="sub-header">🔍 Analisis Fitur Audio</div>', unsafe_allow_html=True)
                    
                    # Dapatkan fitur yang diekstrak
                    preprocessor = AudioPreprocessor(SCALER_PATH)
                    audio_processed, sr = preprocessor.load_and_preprocess_audio(file_path)
                    features = preprocessor.extract_features(audio_processed, sr)
                    
                    # Plot fitur
                    feature_fig = create_feature_plot(features, ALL_FEATURES)
                    st.plotly_chart(feature_fig, use_container_width=True)
                    
                    # Tabel fitur detail
                    with st.expander("📋 Detail Fitur (Raw Values)"):
                        feature_df = pd.DataFrame({
                            "Feature Name": ALL_FEATURES,
                            "Value": features,
                            "Category": (["Time Domain"] * 4 + 
                                       ["Frequency Domain"] * 4 + 
                                       ["MFCC"] * 26)
                        })
                        st.dataframe(feature_df, use_container_width=True)
                    
                    # Interpretasi hasil
                    st.markdown('<div class="sub-header">💡 Interpretasi Hasil</div>', unsafe_allow_html=True)
                    
                    predicted_type = result['prediction_label']
                    confidence = max(result['probabilities'].values())
                    
                    interpretation = {
                        'belly_pain': "Bayi kemungkinan mengalami sakit perut. Coba periksa apakah bayi perlu bersendawa atau ganti popok.",
                        'burping': "Bayi perlu bersendawa. Coba gendong bayi dalam posisi tegak dan tepuk-tepuk punggungnya perlahan.",
                        'discomfort': "Bayi merasa tidak nyaman. Periksa suhu ruangan, pakaian, atau posisi bayi.",
                        'hungry': "Bayi lapar dan perlu diberi makan. Siapkan susu atau makanan untuk bayi.",
                        'tired': "Bayi lelah dan perlu tidur. Coba ciptakan suasana tenang dan nyaman untuk bayi."
                    }
                    
                    confidence_level = "tinggi" if confidence > 0.6 else "sedang" if confidence > 0.4 else "rendah"
                    
                    st.markdown(f"""
                    **Prediksi**: {predicted_type.replace('_', ' ').title()}  
                    **Tingkat Kepercayaan**: {confidence:.1%} ({confidence_level})  
                    **Saran**: {interpretation.get(predicted_type, "Monitoring lebih lanjut diperlukan.")}
                    """)
                    
                    if confidence < 0.5:
                        st.warning("⚠️ Tingkat kepercayaan rendah. Pertimbangkan untuk menggunakan audio dengan kualitas lebih baik atau durasi lebih panjang.")
                
                except Exception as e:
                    st.markdown(f'<div class="error-box">❌ <strong>Error during prediction:</strong> {str(e)}</div>', 
                               unsafe_allow_html=True)
        
        # Cleanup temporary files
        try:
            if os.path.exists(file_path):
                os.unlink(file_path)
        except:
            pass
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <small>
        Baby Cry Classifier v1.0 | Powered by SVM & Streamlit<br>
        Developed for infant care assistance
        </small>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main() 