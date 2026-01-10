import streamlit as st
import numpy as np
from transformers import pipeline
import tempfile
import os
import librosa
import soundfile as sf

# Page config
st.set_page_config(
    page_title="Autism Speech Delay Detector",
    page_icon="🎤",
    layout="centered"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        background: linear-gradient(to bottom right, #f0f9ff, #e0f2fe);
    }
    .metric-card {
        background: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 10px 0;
    }
    .warning-box {
        background: #fef3c7;
        border-left: 4px solid #f59e0b;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .success-box {
        background: #d1fae5;
        border-left: 4px solid #10b981;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'analyzed' not in st.session_state:
    st.session_state.analyzed = False

# Load Distil-Whisper model
@st.cache_resource
def load_model():
    """Load Distil-Whisper model with word timestamps"""
    try:
        pipe = pipeline(
            "automatic-speech-recognition",
            model="distil-whisper/distil-large-v3",
            return_timestamps="word"
        )
        return pipe
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def calculate_speech_metrics(result):
    """Calculate autism-related speech delay metrics"""
    
    if not result or 'chunks' not in result:
        return None
    
    chunks = result['chunks']
    
    if len(chunks) == 0:
        return None
    
    # Extract word timings
    word_timings = []
    for chunk in chunks:
        if 'timestamp' in chunk and chunk['timestamp'][0] is not None:
            word_timings.append({
                'word': chunk['text'].strip(),
                'start': chunk['timestamp'][0],
                'end': chunk['timestamp'][1] if chunk['timestamp'][1] is not None else chunk['timestamp'][0] + 0.5
            })
    
    if len(word_timings) == 0:
        return None
    
    # Calculate metrics
    metrics = {}
    
    # 1. Initial response time (key autism indicator)
    metrics['initial_response_time'] = word_timings[0]['start']
    
    # 2. Calculate inter-word gaps
    gaps = []
    for i in range(len(word_timings) - 1):
        gap = word_timings[i + 1]['start'] - word_timings[i]['end']
        if gap >= 0:  # Only positive gaps
            gaps.append(gap)
    
    if gaps:
        metrics['average_pause'] = np.mean(gaps)
        metrics['max_pause'] = np.max(gaps)
        metrics['pause_variance'] = np.std(gaps)
        metrics['long_pauses_count'] = sum(1 for g in gaps if g > 1.0)
    else:
        metrics['average_pause'] = 0
        metrics['max_pause'] = 0
        metrics['pause_variance'] = 0
        metrics['long_pauses_count'] = 0
    
    # 3. Speech rate
    total_duration = word_timings[-1]['end'] - word_timings[0]['start']
    word_count = len(word_timings)
    metrics['words_per_minute'] = (word_count / total_duration) * 60 if total_duration > 0 else 0
    
    # 4. Total duration
    metrics['total_duration'] = total_duration
    
    # 5. Detect repetitions (echolalia indicator)
    words = [w['word'].lower() for w in word_timings]
    repetitions = {}
    for i in range(len(words) - 1):
        bigram = f"{words[i]} {words[i+1]}"
        if bigram in repetitions:
            repetitions[bigram] += 1
        else:
            repetitions[bigram] = 1
    
    metrics['repeated_phrases'] = {k: v for k, v in repetitions.items() if v > 1}
    
    return metrics, word_timings

def analyze_delays(metrics):
    """Analyze if speech patterns indicate potential delays"""
    flags = []
    
    if metrics['initial_response_time'] > 2.0:
        flags.append({
            'type': 'warning',
            'message': f"⚠️ Delayed initial response: {metrics['initial_response_time']:.2f}s (typical: <2s)"
        })
    
    if metrics['average_pause'] > 0.8:
        flags.append({
            'type': 'warning',
            'message': f"⚠️ Longer than typical pauses: {metrics['average_pause']:.2f}s average"
        })
    
    if metrics['pause_variance'] > 0.5:
        flags.append({
            'type': 'warning',
            'message': f"⚠️ Inconsistent pause patterns detected (variance: {metrics['pause_variance']:.2f})"
        })
    
    if metrics['long_pauses_count'] > 3:
        flags.append({
            'type': 'warning',
            'message': f"⚠️ Multiple long pauses detected ({metrics['long_pauses_count']} pauses >1s)"
        })
    
    if metrics['words_per_minute'] < 80:
        flags.append({
            'type': 'warning',
            'message': f"⚠️ Slow speech rate: {metrics['words_per_minute']:.1f} words/min (typical: 100-130)"
        })
    
    if metrics['repeated_phrases']:
        flags.append({
            'type': 'info',
            'message': f"ℹ️ Repeated phrases detected (possible echolalia): {list(metrics['repeated_phrases'].keys())[:3]}"
        })
    
    if not flags:
        flags.append({
            'type': 'success',
            'message': "✓ No significant delay patterns detected"
        })
    
    return flags

# Header
st.title("🎤 Autism Speech Delay Detector")
st.markdown("Analyze speech patterns to identify potential delays in autistic children")

# Info section
with st.expander("ℹ️ How it works"):
    st.markdown("""
    This tool analyzes speech audio to detect patterns associated with autism:
    
    **Key Indicators:**
    - **Initial Response Time**: Time before first word (delayed if >2s)
    - **Pause Patterns**: Frequency and duration of pauses
    - **Speech Rate**: Words per minute (typical: 100-130)
    - **Repetitions**: Echolalia detection (repeated phrases)
    - **Consistency**: Variance in pause patterns
    
    **Note:** This is a screening tool, not a diagnostic instrument. Always consult healthcare professionals.
    """)

# Load model
with st.spinner("Loading speech recognition model..."):
    model = load_model()

if model is None:
    st.error("Failed to load model. Please refresh the page.")
    st.stop()

# File upload
st.markdown("### Upload Audio Recording")
audio_file = st.file_uploader(
    "Choose an audio file (WAV, MP3, M4A)",
    type=['wav', 'mp3', 'm4a'],
    help="Upload a recording of speech. For best results, use clear audio with minimal background noise."
)

if audio_file is not None:
    # Display audio player
    st.audio(audio_file)
    
    # Analyze button
    if st.button("🔍 Analyze Speech Patterns", type="primary"):
        with st.spinner("Analyzing speech patterns..."):
            try:
                # Save uploaded file to temp location
                with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(audio_file.name)[1]) as tmp_file:
                    tmp_file.write(audio_file.read())
                    tmp_path = tmp_file.name
                
                # Convert to WAV if needed (for compatibility)
                audio_data, sample_rate = librosa.load(tmp_path, sr=16000)
                wav_path = tmp_path.replace(os.path.splitext(tmp_path)[1], '.wav')
                sf.write(wav_path, audio_data, sample_rate)
                
                # Transcribe with word-level timestamps
                result = model(wav_path)
                
                # Clean up temp files
                os.unlink(tmp_path)
                if wav_path != tmp_path:
                    os.unlink(wav_path)
                
                # Calculate metrics
                analysis = calculate_speech_metrics(result)
                
                if analysis is None:
                    st.error("Could not analyze the audio. Please ensure the file contains clear speech.")
                else:
                    metrics, word_timings = analysis
                    st.session_state.analyzed = True
                    
                    # Display transcript
                    st.markdown("### 📝 Transcript")
                    st.info(result['text'])
                    
                    # Display metrics
                    st.markdown("### 📊 Speech Analysis")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric(
                            "Initial Response",
                            f"{metrics['initial_response_time']:.2f}s",
                            delta="Delayed" if metrics['initial_response_time'] > 2.0 else "Normal",
                            delta_color="inverse"
                        )
                    
                    with col2:
                        st.metric(
                            "Average Pause",
                            f"{metrics['average_pause']:.2f}s",
                            delta="Long" if metrics['average_pause'] > 0.8 else "Normal",
                            delta_color="inverse"
                        )
                    
                    with col3:
                        st.metric(
                            "Speech Rate",
                            f"{metrics['words_per_minute']:.0f} wpm",
                            delta="Slow" if metrics['words_per_minute'] < 80 else "Normal",
                            delta_color="inverse"
                        )
                    
                    col4, col5, col6 = st.columns(3)
                    
                    with col4:
                        st.metric("Max Pause", f"{metrics['max_pause']:.2f}s")
                    
                    with col5:
                        st.metric("Long Pauses (>1s)", metrics['long_pauses_count'])
                    
                    with col6:
                        st.metric("Total Duration", f"{metrics['total_duration']:.1f}s")
                    
                    # Analysis flags
                    st.markdown("### 🔍 Pattern Analysis")
                    flags = analyze_delays(metrics)
                    
                    for flag in flags:
                        if flag['type'] == 'warning':
                            st.markdown(f'<div class="warning-box">{flag["message"]}</div>', unsafe_allow_html=True)
                        elif flag['type'] == 'success':
                            st.markdown(f'<div class="success-box">{flag["message"]}</div>', unsafe_allow_html=True)
                        else:
                            st.info(flag['message'])
                    
                    # Word-level timeline
                    with st.expander("📈 Detailed Word Timeline"):
                        st.markdown("**Word-by-word timing analysis:**")
                        for i, word in enumerate(word_timings):
                            gap = ""
                            if i > 0:
                                pause = word['start'] - word_timings[i-1]['end']
                                if pause > 0.5:
                                    gap = f" (⏸️ {pause:.2f}s pause)"
                            st.text(f"{word['start']:.2f}s - {word['end']:.2f}s: {word['word']}{gap}")
                    
                    # Recommendations
                    st.markdown("### 💡 Recommendations")
                    if any(f['type'] == 'warning' for f in flags):
                        st.warning("""
                        **Patterns detected that may warrant professional evaluation:**
                        - Consult with a speech-language pathologist
                        - Consider developmental screening
                        - Track patterns over time
                        - This tool is for screening only, not diagnosis
                        """)
                    else:
                        st.success("""
                        **No significant delay patterns detected.**
                        - Continue monitoring development
                        - Encourage regular communication
                        - Consult healthcare provider if concerns arise
                        """)
            
            except Exception as e:
                st.error(f"Error analyzing audio: {str(e)}")
                st.info("Please try with a different audio file or format.")

# Sidebar information
with st.sidebar:
    st.markdown("### About")
    st.markdown("""
    This tool uses Distil-Whisper for speech-to-text with word-level timestamps to analyze:
    
    - Response delays
    - Pause patterns
    - Speech rate
    - Repetitions (echolalia)
    
    Designed to help identify speech patterns associated with autism in children.
    """)
    
    st.markdown("---")
    st.markdown("### Usage Tips")
    st.markdown("""
    - Use clear audio recordings
    - Minimize background noise
    - Record 30-60 seconds of speech
    - Ask simple questions for responses
    """)
    
    st.markdown("---")
    st.markdown("**⚠️ Disclaimer:** Not a diagnostic tool. Consult healthcare professionals for proper evaluation.")
