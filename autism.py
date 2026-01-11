import streamlit as st
import numpy as np
import librosa
import soundfile as sf
import tempfile
import os
from groq import Groq
import requests
import json

# Configuration
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
OPENROUTER_API_KEY = st.secrets["OPENROUTER_API_KEY"]
WHISPER_MODEL = "whisper-large-v3"
SAMPLE_RATE = 16000

# Clinical thresholds
INITIAL_RESPONSE_THRESHOLD = 2.0
MIN_SPEECH_RATE = 80
MAX_AVG_WORD_DURATION = 1.5

st.set_page_config(
    page_title="Speech Pattern Analyzer",
    page_icon="🎙️",
    layout="wide"
)

st.title("Speech Pattern Analyzer")
st.write("Upload an audio file to analyze speech patterns and timing metrics.")


@st.cache_resource
def get_groq_client():
    return Groq(api_key=GROQ_API_KEY)


def transcribe_audio(audio_path):
    """Transcribe audio file using Groq Whisper API."""
    client = get_groq_client()
    try:
        with open(audio_path, 'rb') as audio_file:
            transcription = client.audio.transcriptions.create(
                file=(audio_path, audio_file.read()),
                model=WHISPER_MODEL,
                response_format="verbose_json",
                timestamp_granularities=["word"],
                language="en"
            )
        
        result = {'text': transcription.text, 'chunks': []}
        
        if hasattr(transcription, 'words') and transcription.words:
            for word_data in transcription.words:
                if isinstance(word_data, dict):
                    result['chunks'].append({
                        'text': word_data.get('word', ''),
                        'timestamp': [word_data.get('start', 0), word_data.get('end', 0)]
                    })
                else:
                    result['chunks'].append({
                        'text': word_data.word,
                        'timestamp': [word_data.start, word_data.end]
                    })
        
        return result
    except Exception as e:
        st.error(f"Transcription error: {str(e)}")
        return None


def calculate_speech_metrics(result):
    """Calculate speech metrics from transcription."""
    if not result or 'chunks' not in result or len(result['chunks']) == 0:
        return None

    word_timings = [
        {
            'word': chunk['text'].strip(),
            'start': chunk['timestamp'][0],
            'end': chunk['timestamp'][1] if chunk['timestamp'][1] else chunk['timestamp'][0] + 0.5
        }
        for chunk in result['chunks']
        if 'timestamp' in chunk and chunk['timestamp'][0] is not None
    ]

    if not word_timings:
        return None

    total_duration = word_timings[-1]['end'] - word_timings[0]['start']
    word_count = len(word_timings)
    word_durations = [w['end'] - w['start'] for w in word_timings]
    
    words = [w['word'].lower() for w in word_timings]
    repetitions = {}
    for i in range(len(words) - 1):
        bigram = f"{words[i]} {words[i+1]}"
        repetitions[bigram] = repetitions.get(bigram, 0) + 1
    
    metrics = {
        'initial_response_time': word_timings[0]['start'],
        'words_per_minute': (word_count / total_duration) * 60 if total_duration > 0 else 0,
        'total_duration': total_duration,
        'average_word_duration': np.mean(word_durations),
        'max_word_duration': np.max(word_durations),
        'word_count': word_count,
        'repeated_phrases': {k: v for k, v in repetitions.items() if v > 1}
    }

    return metrics, word_timings


def analyze_delays(metrics):
    """Analyze metrics against clinical thresholds."""
    flags = []

    if metrics['initial_response_time'] > INITIAL_RESPONSE_THRESHOLD:
        flags.append(f"WARNING: Delayed initial response: {metrics['initial_response_time']:.2f}s (typical: <{INITIAL_RESPONSE_THRESHOLD}s)")

    if metrics['words_per_minute'] < MIN_SPEECH_RATE:
        flags.append(f"WARNING: Reduced speech rate: {metrics['words_per_minute']:.1f} wpm (typical: 100-130 wpm)")

    if metrics['average_word_duration'] > MAX_AVG_WORD_DURATION:
        flags.append(f"WARNING: Extended word durations: {metrics['average_word_duration']:.2f}s avg (typical: <{MAX_AVG_WORD_DURATION}s)")

    if metrics['repeated_phrases']:
        phrases = list(metrics['repeated_phrases'].keys())[:3]
        flags.append(f"NOTE: Repeated phrases detected (possible echolalia): {phrases}")

    if not flags:
        flags.append("RESULT: No significant delay patterns detected")

    return flags


def get_ai_analysis(metrics, flags, transcript):
    """Get AI analysis of the speech patterns using OpenRouter."""
    try:
        openrouter_key = st.secrets["OPENROUTER_API_KEY"]
    except:
        return "AI analysis unavailable. Please add OPENROUTER_API_KEY to Streamlit secrets."
    
    prompt = f"""You are a speech pattern analyst. Analyze the following speech metrics and provide insights:

TRANSCRIPT: {transcript}

METRICS:
- Initial Response Time: {metrics['initial_response_time']:.2f} seconds
- Speech Rate: {metrics['words_per_minute']:.0f} words per minute
- Average Word Duration: {metrics['average_word_duration']:.2f} seconds
- Maximum Word Duration: {metrics['max_word_duration']:.2f} seconds
- Total Duration: {metrics['total_duration']:.1f} seconds
- Word Count: {metrics['word_count']}

PATTERN ANALYSIS:
{chr(10).join(flags)}

REPEATED PHRASES: {list(metrics['repeated_phrases'].keys()) if metrics['repeated_phrases'] else 'None'}

Please provide:
1. What these numbers indicate about the speech pattern
2. How the metrics relate to typical speech development
3. Notable observations about timing, pacing, and fluency
4. Any patterns that stand out

Be analytical and educational, not diagnostic. Focus on explaining what the data shows."""

    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {openrouter_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": "deepseek/deepseek-r1-0528:free",
                "messages": [{"role": "user", "content": prompt}]
            }
        )
        
        if response.status_code == 200:
            data = response.json()
            return data['choices'][0]['message']['content']
        else:
            return f"Error getting AI analysis: {response.status_code}"
    
    except Exception as e:
        return f"Error: {str(e)}"


# Main App
uploaded_file = st.file_uploader(
    "Upload Audio File",
    type=['wav', 'mp3', 'm4a', 'flac', 'ogg'],
    help="Maximum file size: 25MB"
)

if uploaded_file is not None:
    st.audio(uploaded_file)
    
    if st.button("Analyze Speech", type="primary"):
        with st.spinner("Processing audio..."):
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                audio_data, _ = librosa.load(uploaded_file, sr=SAMPLE_RATE)
                sf.write(tmp_file.name, audio_data, SAMPLE_RATE)
                tmp_path = tmp_file.name
            
            # Transcribe
            result = transcribe_audio(tmp_path)
            
            if result:
                st.success("Transcription complete!")
                
                # Display transcript
                st.subheader("Transcript")
                st.write(result['text'])
                
                # Calculate metrics
                analysis = calculate_speech_metrics(result)
                
                if analysis:
                    metrics, word_timings = analysis
                    flags = analyze_delays(metrics)
                    
                    # Display metrics
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Initial Response", f"{metrics['initial_response_time']:.2f}s")
                        st.metric("Speech Rate", f"{metrics['words_per_minute']:.0f} wpm")
                    
                    with col2:
                        st.metric("Avg Word Duration", f"{metrics['average_word_duration']:.2f}s")
                        st.metric("Max Word Duration", f"{metrics['max_word_duration']:.2f}s")
                    
                    with col3:
                        st.metric("Total Duration", f"{metrics['total_duration']:.1f}s")
                        st.metric("Word Count", metrics['word_count'])
                    
                    # Pattern Analysis
                    st.subheader("Pattern Analysis")
                    for flag in flags:
                        if "WARNING" in flag:
                            st.warning(flag)
                        elif "NOTE" in flag:
                            st.info(flag)
                        else:
                            st.success(flag)
                    
                    # AI Analysis
                    st.subheader("AI Analysis")
                    with st.spinner("Generating analysis..."):
                        ai_analysis = get_ai_analysis(metrics, flags, result['text'])
                        st.write(ai_analysis)
                    
                    # Word Timeline
                    with st.expander("View Word Timeline"):
                        for word in word_timings:
                            duration = word['end'] - word['start']
                            st.text(f"{word['start']:6.2f}s - {word['end']:6.2f}s ({duration:.2f}s): {word['word']}")
                else:
                    st.error("Unable to analyze. Ensure audio contains clear speech (5-10s minimum).")
            else:
                st.error("Transcription failed.")
            
            # Cleanup
            os.remove(tmp_path)

st.markdown("---")
st.caption("DISCLAIMER: This is a screening tool only. Professional evaluation required for diagnosis.")
