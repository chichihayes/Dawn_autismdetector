# Autism Speech Pattern Analyzer

A clinical-style speech timing and fluency analysis platform built with Python and Streamlit.

The application analyzes uploaded speech recordings using automatic speech recognition, temporal speech metrics, fluency calculations, repetition analysis, and AI-generated interpretation.

The system is designed for educational and research-oriented speech pattern analysis and is not intended to provide medical diagnosis.

---

# Overview

The application performs:

- Automatic speech transcription
- Word-level timestamp extraction
- Speech timing analysis
- Speech fluency analysis
- Delay detection
- Repetition pattern analysis
- AI-assisted interpretation

The system combines:
- Digital Signal Processing (DSP)
- Speech Analytics
- Temporal Linguistic Analysis
- Clinical-style Speech Metrics
- Large Language Model Interpretation

---

# Core Technologies

- Python
- Streamlit
- NumPy
- Librosa
- SoundFile
- Groq Whisper API
- OpenRouter API

---

# Features

## Audio Processing
- WAV
- MP3
- M4A
- FLAC
- OGG

support.

---

## Speech Transcription

Uses Groq Whisper Large V3 for:
- high-speed transcription
- word-level timestamps
- speech segmentation

---

## Temporal Speech Metrics

The system calculates:

- Initial response latency
- Words per minute
- Average word duration
- Maximum word duration
- Total speech duration
- Word count
- Repeated phrase frequency

---

## AI Analysis

Large language models generate:
- speech pacing interpretation
- fluency observations
- timing analysis
- repetition insights
- educational explanations

---

# Installation

## Requirements

- Python 3.8+
- pip

---

# Install Dependencies

```bash
pip install streamlit numpy librosa soundfile groq requests
```

---

# Streamlit Secrets Configuration

Create:

```bash
.streamlit/secrets.toml
```

Add:

```toml
GROQ_API_KEY = "your_groq_key"
OPENROUTER_API_KEY = "your_openrouter_key"
```

---

# Running the Application

```bash
streamlit run app.py
```

---

# System Architecture

```text
Audio Upload
      ↓
Audio Resampling
      ↓
Whisper Transcription
      ↓
Word Timestamp Extraction
      ↓
Speech Metric Computation
      ↓
Clinical Threshold Analysis
      ↓
AI Interpretation
      ↓
Visualization & Report
```

---

# Mathematical Foundations

# 1. Speech Rate Calculation

Words per minute (WPM) measures speech fluency speed.

Formula:

:contentReference[oaicite:0]{index=0}

Where:
- Word Count = total recognized words
- Speech Duration = total speaking time in seconds

---

# 2. Average Word Duration

Measures articulation pacing.

Formula:

:contentReference[oaicite:1]{index=1}

Where:
- \( End_i \) = word end timestamp
- \( Start_i \) = word start timestamp
- \( n \) = total number of words

---

# 3. Maximum Word Duration

Measures the longest articulation interval.

Formula:

:contentReference[oaicite:2]{index=2}

---

# 4. Initial Response Time

Measures latency before speech begins.

Formula:

:contentReference[oaicite:3]{index=3}

---

# 5. Repetition Detection

The system computes repeated bigrams.

Example:

```text
"I want"
"I want"
```

Algorithm:

```text
Bigram_i = Word_i + Word_{i+1}
```

Repeated frequency:

:contentReference[oaicite:4]{index=4}

---

# Clinical Threshold Logic

The application compares calculated metrics against predefined thresholds.

---

## Initial Response Threshold

```python
INITIAL_RESPONSE_THRESHOLD = 2.0
```

Condition:

:contentReference[oaicite:5]{index=5}

---

## Speech Rate Threshold

```python
MIN_SPEECH_RATE = 80
```

Condition:

:contentReference[oaicite:6]{index=6}

---

## Average Word Duration Threshold

```python
MAX_AVG_WORD_DURATION = 1.5
```

Condition:

:contentReference[oaicite:7]{index=7}

---

# Transcription Engine

The platform uses:

## Groq Whisper Large V3

Model:

```text
whisper-large-v3
```

Capabilities:
- speech-to-text
- word timestamps
- multilingual processing
- low latency inference

---

# AI Interpretation Engine

Uses OpenRouter models for:
- analytical explanations
- fluency interpretation
- pacing analysis
- educational reporting

The AI does not diagnose conditions.

---

# Output Metrics

| Metric | Description |
|---|---|
| Initial Response Time | Delay before speech begins |
| Words Per Minute | Speech fluency speed |
| Average Word Duration | Mean articulation time |
| Maximum Word Duration | Longest articulation interval |
| Word Count | Total recognized words |
| Repeated Phrases | Repeated bigram sequences |

---

# Example Workflow

## Step 1
Upload audio file.

## Step 2
System resamples audio to:

```text
16000 Hz
```

---

## Step 3
Whisper transcribes speech.

---

## Step 4
Word timestamps are extracted.

---

## Step 5
Speech metrics are computed.

---

## Step 6
Threshold analysis is performed.

---

## Step 7
AI interpretation is generated.

---

# Supported Audio Formats

| Format | Supported |
|---|---|
| WAV | Yes |
| MP3 | Yes |
| FLAC | Yes |
| OGG | Yes |
| M4A | Yes |

---

# File Structure

```text
project/
│
├── app.py
├── README.md
└── .streamlit/
    └── secrets.toml
```

---

# Limitations

- Not a diagnostic system
- Depends on transcription quality
- Background noise may affect metrics
- Accent variability may affect timing extraction
- Clinical conclusions should not be inferred

---

# Future Improvements

Potential extensions:

- Spectral voice analysis
- Prosody analysis
- Pause duration analysis
- Emotional speech detection
- Phoneme-level timing
- Machine learning classification
- Longitudinal tracking
- Clinical dashboard export

---

# Research Domains

This project intersects:

- Computational Linguistics
- Speech Pathology
- Digital Signal Processing
- AI Speech Analytics
- Clinical Informatics
- Human-Computer Interaction

---

# Disclaimer

This platform is intended strictly for:
- education
- experimentation
- research
- computational speech analysis

It is not a medical device and does not diagnose autism or any neurological condition.

Clinical assessment should only be performed by licensed healthcare professionals.

---

# License

Open-source educational project.

---

# Author

Developed using:
- Python
- Streamlit
- Whisper AI
- OpenRouter
- NumPy
- Librosa
