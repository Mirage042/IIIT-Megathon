# AI-Powered Mental Health and Neurological Analysis Dashboard

## Final Repository Submission

The final-final repository serves as the definitive version for submission. Please disregard any previous repositories, as they were utilized for testing and contain outdated code.

The ML_script branch contains all the machine learning scripts.

![image](https://github.com/user-attachments/assets/d959e0cd-5271-486d-b2f5-0f3c9bf0f223)


An advanced healthcare analysis platform that leverages state-of-the-art AI models to analyze mental health patterns and neurological conditions. The system combines sophisticated natural language processing, computer vision, and machine learning techniques to provide comprehensive mental health insights and early detection of neurological disorders.

## Core Features

### 1. Mental Health Analysis
- Multi-modal input processing (text and images)
- Contextual emotional state assessment
- Real-time sentiment analysis
- Predictive mental health monitoring

### 2. Neurological Disease Detection
- MRI Image Analysis for Alzheimer's Detection
  - Advanced CNN architecture for brain scan analysis
  - Region-specific anomaly detection
  - Early-stage Alzheimer's identification
  - Progression tracking and monitoring
  - Automated reporting of neurological findings

### 3. Advanced Sentiment Analysis
- BERT and Gemini model integration
- Custom mental health lexicon
- Multi-layered polarity detection
- Sentiment shift tracking

### 4. Intelligent Concern Classification
- Named Entity Recognition (NER) for mental health terminology
- Dynamic concern categorization
- Multi-dimensional intensity scoring
- Adaptive classification system

### 5. Predictive Analytics
- Proactive sentiment monitoring
- Pattern recognition
- Early warning system for emotional escalation
- Temporal sentiment shift analysis

## Backend Components

### __init__.py
Core system file managing:
- AI model initialization and integration
- Sentiment analysis pipeline
- NER and classification logic
- MRI image processing pipeline
- API endpoints and routing
- Real-time processing of user inputs
- Predictive analytics implementation

### ML.py
- Processes images and calculates various metrics
- Currently not being used in the main code

### chatbot.html
- Contains the html of the assisstant chatbot icon
  
## Getting Started

### Prerequisites
- Python 3.8+
- pip (Python package installer)
- GPU support required for MRI image analysis and model inference
- CUDA toolkit for deep learning models

### Installation

1. Clone the repository:
```bash
git clone [repository-url]
cd IIIT-Megathon
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

### Running the Application

1. Navigate to the project root directory
2. Run the application:
```bash
python run.py
```

## AI Model Integration

### Medical Imaging Analysis
- Convolutional Neural Networks for MRI processing
- Region-specific attention mechanisms
- Automated lesion detection
- Volumetric analysis of brain regions
- Progressive monitoring capabilities

### Sentiment Analysis Models
- BERT-based sentiment analysis
- Custom-trained mental health lexicon
- Multi-dimensional polarity detection

### Classification System
- Dynamic concern categorization
- Intensity scoring algorithm
- Temporal pattern recognition

### Predictive Analytics
- Sentiment shift prediction
- Early warning system
- Pattern-based recommendations

## Visualization Features

### Interactive Dashboards
- MRI scan visualization and analysis
- Brain region highlighting
- Sentiment timeline tracking
- Concern intensity heatmaps
- Polarity distribution charts
- Temporal shift analysis

### Real-time Analytics
- Live sentiment monitoring
- Dynamic concern reclassification
- Adaptive intensity scoring
- Progression tracking for neurological conditions

## Model Training and Updates

### Neurological Analysis Models
- Training data requirements for MRI analysis
- Model fine-tuning procedures
- Validation protocols
- Performance metrics
