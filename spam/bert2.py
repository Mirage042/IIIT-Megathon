from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch.nn.functional as F

# Load a model that can detect positive, neutral, and negative sentiments
model_name = "cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Set up the pipeline with the three-class model
sentiment_pipeline = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

# Mapping for labels to descriptive names
label_mapping = {
    "LABEL_0": "Negative",
    "LABEL_1": "Neutral",
    "LABEL_2": "Positive"
}

# Function to analyze polarity with detailed scores
def analyze_polarity(text):
    # Tokenize input and get raw logits from the model
    inputs = tokenizer(text, return_tensors="pt")
    outputs = model(**inputs).logits
    
    # Apply softmax to get probabilities for each sentiment
    probabilities = F.softmax(outputs, dim=1).squeeze().tolist()
    
    # Map each probability to its sentiment label
    sentiment_scores = {
        label_mapping[f'LABEL_{i}']: round(prob, 4) for i, prob in enumerate(probabilities)
    }
    return sentiment_scores

# Example usage
example_text = "I am thrilled with the progress I’ve made; everything is going perfectly as planned!"
polarity = analyze_polarity(example_text)
print("Polarity Detection Result:", polarity)
