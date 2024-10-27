from transformers import pipeline

# Load the sentiment-analysis pipeline from Hugging Face
sentiment_pipeline = pipeline("sentiment-analysis")

# Function to analyze polarity
def analyze_polarity(text):
    result = sentiment_pipeline(text)
    return result

# Example usage
example_text = "I went to the store to buy some groceries and then came back home."
polarity = analyze_polarity(example_text)
print("Polarity Detection Result:", polarity)
