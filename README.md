# my-first-repo


from transformers import pipeline

# Load pre-trained model for text classification
classifier = pipeline('zero-shot-classification', model='facebook/bart-large-mnli')

# Function to classify text as human or AI-written
def detect_ai_or_human(text):
    labels = ['AI-generated', 'Human-written']
    result = classifier(text, candidate_labels=labels)
    label = result['labels'][0]
    return label

# Example usage
if __name__ == "__main__":
    text_input = input("Enter the text to analyze: ")
    prediction = detect_ai_or_human(text_input)
    print(f"The text is likely: {prediction}")
