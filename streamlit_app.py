import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

@st.cache_resource
# Function to load the saved model
def load_model():
  tokenizer = AutoTokenizer.from_pretrained("Jayveersinh-Raj/PolyGuard")
  model = AutoModelForSequenceClassification.from_pretrained("Jayveersinh-Raj/PolyGuard")
  return tokenizer, model

# Function to classify the text as toxic or not
def classify_toxicity(text, tokenizer, model):
    inputs = tokenizer.encode(text, return_tensors="pt", max_length=512, truncation=True)
    outputs = model(inputs)[0]
    probabilities = torch.softmax(outputs, dim=1)
    predicted_class = torch.argmax(probabilities).item()

    if predicted_class == 1:
        return "Toxic"
    else:
        return "Not toxic"
      
# loading model, tokenizer
tokenizer, model = load_model()

# Streamlit app
st.title("Toxicity Analysis / Abuse detection")
input_text = st.text_area("Enter a comment:", "")

if st.button("Detect"):
    result = classify_toxicity(input_text, tokenizer, model)
    st.success(f"Detection: {result}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    st.write(f"Running on {device}")
