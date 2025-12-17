"""
Flask Backend for Medical Report Translator
Save this as app.py
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import tempfile
from werkzeug.utils import secure_filename

# Import your existing functions
import spacy
from spacy_layout import spaCyLayout
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import wikipedia
import re

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend communication

# Configure upload settings
UPLOAD_FOLDER = tempfile.gettempdir()
ALLOWED_EXTENSIONS = {'pdf'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Initialize models (do this once at startup)
print("Loading models...")

# PDF extraction
nlp_pdf = spacy.load("en_core_web_sm")
layout = spaCyLayout(nlp_pdf)

# Medical simplification
device = "cuda" if torch.cuda.is_available() else "cpu"
simplifier_model_name = "jkhan447/t5-medical-text-simplification"
try:
    simplifier_tokenizer = AutoTokenizer.from_pretrained(simplifier_model_name)
    simplifier_model = AutoModelForSeq2SeqLM.from_pretrained(simplifier_model_name).to(device)
except:
    simplifier_tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    simplifier_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base").to(device)

# Medical entity recognition
nlp_medical = spacy.load("en_ner_bc5cdr_md")

print("All models loaded!")


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def extract_text_from_pdf(pdf_path):
    """Extract text from PDF"""
    doc = layout(pdf_path)
    return doc.text


def simplify_medical_text(text):
    """Simplify medical text using T5 model"""
    input_text = "simplify: " + text
    inputs = simplifier_tokenizer(
        input_text, 
        return_tensors="pt", 
        max_length=512, 
        truncation=True
    ).to(device)
    
    outputs = simplifier_model.generate(
        inputs.input_ids, 
        max_length=200, 
        num_beams=4, 
        early_stopping=True
    )
    
    return simplifier_tokenizer.decode(outputs[0], skip_special_tokens=True)


def process_long_text(text, chunk_size=500):
    """Split and process long text"""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        if len(current_chunk) + len(sentence) < chunk_size:
            current_chunk += sentence + " "
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence + " "
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    simplified_chunks = []
    for chunk in chunks:
        simplified_chunks.append(simplify_medical_text(chunk))
    
    return "\n\n".join(simplified_chunks)


def get_wikipedia_definition(term):
    """Get Wikipedia definition for medical term"""
    try:
        definition = wikipedia.summary(term, sentences=1, auto_suggest=True)
        return definition
    except wikipedia.exceptions.DisambiguationError as e:
        try:
            return wikipedia.summary(e.options[0], sentences=1)
        except:
            return None
    except:
        return None


def extract_medical_terms(text):
    """Extract medical terms and their definitions"""
    doc = nlp_medical(text)
    definitions = {}
    
    unique_entities = list(set([ent.text for ent in doc.ents]))
    
    for term in unique_entities:
        if len(term) < 3:
            continue
        
        defn = get_wikipedia_definition(term)
        if defn:
            definitions[term] = defn
    
    return definitions


@app.route('/')
def index():
    return "Medical Report Translator API is running!"


@app.route('/api/upload', methods=['POST'])
def upload_file():
    """Handle PDF upload and processing"""
    
    # Check if file was uploaded
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Only PDF files are allowed'}), 400
    
    try:
        # Save file temporarily
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Extract text from PDF
        print(f"Extracting text from {filename}...")
        original_text = extract_text_from_pdf(filepath)
        
        # Simplify text
        print("Simplifying text...")
        simplified_text = process_long_text(original_text)
        
        # Extract medical terms
        print("Extracting medical terms...")
        glossary = extract_medical_terms(original_text)
        
        # Format glossary for frontend
        glossary_list = [
            {"term": term, "definition": defn}
            for term, defn in glossary.items()
        ]
        
        # Clean up temporary file
        os.remove(filepath)
        
        # Return results
        return jsonify({
            'success': True,
            'original': original_text,
            'simplified': simplified_text,
            'glossary': glossary_list
        })
        
    except Exception as e:
        # Clean up on error
        if os.path.exists(filepath):
            os.remove(filepath)
        
        print(f"Error processing file: {str(e)}")
        return jsonify({'error': f'Error processing file: {str(e)}'}), 500


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'models_loaded': True,
        'device': device
    })


if __name__ == '__main__':
    # Run the Flask app
    # For production, use a proper WSGI server like Gunicorn
    app.run(debug=True, use_reloader=False, host='0.0.0.0', port=5000)
