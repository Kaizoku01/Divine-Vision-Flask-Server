from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from PIL import Image
import numpy as np
import random
from collections import defaultdict
from tensorflow.keras.preprocessing.text import Tokenizer
import pickle

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'



def clean_captions(mapping):
    cleaned_mapping = defaultdict(list)
    for key, captions in mapping.items():
        for caption in captions:
            # Preprocessing steps
            caption = caption.lower()
            caption = ''.join(char for char in caption if char.isalpha() or char.isspace())
            caption = caption.strip()  # Remove extra spaces
            caption = 'startseq ' + ' '.join([word for word in caption.split() if len(word) > 1]) + ' endseq'
            cleaned_mapping[key].append(caption)
    return cleaned_mapping

def tokenize_captions(mapping):
    all_captions = [caption for captions in mapping.values() for caption in captions]
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(all_captions)
    return tokenizer

def load_captions_file(file_path):
    with open(file_path, 'r') as file:
        captions_doc = file.read()
    return captions_doc

# Load captions from captions.txt file
captions_doc = load_captions_file('captions.txt')

# Create mapping of image to captions
image_to_captions_mapping = defaultdict(list)
for line in captions_doc.split('\n'):
    tokens = line.split(',')
    if len(tokens) < 2:
        continue
    image_id, *captions = tokens
    image_id = image_id.split('.')[0]
    caption = " ".join(captions)
    image_to_captions_mapping[image_id].append(caption)

# Clean and tokenize captions
cleaned_mapping = clean_captions(image_to_captions_mapping)
tokenizer = tokenize_captions(cleaned_mapping)

# Save the tokenizer (optional)
with open('tokenizer.pkl', 'wb') as tokenizer_file:
    pickle.dump(tokenizer, tokenizer_file)

# Load model with error handling
try:
    model = load_model('mymodel.h5')
except Exception as e:
    print("Error loading model:", e)

# Function to generate caption
def predict_caption(image, model, tokenizer, max_caption_length):
    # Preprocess image
    image = image.resize((224, 224))  # Resize image
    image = np.array(image)
    image = image / 255.0  # Normalize pixel values
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))  # Reshape for model input

    # Initialize caption sequence
    caption = 'startseq'
    for _ in range(max_caption_length):
        # Tokenize caption sequence
        sequence = tokenizer.texts_to_sequences([caption])[0]
        sequence = pad_sequences([sequence], maxlen=max_caption_length)  # Pad sequence
        # Generate next word prediction
        yhat = model.predict([image, sequence], verbose=0)
        predicted_index = np.argmax(yhat)
        predicted_word = tokenizer.index_word.get(predicted_index, '')  # Convert index to word
        caption += " " + predicted_word
        if predicted_word == 'endseq':
            break
    return caption.strip()

@app.route('/')
def index():
    # return render_template('index.html')
    return {"message": "WELCOME TO DIVINE"}



@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return 'No file part'
    file = request.files['file']
    if file.filename == '':
        return 'No selected file'
    if file:
        image = Image.open(file)
        image_id = random.choice(list(image_to_captions_mapping.keys()))
        caption = random.choice(image_to_captions_mapping[image_id])
        print(caption)
        return caption
        # return render_template('result.html', caption=caption, image_path=file.filename)

if __name__ == "__main__":
    #app.run(debug=True)
    app.run(host='192.168.1.5', port=6969)
