from flask import Flask, request, jsonify, render_template
from PIL import Image
import google.generativeai as genai

app = Flask(__name__)


api_key = "AIzaSyBVec6inSuyB7rE4qOPdskMM_UTWQ525GM"

# Configure Google Generative AI
genai.configure(api_key=api_key)

# Model configuration
generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config=generation_config,
)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def analyze():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
    
    image = request.files['image']
    prompt = request.form.get('prompt')

    if not prompt:
        return jsonify({"error": "Prompt is missing"}), 400
    
    if image.filename == '':
        return jsonify({"error": "No selected image"}), 400
    
    # Save the uploaded image to a temporary file
    image_path = "/tmp"
    image.save(image_path)

    # Upload the image to Gemini
    uploaded_file = upload_to_gemini(image_path, mime_type="image/jpeg")
    
    # Start a chat session with the model and send the prompt along with the uploaded image
    chat_session = model.start_chat(
        history=[
            {
                "role": "user",
                "parts": [uploaded_file, prompt],
            }
        ]
    )

    response = chat_session.send_message(prompt)

    if response:
        result = response.text
    else:
        result = "Error occurred while analyzing the image"

    return jsonify({"result": result})

def upload_to_gemini(path, mime_type=None):
    """Uploads the given file to Gemini."""
    file = genai.upload_file(path, mime_type=mime_type)
    print(f"Uploaded file '{file.display_name}' as: {file.uri}")
    return file

if __name__ == "__main__":
    app.run(debug=True, port=5656)