from flask import Flask, request, Response, stream_with_context, jsonify, send_file
from flask_cors import CORS
import requests
import os
import base64
from diffusers import DiffusionPipeline
import torch

app = Flask(__name__)
CORS(app)

IMAGE_DIR = 'saved_images'
os.makedirs(IMAGE_DIR, exist_ok=True)

pipe = DiffusionPipeline.from_pretrained(
    "SimianLuo/LCM_Dreamshaper_v7",
    custom_pipeline="latent_consistency_txt2img",
    custom_revision="main",
    revision="fb9c5d"
)
pipe.to(torch_device="cuda", torch_dtype=torch.float32)

def sanitize_filename(filename):
    return ''.join(c if c.isalnum() or c in ['-', '_', '.'] else '_' for c in filename)

@app.route('/v1/generate-image', methods=['POST'])
def generate_image():
    try:
        data = request.json
        prompt = data.get('prompt')
        if not prompt:
            return jsonify({"error": "Prompt is required"}), 400
        
        num_inference_steps = data.get('num_inference_steps', 4)
        guidance_scale = data.get('guidance_scale', 8.0)
        images = pipe(prompt=prompt, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale, output_type="pil").images
        image = images[0]
        safe_prompt = sanitize_filename(prompt)
        image_path = os.path.join(IMAGE_DIR, f"{safe_prompt}.png")
        image.save(image_path)
        image_url = """/v1/get-image/"""+safe_prompt
        
        return jsonify({"message": "Image generated and saved successfully", "image_path": image_path, "image_url": image_url}), 200
    
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/v1/get-image/<string:prompt>', methods=['GET'])
def get_image(prompt):
    try:
        safe_prompt = sanitize_filename(prompt)
        image_path = os.path.join(IMAGE_DIR, f"{safe_prompt}.png")
        
        if not os.path.exists(image_path):
            return jsonify({"error": "Image not found"}), 404
        
        return send_file(image_path, mimetype='image/png')
    
    except Exception as e:
        return jsonify({"error": str(e)}), 400

def get_image_buffer_from_url(image_url):
    response = requests.get(image_url)
    if response.status_code == 200:
        return base64.b64encode(response.content).decode('utf-8')
    else:
        raise Exception(f"Failed to retrieve image from URL: {response.status_code}")

@app.route('/v1/vision', methods=['POST'])
def generate_image_info():
    try:
        data = request.json
        image_url = data['image_url']
        image_buffer = get_image_buffer_from_url(image_url)
        
        chat_messages = data.get('chat_history', [])
        prompt = data.get('prompt', "What is in this picture?")
        
        payload = {
            "model": "llava-llama3:8b-v1.1-fp16",
            "prompt": prompt,
            "stream": True,
            "images": [image_buffer],
            "chat_history": chat_messages
        }
        
        response = requests.post('http://127.0.0.1:11434/api/generate', json=payload, stream=True)
        
        def generate():
            for chunk in response.iter_content(chunk_size=8192):
                yield chunk
        
        headers_list = [(key, value) for key, value in response.headers.items()]
        
        return Response(stream_with_context(generate()), status=response.status_code, headers=headers_list, content_type=response.headers.get('content-type'))
    
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/', defaults={'path': ''}, methods=['GET', 'POST', 'OPTIONS'])
@app.route('/<path:path>', methods=['GET', 'POST', 'OPTIONS'])
def proxy(path):
    remote_host = 'http://127.0.0.1:11434'
    url = remote_host + '/' + path
    headers = {
        'Content-Type': 'text/plain'
    }

    if request.method == 'OPTIONS':
        return '', 200, headers

    if request.method == 'POST':
        response = requests.post(url, data=request.get_data(), headers=dict(request.headers), stream=True)
    else:
        response = requests.get(url, headers=dict(request.headers), stream=True)
        
    if response.status_code == 400:
        print("Error:", response.text)  # Print the error message

    def generate():
        for chunk in response.iter_content(chunk_size=8192):
            yield chunk

    # Convert headers to list of tuples
    headers_list = [(key, value) for key, value in response.headers.items()]

    return Response(stream_with_context(generate()), status=response.status_code, headers=headers_list, content_type=response.headers.get('content-type'))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=11199)
