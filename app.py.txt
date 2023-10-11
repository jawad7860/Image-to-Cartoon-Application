from flask import Flask, request, jsonify, render_template, redirect, url_for
from itsdangerous import URLSafeTimedSerializer as Serializer, BadSignature, SignatureExpired
import os
from werkzeug.utils import secure_filename 
from PIL import Image
import argparse
import os
import mimetypes
from utils.transforms import get_no_aug_transform
import torch
from models.generator import Generator
import numpy as np
import torchvision.transforms.functional as TF
import torch.nn.functional as F
from torchvision import transforms
import cv2
from torchvision import utils as vutils
import subprocess
import tempfile
import re
from tqdm import tqdm
import shutil
import requests
from zipfile import ZipFile
from io import BytesIO



def Download_Model():

  # Define the folder name to check and the GitHub repository URL
  folder_to_check = "checkpoints"
  github_repo_url = "https://github.com/jawad7860/Image-to-Cartoon-Application/archive/master.zip"

  # Check if the folder exists in the current directory
  if not os.path.exists(folder_to_check):
      try:
          # Download the GitHub repository as a ZIP file
          response = requests.get(github_repo_url)
          
          if response.status_code == 200:
              # Extract the ZIP file in memory
              with ZipFile(BytesIO(response.content), "r") as zip_ref:
                  # Extract the specific folder you want (e.g., checkpoints)
                  zip_ref.extractall(".")
                  # Rename the extracted folder to the desired name
                  os.rename("Image-to-Cartoon-Application-master/checkpoints", folder_to_check)
                  # Remove the temporary directory created during extraction
                  shutil.rmtree("Image-to-Cartoon-Application-master")
              print(f"Downloaded and extracted the {folder_to_check} folder.")
          else:
              print("Failed to download the GitHub repository ZIP file.")
      except Exception as e:
          print(f"Error downloading and extracting the GitHub repository: {str(e)}")


def inv_normalize(img):
    # Adding 0.1 to all normalization values since the model is trained (erroneously) without correct de-normalization
    mean = torch.Tensor([0.485, 0.456, 0.406]).to(device)
    std = torch.Tensor([0.229, 0.224, 0.225]).to(device)

    img = img * std.view(1, 3, 1, 1) + mean.view(1, 3, 1, 1)
    img = img.clamp(0, 1)
    return img

def predict_images(image_list):
    trf = get_no_aug_transform()
    image_list = torch.from_numpy(np.array([trf(img).numpy() for img in image_list])).to(device)

    with torch.no_grad():
        generated_images = netG(image_list)
    generated_images = inv_normalize(generated_images)

    pil_images = []
    for i in range(generated_images.size()[0]):
        generated_image = generated_images[i].cpu()
        pil_images.append(TF.to_pil_image(generated_image))
    return pil_images

def listdir_fullpath(d):
    return [os.path.join(d, f) for f in os.listdir(d)]

def divide_chunks(l, n): 
    # looping till length l 
    for i in range(0, len(l), n):  
        yield l[i:i + n]


# Generate a token for a user
def generate_token(username):
    token = serializer.dumps({'username': username})
    return token

# Verify a token
def verify_token(token):
    try:
        data = serializer.loads(token, max_age=3600)  # Token expires in 1 hour
        return data  # Returns the data inside the token (e.g., username)
    except (BadSignature, SignatureExpired):
        return None  # Token is invalid or expired


app = Flask(__name__)

# Secret key for token generation (keep this secret)
SECRET_KEY = os.urandom(24)

# User database (for demonstration purposes)
users_db = {
    'user1': {'password': 'password1'},
    'user2': {'password': 'password2'}
}

# Create a token serializer
serializer = Serializer(SECRET_KEY)

#Download model if not available
Download_Model()

#Loading Model
batch_size = 4
user_stated_device='cpu'
device = torch.device(user_stated_device)
pretrained_dir = "./checkpoints/trained_netG.pth"
netG = Generator().to(device)
netG.eval()

# Load weights
if user_stated_device == "cuda":
    netG.load_state_dict(torch.load(pretrained_dir))
else:
    netG.load_state_dict(torch.load(pretrained_dir, map_location=torch.device('cpu')))


# Landing page with registration and login options
@app.route('/', methods=['GET'])
def landing_page():
    return render_template('landing.html')

# User registration form (GET request)
@app.route('/register', methods=['GET'])
def registration_form():
    return render_template('register.html')

# User registration (POST request)
@app.route('/register', methods=['POST'])
def register():
    data = request.form  # Use request.form to access form data
    username = data.get('username')
    password = data.get('password')

    if not username or not password:
        return jsonify({'message': 'Username and password are required'}), 400

    if username in users_db:
        return jsonify({'message': 'Username already exists'}), 400

    users_db[username] = {'password': password}
    return jsonify({'message': 'Registration successful'}), 201


# User login form (GET request)
@app.route('/login', methods=['GET'])
def login_form():
    return render_template('login.html')

# User login and token issuance (POST request)
@app.route('/login', methods=['POST'])
def login():
    data = request.form
    username = data.get('username')
    password = data.get('password')

    if not username or not password:
        return jsonify({'message': 'Username and password are required'}), 400

    user = users_db.get(username)
    if not user or user['password'] != password:
        return jsonify({'message': 'Invalid credentials'}), 401

    token = generate_token(username)
    return redirect(url_for('protected_resource', token=token))

# Protected route that requires a valid token
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Check if the uploaded file has an allowed extension
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Protected route that requires a valid token
@app.route('/protected', methods=['GET', 'POST'])
def protected_resource():
    token = request.args.get('token')

    if not token:
        return jsonify({'message': 'Token is missing'}), 401

    data = verify_token(token)
    if not data:
        return jsonify({'message': 'Token is invalid or expired'}), 401

    img_data = None

    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)

        file = request.files['file']

        if file.filename == '':
            return redirect(request.url)

        if file and allowed_file(file.filename):
            image = Image.open(file).convert('RGB')
            predicted_image = predict_images([image])[0]

            predicted_image_byte_array = io.BytesIO()
            predicted_image.save(predicted_image_byte_array, format='PNG')
            img_data = base64.b64encode(predicted_image_byte_array.getvalue()).decode('utf-8')

    return render_template('index.html', img_data=img_data)

if __name__ == '__main__':
    app.run(debug=True)




















