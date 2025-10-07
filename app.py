from flask import Flask, render_template, request, jsonify, session, redirect, url_for
import cv2
import mediapipe as mp
import numpy as np
import math
import base64
import sqlite3
import json
import os

app = Flask(__name__)
app.secret_key = 'supersecret'

# ===== Initialize MediaPipe FaceMesh =====
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True
)

# ===== Database Setup =====
DB_FILE = "users.db"

def init_db():
    if not os.path.exists(DB_FILE):
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        c.execute('''
            CREATE TABLE users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                embedding TEXT NOT NULL
            )
        ''')
        conn.commit()
        conn.close()

def get_db_connection():
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    return conn

init_db()

# ===== Helpers =====
def eye_aspect_ratio(landmarks, eye_indices):
    """Compute Eye Aspect Ratio (EAR) for blink detection"""
    p1, p2, p3, p4, p5, p6 = [landmarks[i] for i in eye_indices]
    vertical1 = math.dist(p2, p6)
    vertical2 = math.dist(p3, p5)
    horizontal = math.dist(p1, p4)
    return (vertical1 + vertical2) / (2.0 * horizontal)

def get_face_embedding(landmarks):
    """Generate normalized face embedding from landmarks"""
    arr = np.array(landmarks).flatten()
    arr = arr / np.linalg.norm(arr)
    return arr

def compare_embeddings(e1, e2, threshold=0.05):
    """Compare embeddings with a tolerance threshold"""
    return np.linalg.norm(e1 - e2) < threshold

def b64_to_cv2_img(b64_string):
    """Convert base64 string to OpenCV image"""
    header, encoded = b64_string.split(',', 1)
    data = base64.b64decode(encoded)
    nparr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img

# ===== Routes =====
@app.route('/')
def home():
    return render_template('index.html')

# --- Register new user ---
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        img_b64 = request.form.get('face_image')

        if not img_b64:
            return "⚠️ No face data captured. Please blink to record your face."

        # Convert image and process landmarks
        frame = b64_to_cv2_img(img_b64)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(frame_rgb)

        if not results.multi_face_landmarks:
            return "⚠️ No face detected. Try again."

        face_landmarks = results.multi_face_landmarks[0].landmark
        landmarks = [(lm.x, lm.y) for lm in face_landmarks]
        embedding = get_face_embedding(landmarks)

        conn = get_db_connection()
        c = conn.cursor()

        # Check if user exists
        c.execute("SELECT * FROM users WHERE username = ?", (username,))
        existing_user = c.fetchone()

        if existing_user:
            conn.close()
            return "⚠️ Username already exists. Try a different one."

        # Save embedding to SQLite
        c.execute("INSERT INTO users (username, embedding) VALUES (?, ?)",
                  (username, json.dumps(embedding.tolist())))
        conn.commit()
        conn.close()

        return f"✅ Registration successful! Face saved for {username}"

    return render_template('register.html')

# --- Login form ---
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']

        conn = get_db_connection()
        c = conn.cursor()
        c.execute("SELECT * FROM users WHERE username = ?", (username,))
        user = c.fetchone()
        conn.close()

        if not user:
            return "❌ User not found. Please register first."

        session['username'] = username
        return redirect(url_for('face_login'))
    return render_template('login.html')

# --- Blink + Face login verification ---
@app.route('/face_login')
def face_login():
    if 'username' not in session:
        return redirect(url_for('login'))
    return render_template('face_login.html', username=session['username'])

# --- Analyze captured frame for blink and embedding ---
@app.route('/analyze_frame', methods=['POST'])
def analyze_frame():
    data = request.json
    img_b64 = data.get('image')
    if not img_b64:
        return jsonify({"error": "No image"}), 400

    try:
        frame = b64_to_cv2_img(img_b64)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(frame_rgb)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    if not results.multi_face_landmarks:
        return jsonify({"face": False, "eye_closed": False, "ear": 0.0})

    face_landmarks = results.multi_face_landmarks[0].landmark
    landmarks = [(lm.x, lm.y) for lm in face_landmarks]

    left_eye_idx = [33, 160, 158, 133, 153, 144]
    right_eye_idx = [362, 385, 387, 263, 373, 380]

    left_ear = eye_aspect_ratio(landmarks, left_eye_idx)
    right_ear = eye_aspect_ratio(landmarks, right_eye_idx)
    ear = (left_ear + right_ear) / 2.0

    eye_closed = ear < 0.21
    username = session.get('username')

    authenticated = False
    if eye_closed and username:
        conn = get_db_connection()
        c = conn.cursor()
        c.execute("SELECT embedding FROM users WHERE username = ?", (username,))
        user = c.fetchone()
        conn.close()

        if user:
            stored_embedding = np.array(json.loads(user['embedding']))
            live_embedding = get_face_embedding(landmarks)
            if compare_embeddings(stored_embedding, live_embedding):
                authenticated = True
                session['authenticated'] = True

    return jsonify({
        "face": True,
        "eye_closed": eye_closed,
        "ear": float(ear),
        "authenticated": authenticated
    })

# --- Logout ---
@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('home'))

# ===== Main =====
if __name__ == '__main__':
    app.run(debug=True)
