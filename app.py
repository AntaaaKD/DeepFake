import contextlib
import os
import re
import sqlite3
from datetime import timedelta
import numpy as np
from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError
from flask import Flask, render_template, request, session, redirect, url_for
from torchvision import models
from werkzeug.utils import secure_filename

from create_database import setup_database
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
from utils import login_required, set_session
from PIL import Image
from keras.models import load_model

import numpy as np
import time
from glob import glob
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Supprime les warnings INFO
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
app = Flask(__name__)

# Charger le modèle TensorFlow/Keras entraîné
adv_model = load_model("model.h5")
# Après adv_model = load_model("model.h5")
adv_model.compile(optimizer='adam',
                 loss='categorical_crossentropy',
                 metrics=['accuracy'])




# Charger le modèle Keras
deepfake_model = load_model("deepfake.h5")
deepfake_model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Charger le modèle de défense une seule fois
defense_model = load_model('defense_model.h5')
CLASSES_DEFENSE = ['Fake', 'Real']  # adapte selon tes classes
# Après defense_model = load_model('defense_model.h5')
defense_model.compile(optimizer='adam',
                     loss='binary_crossentropy',
                     metrics=['accuracy'])





from werkzeug.utils import secure_filename
import imghdr  # Pour vérification du type d'image

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def validate_image(stream):
    header = stream.read(512)
    stream.seek(0)
    format = imghdr.what(None, header)
    if not format:
        return None
    return '.' + (format if format != 'jpeg' else 'jpg')



def generate_adversarial_example(image, model, epsilon=0.06, original_size=(224, 224)):


    image_tensor = tf.convert_to_tensor(image, dtype=tf.float32)

    with tf.GradientTape() as tape:
        tape.watch(image_tensor)
        prediction = model(image_tensor)
        loss = tf.keras.losses.categorical_crossentropy(prediction, prediction)

    gradient = tape.gradient(loss, image_tensor)
    perturbation = epsilon * tf.sign(gradient)
    adversarial_image = image_tensor + perturbation
    adversarial_image = tf.clip_by_value(adversarial_image, 0, 1)

    # Convertir en image PIL
    adversarial_image_np = (adversarial_image.numpy()[0] * 255).astype('uint8')
    adversarial_image_pil = Image.fromarray(adversarial_image_np)

    # 🔥 Ajuster la taille à celle de l'image originale
    adversarial_image_pil = adversarial_image_pil.resize(original_size, Image.LANCZOS)

    return adversarial_image_pil


# Config Flask
app.config['SECRET_KEY'] = 'EXAMPLE_xpSm7p5bgJY8rNoBjGWiz5yjxMNlW6231IBI62OkLc='
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(days=15)

setup_database(name='users.db')

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER






@app.route('/')
@login_required
def index():
    return render_template('index.html', username=session.get('username'))

@app.route('/logout')
def logout():
    session.clear()
    return redirect('/login')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'GET':
        return render_template('login.html')

    username = request.form.get('username')
    password = request.form.get('password')

    query = 'SELECT username, password FROM users WHERE username = :username;'
    with contextlib.closing(sqlite3.connect('users.db')) as conn:
        with conn:
            account = conn.execute(query, {'username': username}).fetchone()

    if not account:
        return render_template('login.html', error='Username does not exist')

    try:
        ph = PasswordHasher()
        ph.verify(account[1], password)
    except VerifyMismatchError:
        return render_template('login.html', error='Incorrect password')

    if ph.check_needs_rehash(account[1]):
        update_query = 'UPDATE users SET password = :password WHERE username = :username;'
        params = {'password': ph.hash(password), 'username': username}
        with contextlib.closing(sqlite3.connect('users.db')) as conn:
            with conn:
                conn.execute(update_query, params)

    set_session(username=username, remember_me='remember-me' in request.form)
    return redirect('/')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'GET':
        return render_template('register.html')

    username = request.form.get('username')
    password = request.form.get('password')
    confirm_password = request.form.get('confirm-password')
    email = request.form.get('email')

    if len(password) < 8:
        return render_template('register.html', error='Your password must be 8 or more characters')
    if password != confirm_password:
        return render_template('register.html', error='Passwords do not match')
    if not re.match(r'^[a-zA-Z0-9]+$', username):
        return render_template('register.html', error='Username must only contain letters and numbers')
    if not 3 < len(username) < 26:
        return render_template('register.html', error='Username must be between 4 and 25 characters')

    query = 'SELECT username FROM users WHERE username = :username;'
    with contextlib.closing(sqlite3.connect('users.db')) as conn:
        with conn:
            if conn.execute(query, {'username': username}).fetchone():
                return render_template('register.html', error='Username already exists')

    ph = PasswordHasher()
    hashed_password = ph.hash(password)

    insert_query = 'INSERT INTO users(username, password, email) VALUES (:username, :password, :email);'
    params = {'username': username, 'password': hashed_password, 'email': email}

    with contextlib.closing(sqlite3.connect('users.db')) as conn:
        with conn:
            conn.execute(insert_query, params)

    set_session(username=username)
    return redirect('/')

# Route pour la détection Deepfake
@app.route('/deepfake', methods=['GET', 'POST'])
@login_required
def deepfake_detection():
    if request.method == 'POST':
        file = request.files.get('file')
        if not file:
            return render_template('deepfake.html', error="Veuillez sélectionner une image.")

        # Générer un nom de fichier unique
        unique_filename = str(uuid.uuid4()) + "_" + file.filename
        filepath = os.path.join("static/uploads", unique_filename)
        file.save(filepath)

        # Prétraitement de l'image
        img = Image.open(filepath).resize((224, 224))
        img_array = np.array(img)
        img_array = np.expand_dims(img_array, axis=0)

        # Prédiction avec le modèle Keras
        prediction = deepfake_model.predict(img_array)
        predicted_class = np.argmax(prediction, axis=1)[0]
        confidence = float(prediction[0][predicted_class]) * 100

        # Définition des classes
        labels = ["Fake", "Real"]
        predicted_label = labels[predicted_class]

        # Chemin relatif pour Flask
        file_rel_path = os.path.relpath(filepath, "static").replace("\\", "/")

        return redirect(url_for("deepfake_result", label=predicted_label, confidence=confidence, file_path=file_rel_path, source='deepfake'))

    return render_template("deepfake.html")


@app.route('/deepfake/result')
@login_required
def deepfake_result():
    """Affiche le résultat de la détection"""
    label = request.args.get('label', "Résultat non disponible")
    confidence = request.args.get('confidence', 0)
    file_path = request.args.get('file_path', "")
    source = request.args.get('source', 'deepfake')

    return render_template("result.html", label=label, confidence=confidence, file_path=file_path, source=source)
def preprocess_image(image_file):
    """Prépare l'image pour la passer au modèle"""
    try:
        img = Image.open(image_file)
        img = img.resize((224, 224))  # Redimensionner
        img = np.array(img) / 255.0   # Normaliser entre 0 et 1
        img = np.expand_dims(img, axis=0)  # Ajouter la dimension batch (1, 224, 224, 3)
        return img.astype(np.float32)  # Convertir en float32 pour TensorFlow
    except Exception as e:
        print(f"Erreur de traitement d'image: {e}")
        return None


import uuid  # Pour générer des noms de fichiers uniques

@app.route('/adversarial', methods=['GET', 'POST'])
def adversarial():
    if request.method == 'POST':
        file = request.files.get('file')
        if not file:
            return render_template('adversarial.html', error="Veuillez sélectionner une image.")

        # Générer un nom de fichier unique
        unique_filename = str(uuid.uuid4()) + "_" + file.filename
        original_path = os.path.join("static/adversarial_images", unique_filename)
        file.save(original_path)

        # Charger l'image pour obtenir sa taille originale
        original_image = Image.open(file)
        original_size = original_image.size  # 🔹 Stocker la taille d'origine

        # Charger et prétraiter l'image
        processed_image = preprocess_image(file)
        if processed_image is None:
            return render_template('adversarial.html', error="Erreur lors du traitement de l'image.")

        # Générer l'image adversariale et la redimensionner à la taille d'origine
        adversarial_image = generate_adversarial_example(processed_image, adv_model, original_size=original_size)

        # Sauvegarder l'image adversariale
        adversarial_filename = f"adv_{unique_filename}"
        adversarial_path = os.path.join("static/adversarial_images", adversarial_filename)
        adversarial_image.save(adversarial_path)

        # Chemins relatifs pour Flask
        original_rel_path = os.path.relpath(original_path, "static").replace("\\", "/")
        adversarial_rel_path = os.path.relpath(adversarial_path, "static").replace("\\", "/")

        return redirect(url_for("result_adversarial", original=original_rel_path, adversarial=adversarial_rel_path))

    return render_template('adversarial.html')




@app.route('/result_adversarial')
def result_adversarial():
    """Affiche les résultats de la génération adversariale"""
    original = request.args.get('original', "")
    adversarial = request.args.get('adversarial', "")

    return render_template("result_adversarial.html",
                           original=original,
                           adversarial=adversarial)


@app.route('/adversarial_from_result')
@login_required
def adversarial_from_result():
    """Redirige vers la génération d'image adversariale en utilisant l'image déjà chargée."""
    file_path = request.args.get('file_path', "")

    if not file_path:
        return redirect(
            url_for('deepfake_detection'))  # Redirige vers la page de détection si aucun fichier n'est fourni

    # Chemin complet de l'image originale
    original_path = os.path.join("static", file_path)

    # Charger l'image pour obtenir sa taille originale
    original_image = Image.open(original_path)
    original_size = original_image.size  # Stocker la taille d'origine

    # Charger et prétraiter l'image
    processed_image = preprocess_image(original_path)
    if processed_image is None:
        return render_template('deepfake.html', error="Erreur lors du traitement de l'image.")

    # Générer l'image adversariale et la redimensionner à la taille d'origine
    adversarial_image = generate_adversarial_example(processed_image, adv_model, original_size=original_size)

    # Sauvegarder l'image adversariale
    adversarial_filename = f"adv_{os.path.basename(file_path)}"
    adversarial_path = os.path.join("static/adversarial_images", adversarial_filename)
    adversarial_image.save(adversarial_path)

    # Chemins relatifs pour Flask
    original_rel_path = os.path.relpath(original_path, "static").replace("\\", "/")
    adversarial_rel_path = os.path.relpath(adversarial_path, "static").replace("\\", "/")

    return redirect(url_for("result_adversarial", original=original_rel_path, adversarial=adversarial_rel_path))


@app.route('/detect_adversarial')
@login_required
def detect_adversarial():
    """Détecte si l'image adversaire est Fake ou Real."""
    file_path = request.args.get('file_path', "")

    if not file_path:
        return redirect(url_for('adversarial'))  # Redirige vers la page de génération si aucun fichier n'est fourni

    # Chemin complet de l'image adversaire
    adversarial_path = os.path.join("static", file_path)

    # Prétraitement de l'image
    img = Image.open(adversarial_path).resize((224, 224))
    img_array = np.array(img)  # Ne pas diviser par 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prédiction avec le modèle Keras
    prediction = deepfake_model.predict(img_array)
    print("Prédiction brute :", prediction)

    # Seuil de décision
    threshold = 0.5  # Vous pouvez ajuster cette valeur
    predicted_prob = prediction[0][0]  # Probabilité que l'image soit "Fake"

    if predicted_prob > threshold:
        predicted_label = "Fake"
    else:
        predicted_label = "Real"

    confidence = float(predicted_prob) * 100

    # Logs pour vérifier la prédiction
    print("Classe prédite :", predicted_label)
    print("Confiance :", confidence)

    # Chemin relatif pour Flask
    file_rel_path = os.path.relpath(adversarial_path, "static").replace("\\", "/")

    return redirect(url_for("deepfake_result", label=predicted_label, confidence=confidence, file_path=file_rel_path, source='adversarial'))





@app.route('/defense', methods=['GET', 'POST'])
@login_required
def defense():
    if request.method == 'POST':
        # Vérification du fichier
        if 'file' not in request.files:
            return render_template('defense.html', error="Aucun fichier sélectionné")

        file = request.files['file']
        if file.filename == '':
            return render_template('defense.html', error="Aucun fichier sélectionné")

        # Validation de l'extension
        filename = secure_filename(file.filename)
        if not allowed_file(filename):
            return render_template('defense.html', error="Format non supporté (seulement .jpg, .png)")

        # Sauvegarde originale
        unique_id = str(uuid.uuid4())
        original_filename = f"original_{unique_id}_{filename}"
        original_path = os.path.join("static", "images", original_filename)
        os.makedirs(os.path.dirname(original_path), exist_ok=True)
        file.save(original_path)

        # Vérification que l'image est valide
        try:
            with Image.open(original_path) as img:
                img.verify()  # Vérifie si l'image est valide
        except (IOError, SyntaxError) as e:
            os.remove(original_path)  # Supprime le fichier invalide
            return render_template('defense.html', error=f"Fichier image invalide: {str(e)}")

        # Prétraitement robuste
        try:
            img = Image.open(original_path).convert('RGB')  # Force le mode RGB
            img = img.resize((224, 224))
            img_array = np.array(img) / 255.0

            # Vérification des dimensions
            if img_array.shape != (224, 224, 3):
                raise ValueError("Dimensions incorrectes après traitement")

            img_array = np.expand_dims(img_array, axis=0)

            # Prédiction
            prediction = defense_model.predict(img_array)
            predicted_class = CLASSES_DEFENSE[np.argmax(prediction)]
            confidence = round(float(np.max(prediction)) * 100, 2)

            return render_template('defense_result.html',
                                   label=predicted_class,
                                   confidence=confidence,
                                   file_path=f"images/{original_filename}")

        except Exception as e:
            error_msg = f"Erreur de traitement: {str(e)}"
            print(error_msg)
            return render_template('defense.html', error=error_msg)

    return render_template('defense.html')


@app.route('/defense_from_result')
@login_required
def defense_from_result():
    file_path = request.args.get('file_path', "")
    if not file_path:
        return redirect(url_for('deepfake_detection'))

    full_path = os.path.join("static", file_path)
    processed_image = preprocess_image(full_path)
    if processed_image is None:
        return render_template('defense.html', error="Erreur lors du traitement de l'image.")

    prediction = defense_model.predict(processed_image)
    predicted_class = CLASSES_DEFENSE[np.argmax(prediction)]
    confidence = prediction[0][np.argmax(prediction)]
    confidence_pct = round(confidence * 100, 2)

    return render_template('defense_result.html',
                           label=predicted_class,
                           confidence=confidence_pct,
                           file_path=file_path)


def cleanup_old_files(directory="static/images", max_age_hours=24):
    now = time.time()
    for filepath in glob(os.path.join(directory, "*")):
        if os.stat(filepath).st_mtime < now - max_age_hours * 3600:
            os.remove(filepath)
            print(f"Supprimé : {filepath}")



if __name__ == '__main__':
    app.run(debug=True)