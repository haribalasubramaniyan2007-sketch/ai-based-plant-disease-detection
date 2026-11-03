"""
Flask Web Application for AI-Based Plant Disease Detection System.
Provides RESTful APIs for leaf image upload, OpenCV preprocessing,
CNN model inference, SQLite history tracking, dashboard metrics, and disease library.
"""

import os
import uuid
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename

import database
import model_service

# Initialize Flask
app = Flask(__name__, static_folder='static', template_folder='templates')
app.config['SECRET_KEY'] = 'agri-plant-ai-disease-detector-2026'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max upload

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static', 'uploads')
SAMPLE_DIR = os.path.join(BASE_DIR, 'static', 'images', 'sample_leaves')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(SAMPLE_DIR, exist_ok=True)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.before_request
def initialize_app_context():
    """Ensure database is initialized before handling requests."""
    pass


# -------------------------------------------------------------------------
# Web Routes
# -------------------------------------------------------------------------

@app.route('/')
def index():
    """Render the main single page web application."""
    model_status = model_service.get_model_status()
    return render_template('index.html', model_status=model_status)


# -------------------------------------------------------------------------
# API Endpoints
# -------------------------------------------------------------------------

@app.route('/api/status', methods=['GET'])
def get_system_status():
    """Return backend, AI engine, and database operational status."""
    status = model_service.get_model_status()
    stats = database.get_dashboard_stats()
    status['total_records'] = stats['total_analyzed']
    return jsonify({
        "status": "success",
        "data": status
    })


@app.route('/api/detect', methods=['POST'])
def detect_disease():
    """
    Handle plant leaf image submission (file upload or sample leaf selection).
    Runs OpenCV preprocessing and AI prediction, saves to SQLite, and returns diagnosis.
    """
    try:
        sample_name = request.form.get('sample_name')
        image_path = None
        saved_filename = None
        web_image_url = None

        if sample_name:
            # User selected a bundled sample leaf
            sample_file = secure_filename(sample_name)
            candidate_path = os.path.join(SAMPLE_DIR, sample_file)
            if not os.path.exists(candidate_path):
                return jsonify({"status": "error", "message": f"Sample leaf '{sample_name}' not found."}), 404

            # Make a copy in uploads folder for processing
            unique_prefix = f"sample_{uuid.uuid4().hex[:8]}_"
            saved_filename = unique_prefix + sample_file
            image_path = os.path.join(UPLOAD_FOLDER, saved_filename)
            
            with open(candidate_path, 'rb') as src, open(image_path, 'wb') as dst:
                dst.write(src.read())

            web_image_url = f"/static/uploads/{saved_filename}"

        elif 'image' in request.files:
            file = request.files['image']
            if file.filename == '':
                return jsonify({"status": "error", "message": "No file selected."}), 400

            if not allowed_file(file.filename):
                return jsonify({
                    "status": "error",
                    "message": "Invalid file format. Supported formats: JPG, JPEG, PNG, WEBP."
                }), 400

            ext = file.filename.rsplit('.', 1)[1].lower()
            saved_filename = f"leaf_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}.{ext}"
            image_path = os.path.join(UPLOAD_FOLDER, saved_filename)
            file.save(image_path)
            web_image_url = f"/static/uploads/{saved_filename}"

        else:
            return jsonify({"status": "error", "message": "Please upload an image file or choose a sample leaf."}), 400

        # Run AI prediction and OpenCV diagnostic pipeline
        diagnosis = model_service.predict_disease(image_path, uploads_dir=UPLOAD_FOLDER)

        # Prepare payload for SQLite database
        db_record = {
            'filename': saved_filename,
            'image_path': web_image_url,
            'plant_name': diagnosis['plant_name'],
            'disease_name': diagnosis['disease_name'],
            'confidence': diagnosis['confidence'],
            'is_healthy': diagnosis['is_healthy'],
            'severity': diagnosis['severity'],
            'model_mode': diagnosis['model_mode'],
            'symptoms': diagnosis['symptoms'],
            'causes': diagnosis['causes'],
            'treatment': diagnosis['treatment'],
            'prevention': diagnosis['prevention']
        }

        # Save to SQLite
        record_id = database.save_detection(db_record)
        diagnosis['record_id'] = record_id
        diagnosis['image_url'] = web_image_url

        return jsonify({
            "status": "success",
            "message": "Leaf image successfully analyzed.",
            "data": diagnosis
        })

    except Exception as e:
        print(f"[Detect Error]: {e}")
        return jsonify({"status": "error", "message": f"Analysis failed: {str(e)}"}), 500


@app.route('/api/dashboard/stats', methods=['GET'])
def get_dashboard_metrics():
    """Return aggregated diagnostic statistics for Chart.js charts."""
    try:
        stats = database.get_dashboard_stats()
        recent = database.get_recent_detections(limit=8)
        return jsonify({
            "status": "success",
            "data": {
                "stats": stats,
                "recent_detections": recent
            }
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/api/history', methods=['GET'])
def get_history():
    """Retrieve list of detection records."""
    limit = request.args.get('limit', default=25, type=int)
    history = database.get_recent_detections(limit=limit)
    return jsonify({
        "status": "success",
        "data": history
    })


@app.route('/api/history/<int:record_id>', methods=['GET'])
def get_detection_details(record_id):
    """Retrieve full details for a single diagnosis."""
    record = database.get_detection_by_id(record_id)
    if not record:
        return jsonify({"status": "error", "message": "Record not found."}), 404
    return jsonify({
        "status": "success",
        "data": record
    })


@app.route('/api/history/clear', methods=['POST'])
def clear_detection_history():
    """Clear past detection history."""
    try:
        database.clear_history()
        return jsonify({"status": "success", "message": "Detection history cleared successfully."})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/api/library', methods=['GET'])
def get_disease_library():
    """Retrieve complete catalog of plant diseases and recommendations."""
    plant_filter = request.args.get('plant', default=None)
    items = []
    for disease_name, info in model_service.DISEASE_KNOWLEDGE_BASE.items():
        if plant_filter and info['plant'].lower() != plant_filter.lower():
            continue
        item = dict(info)
        item['name'] = disease_name
        items.append(item)
    return jsonify({
        "status": "success",
        "total": len(items),
        "data": items
    })


@app.route('/api/sample-images', methods=['GET'])
def get_sample_images():
    """Return available sample test leaf images for quick 1-click evaluation."""
    samples = [
        {
            "id": "tomato_early_blight",
            "filename": "tomato_early_blight.jpg",
            "label": "Tomato Early Blight",
            "plant": "Tomato",
            "condition": "Fungal Lesions",
            "thumb": "/static/images/sample_leaves/tomato_early_blight.jpg"
        },
        {
            "id": "tomato_healthy",
            "filename": "tomato_healthy.jpg",
            "label": "Healthy Tomato Leaf",
            "plant": "Tomato",
            "condition": "Optimal Health",
            "thumb": "/static/images/sample_leaves/tomato_healthy.jpg"
        },
        {
            "id": "potato_late_blight",
            "filename": "potato_late_blight.jpg",
            "label": "Potato Late Blight",
            "plant": "Potato",
            "condition": "Severe Blight",
            "thumb": "/static/images/sample_leaves/potato_late_blight.jpg"
        },
        {
            "id": "apple_scab",
            "filename": "apple_scab.jpg",
            "label": "Apple Scab",
            "plant": "Apple",
            "condition": "Velvety Scab",
            "thumb": "/static/images/sample_leaves/apple_scab.jpg"
        },
        {
            "id": "corn_rust",
            "filename": "corn_rust.jpg",
            "label": "Corn Common Rust",
            "plant": "Corn",
            "condition": "Rust Pustules",
            "thumb": "/static/images/sample_leaves/corn_rust.jpg"
        },
        {
            "id": "grape_black_rot",
            "filename": "grape_black_rot.jpg",
            "label": "Grape Black Rot",
            "plant": "Grape",
            "condition": "Necrotic Spots",
            "thumb": "/static/images/sample_leaves/grape_black_rot.jpg"
        }
    ]
    return jsonify({
        "status": "success",
        "data": samples
    })


@app.route('/api/contact', methods=['POST'])
def submit_contact():
    """Handle contact form submissions and persist to SQLite."""
    try:
        data = request.get_json() or request.form
        name = data.get('name', '').strip()
        email = data.get('email', '').strip()
        role = data.get('role', 'Farmer').strip()
        subject = data.get('subject', '').strip()
        message = data.get('message', '').strip()

        if not name or not email or not message:
            return jsonify({
                "status": "error",
                "message": "Please provide your name, email, and message."
            }), 400

        record_id = database.save_contact_message({
            'name': name,
            'email': email,
            'role': role,
            'subject': subject or 'General Inquiry',
            'message': message
        })

        return jsonify({
            "status": "success",
            "message": "Thank you for reaching out! Our agricultural team has received your message.",
            "message_id": record_id
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


if __name__ == '__main__':
    # Initialize SQLite database
    database.init_db()
    
    # Try loading AI model
    model_service.load_ai_model()
    
    print("\n" + "="*60)
    print(" AI-Based Plant Disease Detection System Server ")
    print(" System running on http://127.0.0.1:5000")
    print("="*60 + "\n")
    app.run(host='0.0.0.0', port=5000, debug=False)
