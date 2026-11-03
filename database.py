"""
Database module for AI-Based Plant Disease Detection System
Uses SQLite to persist leaf diagnosis history, system analytics, and user inquiries.
"""

import os
import sqlite3
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'plant_doctor.db')


def get_db_connection():
    """Create a database connection with dict-like row access."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Initialize database tables and seed baseline sample records if empty."""
    conn = get_db_connection()
    cursor = conn.cursor()

    # Detections table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS detections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            image_path TEXT NOT NULL,
            plant_name TEXT NOT NULL,
            disease_name TEXT NOT NULL,
            confidence REAL NOT NULL,
            is_healthy INTEGER NOT NULL DEFAULT 0,
            severity TEXT NOT NULL DEFAULT 'Moderate',
            model_mode TEXT NOT NULL DEFAULT 'Trained CNN Model',
            symptoms TEXT,
            causes TEXT,
            treatment TEXT,
            prevention TEXT,
            detected_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Contact messages table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS contact_messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT NOT NULL,
            role TEXT DEFAULT 'Farmer',
            subject TEXT NOT NULL,
            message TEXT NOT NULL,
            submitted_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)

    conn.commit()

    # Seed initial diagnostic records if empty to ensure rich dashboard visualization immediately
    cursor.execute("SELECT COUNT(*) AS count FROM detections")
    if cursor.fetchone()['count'] == 0:
        seed_data = [
            ("sample_tomato_early_blight.jpg", "/static/images/sample_leaves/tomato_early_blight.jpg",
             "Tomato", "Tomato Early Blight", 95.8, 0, "High", "Trained CNN Model",
             "Dark, concentric rings (target-like spots) on lower, older leaves with yellow chlorotic halos.",
             "Alternaria solani fungus thrives in warm, humid weather with frequent rainfall or heavy dew.",
             "Apply copper-based fungicides or chlorothalonil. Remove lower diseased foliage and ensure adequate air circulation.",
             "Practice 3-year crop rotation, avoid overhead irrigation, mulch soil, and prune lower leaves."),
            
            ("sample_tomato_healthy.jpg", "/static/images/sample_leaves/tomato_healthy.jpg",
             "Tomato", "Healthy Tomato Leaf", 99.4, 1, "Healthy", "Trained CNN Model",
             "Leaves are vibrant green with uniform texture, no discoloration, spots, or lesions.",
             "Optimal nutrient balance, adequate sunlight, and pathogen-free environment.",
             "None needed. Maintain current balanced fertilization and consistent drip irrigation.",
             "Continue regular inspections, maintain pest control, and keep balanced organic soil nutrition."),
            
            ("sample_potato_late_blight.jpg", "/static/images/sample_leaves/potato_late_blight.jpg",
             "Potato", "Potato Late Blight", 97.2, 0, "Severe", "Trained CNN Model",
             "Water-soaked, dark brown to purplish-black lesions starting from leaf tips and margins with white fuzzy mold underneath in humid conditions.",
             "Phytophthora infestans oomycete pathogen, spreading rapidly in cool, wet environments (15-20°C).",
             "Immediately apply targeted systemic fungicides (mancozeb, cymoxanil, metalaxyl). Destroy severely infected plants.",
             "Plant certified disease-free seed tubers, ensure good canopy aeration, and destroy volunteer potato cull piles."),
            
            ("sample_apple_scab.jpg", "/static/images/sample_leaves/apple_scab.jpg",
             "Apple", "Apple Scab", 93.6, 0, "Moderate", "Trained CNN Model",
             "Dull olive-green to velvety brown lesions on leaf surfaces, causing leaf distortion and premature leaf drop.",
             "Venturia inaequalis ascomycete fungus overwintering in fallen leaves.",
             "Apply captan, sulfur, or myclobutanil fungicides during budding and petal fall stages.",
             "Rake and destroy fallen leaves in autumn, prune canopy for airflow, and select scab-resistant cultivars (e.g., Liberty, Freedom)."),
            
            ("sample_corn_rust.jpg", "/static/images/sample_leaves/corn_rust.jpg",
             "Corn", "Corn Common Rust", 96.1, 0, "Moderate", "Trained CNN Model",
             "Small, circular to elongated cinnamon-brown powdery pustules on both upper and lower leaf surfaces.",
             "Puccinia sorghi fungal spores carried over long distances by wind currents.",
             "Apply triazole or strobilurin fungicides if rust pustules appear before silking stage.",
             "Plant rust-resistant hybrid varieties and plant early in the season to minimize exposure to late-arriving spores."),

            ("sample_grape_black_rot.jpg", "/static/images/sample_leaves/grape_black_rot.jpg",
             "Grape", "Grape Black Rot", 94.7, 0, "High", "Trained CNN Model",
             "Small yellowish spots on leaves that turn reddish-brown with dark borders and tiny black fruiting bodies (pycnidia).",
             "Guignardia bidwellii fungus thriving in warm, damp spring weather.",
             "Spray copper or mancozeb fungicide from early spring shoot growth until veraison.",
             "Prune out mummified berry clusters, maintain open vine trellis canopy, and remove diseased canes during winter.")
        ]

        cursor.executemany("""
            INSERT INTO detections (
                filename, image_path, plant_name, disease_name, confidence,
                is_healthy, severity, model_mode, symptoms, causes, treatment, prevention
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, seed_data)
        conn.commit()

    conn.close()


def save_detection(data):
    """Save a new leaf diagnosis result to the database."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO detections (
            filename, image_path, plant_name, disease_name, confidence,
            is_healthy, severity, model_mode, symptoms, causes, treatment, prevention
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        data['filename'],
        data['image_path'],
        data['plant_name'],
        data['disease_name'],
        data['confidence'],
        1 if data.get('is_healthy') else 0,
        data.get('severity', 'Moderate'),
        data.get('model_mode', 'Trained CNN Model'),
        data.get('symptoms', ''),
        data.get('causes', ''),
        data.get('treatment', ''),
        data.get('prevention', '')
    ))
    conn.commit()
    new_id = cursor.lastrowid
    conn.close()
    return new_id


def get_recent_detections(limit=15):
    """Retrieve the most recent detection events."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT * FROM detections
        ORDER BY detected_at DESC, id DESC
        LIMIT ?
    """, (limit,))
    rows = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return rows


def get_detection_by_id(detection_id):
    """Retrieve full details of a specific detection."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM detections WHERE id = ?", (detection_id,))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None


def get_dashboard_stats():
    """Compute aggregate diagnostic metrics and distribution stats for the dashboard."""
    conn = get_db_connection()
    cursor = conn.cursor()

    # Total counts
    cursor.execute("SELECT COUNT(*) AS total FROM detections")
    total = cursor.fetchone()['total']

    cursor.execute("SELECT COUNT(*) AS healthy FROM detections WHERE is_healthy = 1")
    healthy = cursor.fetchone()['healthy']

    diseased = total - healthy

    # Average confidence
    cursor.execute("SELECT AVG(confidence) AS avg_conf FROM detections")
    avg_conf_row = cursor.fetchone()
    avg_confidence = round(avg_conf_row['avg_conf'] if avg_conf_row['avg_conf'] else 0.0, 1)

    # Disease frequency distribution
    cursor.execute("""
        SELECT disease_name, COUNT(*) AS count
        FROM detections
        GROUP BY disease_name
        ORDER BY count DESC
        LIMIT 6
    """)
    disease_dist = [dict(row) for row in cursor.fetchall()]

    # Plant distribution
    cursor.execute("""
        SELECT plant_name, COUNT(*) AS count
        FROM detections
        GROUP BY plant_name
        ORDER BY count DESC
    """)
    plant_dist = [dict(row) for row in cursor.fetchall()]

    # Recent 7 activity timeline points
    cursor.execute("""
        SELECT SUBSTR(detected_at, 1, 10) as date_val, COUNT(*) as count
        FROM detections
        GROUP BY date_val
        ORDER BY date_val DESC
        LIMIT 7
    """)
    timeline_rows = [dict(row) for row in cursor.fetchall()]
    timeline_rows.reverse()

    conn.close()

    return {
        'total_analyzed': total,
        'healthy_count': healthy,
        'diseased_count': diseased,
        'healthy_percentage': round((healthy / total * 100), 1) if total > 0 else 0,
        'diseased_percentage': round((diseased / total * 100), 1) if total > 0 else 0,
        'avg_confidence': avg_confidence,
        'disease_distribution': disease_dist,
        'plant_distribution': plant_dist,
        'timeline': timeline_rows
    }


def save_contact_message(data):
    """Save user inquiry or feedback from the contact form."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO contact_messages (name, email, role, subject, message)
        VALUES (?, ?, ?, ?, ?)
    """, (
        data.get('name'),
        data.get('email'),
        data.get('role', 'Farmer'),
        data.get('subject'),
        data.get('message')
    ))
    conn.commit()
    new_id = cursor.lastrowid
    conn.close()
    return new_id


def clear_history():
    """Reset detection records while keeping the table structure."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM detections")
    conn.commit()
    conn.close()
    return True
