"""
PayirBot 2.0 - Database Layer
Handles all SQLite storage for inspection records.

Used by dashboard.py - not meant to be run directly.
"""

import sqlite3
import os
from datetime import datetime

from config import PROJECT_ROOT

DB_PATH = os.path.join(PROJECT_ROOT, "dashboard_data", "payirbot.db")
IMAGES_DIR = os.path.join(PROJECT_ROOT, "dashboard_data", "captured_images")


def init_db():
    """Create the database and inspections table if they don't exist yet."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    os.makedirs(IMAGES_DIR, exist_ok=True)

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS inspections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            plant_number INTEGER,
            disease TEXT,
            confidence REAL,
            image_path TEXT,
            timestamp TEXT
        )
    """)
    conn.commit()
    conn.close()


def get_next_plant_number():
    """Returns the next sequential plant number for a new inspection."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT MAX(plant_number) FROM inspections")
    result = cursor.fetchone()[0]
    conn.close()
    return (result or 0) + 1


def save_inspection(plant_number, disease, confidence, image_path):
    """Insert a new inspection record into the database."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO inspections (plant_number, disease, confidence, image_path, timestamp)
        VALUES (?, ?, ?, ?, ?)
    """, (
        plant_number,
        disease,
        confidence,
        image_path,
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    ))
    conn.commit()
    conn.close()


def get_all_inspections():
    """Returns all inspection records, most recent first."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT plant_number, disease, confidence, image_path, timestamp
        FROM inspections
        ORDER BY id DESC
    """)
    rows = cursor.fetchall()
    conn.close()
    return rows


def get_summary_stats():
    """Returns (total_count, healthy_count, diseased_count) for the dashboard header."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM inspections")
    total = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM inspections WHERE disease LIKE '%healthy%'")
    healthy = cursor.fetchone()[0]
    conn.close()
    diseased = total - healthy
    return total, healthy, diseased