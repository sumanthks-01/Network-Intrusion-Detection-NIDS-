import sqlite3
import json
from datetime import datetime
from loguru import logger

DATABASE_PATH = "detections.db"

def init_database():
    """Initialize SQLite database"""
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS detections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            prediction TEXT,
            score REAL,
            meta TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    conn.commit()
    conn.close()
    logger.info("Database initialized")

def save_detection_sqlite(data: dict):
    """Save detection to SQLite database"""
    try:
        init_database()
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO detections (prediction, score, meta)
            VALUES (?, ?, ?)
        ''', (
            str(data.get('prediction', '')),
            float(data.get('score', 0.0)),
            json.dumps(data.get('meta', {}))
        ))
        
        conn.commit()
        conn.close()
        logger.info("Detection saved to SQLite")
        return True
    except Exception as e:
        logger.error(f"Failed to save to SQLite: {e}")
        return False

def get_detections_sqlite(limit=100):
    """Get detections from SQLite database"""
    try:
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM detections 
            ORDER BY timestamp DESC 
            LIMIT ?
        ''', (limit,))
        
        rows = cursor.fetchall()
        detections = []
        
        for row in rows:
            detections.append({
                "id": row[0],
                "prediction": row[1],
                "score": row[2],
                "meta": json.loads(row[3]) if row[3] else {},
                "timestamp": row[4]
            })
        
        conn.close()
        return detections
    except Exception as e:
        logger.error(f"Failed to get detections: {e}")
        return []