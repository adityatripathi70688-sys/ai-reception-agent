import os
import psycopg2
import pandas as pd
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv

load_dotenv()

def get_connection():
    """Establishes connection to PostgreSQL database with fallback password."""
    return psycopg2.connect(
        host=os.getenv("DB_HOST", "localhost"),
        database=os.getenv("DB_NAME", "triage_db"),
        user=os.getenv("DB_USER", "postgres"),
        password=os.getenv("DB_PASS"),
        port=os.getenv("DB_PORT", "5432")
    )

def get_db_connection(db_path=None):
    """Alias for get_connection accepting optional arguments."""
    return get_connection()

def init_db(db_path=None):
    """Initializes the call_logs table in PostgreSQL."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS call_logs (
            id SERIAL PRIMARY KEY,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            caller_name VARCHAR(255),
            callback_number VARCHAR(50),
            intent TEXT,
            sentiment VARCHAR(50),
            urgency VARCHAR(20),
            risk_score INT,
            assigned_department VARCHAR(100),
            action_required TEXT
        );
    """)
    conn.commit()
    cur.close()
    conn.close()

def insert_call_log(data: dict, db_path=None):
    """Inserts an extracted AI triage result into PostgreSQL."""
    conn = get_connection()
    cur = conn.cursor()

    query = """
        INSERT INTO call_logs
        (caller_name, callback_number, intent, sentiment, urgency, risk_score, assigned_department, action_required)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s);
    """

    cur.execute(query, (
        data.get("caller_name", "Unknown"),
        data.get("callback_number", "Unknown"),
        data.get("intent", ""),
        data.get("sentiment", "Neutral"),
        data.get("urgency", "Medium"),
        data.get("risk_score", 5),
        data.get("assigned_department", "General"),
        data.get("action_required", "")
    ))

    conn.commit()
    cur.close()
    conn.close()

def fetch_all_logs(db_path=None):
    """Fetches all logs for Streamlit as a pandas DataFrame."""
    conn = get_connection()
    df = pd.read_sql_query("SELECT * FROM call_logs ORDER BY timestamp DESC;", conn)
    conn.close()
    return df
def fetch_filtered_logs(db_path=None, urgency="All", department="All", search_text=None, **kwargs):
    """Fetches logs with filtering by urgency, department, and search text."""
    conn = get_connection()

    query = "SELECT * FROM call_logs WHERE 1=1"
    params = []

    # Support keyword aliases passed by app.py
    urgency_val = kwargs.get("urgency_filter", urgency)
    dept_val = kwargs.get("dept_filter", department)

    if urgency_val and urgency_val != "All":
        query += " AND urgency ILIKE %s"
        params.append(urgency_val)

    if dept_val and dept_val != "All":
        query += " AND assigned_department ILIKE %s"
        params.append(dept_val)

    if search_text:
        query += """ AND (
            caller_name ILIKE %s OR
            intent ILIKE %s OR
            callback_number ILIKE %s
        )"""
        search_pattern = f"%{search_text}%"
        params.extend([search_pattern, search_pattern, search_pattern])

    query += " ORDER BY timestamp DESC;"

    df = pd.read_sql_query(query, conn, params=params)
    conn.close()
    return df


def get_all_logs(db_path=None):
    """Fallback alias for fetch_all_logs."""
    return fetch_all_logs(db_path)

def get_db_stats(db_path=None):
    """Calculates operational dashboard statistics, including breakdown dictionaries."""
    conn = get_connection()
    cur = conn.cursor(cursor_factory=RealDictCursor)

    cur.execute("SELECT COUNT(*) AS total FROM call_logs;")
    total = cur.fetchone()["total"]

    cur.execute("SELECT COUNT(*) AS high_urgency FROM call_logs WHERE urgency ILIKE 'High';")
    high_urgency = cur.fetchone()["high_urgency"]

    cur.execute("SELECT AVG(risk_score) AS avg_risk FROM call_logs;")
    avg_risk_res = cur.fetchone()["avg_risk"]
    avg_risk = round(float(avg_risk_res), 1) if avg_risk_res else 0.0

    cur.execute("SELECT urgency, COUNT(*) as count FROM call_logs GROUP BY urgency;")
    urgency_rows = cur.fetchall()
    urgency_breakdown = {row["urgency"]: row["count"] for row in urgency_rows}

    cur.execute("SELECT assigned_department, COUNT(*) as count FROM call_logs GROUP BY assigned_department;")
    dept_rows = cur.fetchall()
    dept_breakdown = {row["assigned_department"]: row["count"] for row in dept_rows}

    cur.close()
    conn.close()

    return {
        "total_calls": total,
        "total_tickets": total,
        "total_logs": total,
        "high_urgency": high_urgency,
        "urgent_calls": high_urgency,
        "avg_risk": avg_risk,
        "avg_risk_score": avg_risk,
        "average_risk": avg_risk,
        "urgency_breakdown": urgency_breakdown,
        "dept_breakdown": dept_breakdown,
        "department_breakdown": dept_breakdown
    }
