# db.py
import sqlite3
from pathlib import Path

DB_PATH = Path("data/email_assistant.db")

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS accounts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        email_address TEXT NOT NULL,
        domain TEXT NOT NULL,
        imap_server TEXT NOT NULL,
        imap_port INTEGER NOT NULL DEFAULT 993,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        last_sync TIMESTAMP,
        active INTEGER DEFAULT 1
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS emails (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        account_id INTEGER NOT NULL,
        email_uid TEXT NOT NULL,
        sender_name TEXT,
        sender_email TEXT,
        subject TEXT,
        body TEXT,
        date TIMESTAMP,
        has_attachments INTEGER DEFAULT 0,
        raw_path TEXT,
        FOREIGN KEY (account_id) REFERENCES accounts (id)
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS attachments (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        email_id INTEGER NOT NULL,
        filename TEXT,
        file_path TEXT,
        FOREIGN KEY (email_id) REFERENCES emails (id)
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS embeddings (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        email_id INTEGER NOT NULL,
        chunk_id TEXT,
        vector_store_id TEXT,
        FOREIGN KEY (email_id) REFERENCES emails (id)
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS chats (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        account_id INTEGER,
        user_message TEXT NOT NULL,
        assistant_response TEXT NOT NULL,
        sources TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (account_id) REFERENCES accounts (id)
    )
    """)

    conn.commit()
    conn.close()
    print("✅ Database initialized at", DB_PATH)

def get_connection():
    return sqlite3.connect(DB_PATH)
