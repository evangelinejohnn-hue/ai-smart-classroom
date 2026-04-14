import sqlite3

def init():
    conn = sqlite3.connect("data.db")
    c = conn.cursor()

    c.execute("""
    CREATE TABLE IF NOT EXISTS records (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        classroom TEXT,
        user TEXT,
        emotion TEXT,
        focus REAL,
        stress REAL,
        score REAL,
        status TEXT
    )
    """)

    conn.commit()
    conn.close()


def insert(classroom, user, emotion, focus, stress, score, status):
    conn = sqlite3.connect("data.db")
    c = conn.cursor()

    c.execute("""
    INSERT INTO records (classroom, user, emotion, focus, stress, score, status)
    VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (classroom, user, emotion, focus, stress, score, status))

    conn.commit()
    conn.close()


def fetch_all():
    conn = sqlite3.connect("data.db")
    c = conn.cursor()

    c.execute("SELECT * FROM records")
    data = c.fetchall()

    conn.close()
    return data