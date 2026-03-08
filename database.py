import sqlite3

def init_db():
    conn = sqlite3.connect('servidor_ia.db')
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS geracoes 
        (id INTEGER PRIMARY KEY AUTOINCREMENT, prompt TEXT, path TEXT, data DATETIME DEFAULT CURRENT_TIMESTAMP)''')
    conn.commit()
    conn.close()

if __name__ == "__main__":
    init_db()