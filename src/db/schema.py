import sqlite3
import logging
from ..config import Config

def get_connection():
    return sqlite3.connect(Config.SQLITE_PATH)

def init_db():
    conn = get_connection()
    c = conn.cursor()
    
    # Matches table
    c.execute('''
    CREATE TABLE IF NOT EXISTS matches (
        match_id INTEGER PRIMARY KEY,
        dataset TEXT,
        start_time INTEGER,
        duration INTEGER,
        radiant_win BOOLEAN,
        game_mode INTEGER,
        lobby_type INTEGER,
        patch_id INTEGER,
        radiant_team_id INTEGER,
        dire_team_id INTEGER,
        leagueid INTEGER,
        parse_version INTEGER,
        ingested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    
    # Draft events (picks/bans)
    c.execute('''
    CREATE TABLE IF NOT EXISTS draft_events (
        match_id INTEGER,
        "order" INTEGER,
        is_pick BOOLEAN,
        team INTEGER,
        hero_id INTEGER,
        FOREIGN KEY(match_id) REFERENCES matches(match_id)
    )
    ''')
    
    # Match players (optional for MVP but good for future)
    c.execute('''
    CREATE TABLE IF NOT EXISTS match_players (
        match_id INTEGER,
        player_slot INTEGER,
        account_id INTEGER,
        leaver_status INTEGER,
        team_id INTEGER,
        rank_tier INTEGER,
        computed_mmr REAL,
        FOREIGN KEY(match_id) REFERENCES matches(match_id)
    )
    ''')
    
    # Run migrations if columns don't exist in existing table
    c.execute("PRAGMA table_info(match_players)")
    existing_cols = {row[1] for row in c.fetchall()}
    if "rank_tier" not in existing_cols:
        c.execute("ALTER TABLE match_players ADD COLUMN rank_tier INTEGER")
    if "computed_mmr" not in existing_cols:
        c.execute("ALTER TABLE match_players ADD COLUMN computed_mmr REAL")

    
    # Ingestion state (checkpoints)
    c.execute('''
    CREATE TABLE IF NOT EXISTS ingestion_state (
        dataset TEXT PRIMARY KEY,
        last_match_id INTEGER,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    
    # Hero reference (optional)
    c.execute('''
    CREATE TABLE IF NOT EXISTS hero_ref (
        hero_id INTEGER PRIMARY KEY,
        name TEXT,
        localized_name TEXT
    )
    ''')
    
    # Patch reference (optional)
    c.execute('''
    CREATE TABLE IF NOT EXISTS patch_ref (
        patch_id INTEGER PRIMARY KEY,
        name TEXT,
        date TIMESTAMP
    )
    ''')
    
    conn.commit()
    conn.close()
    logging.info(f"Database initialized at {Config.SQLITE_PATH}")

if __name__ == "__main__":
    from ..logger import setup_logging
    setup_logging()
    init_db()
