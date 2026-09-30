import sqlite3
import logging
from contextlib import contextmanager
from ..config import Config

logger = logging.getLogger(__name__)

@contextmanager
def get_db_connection():
    conn = sqlite3.connect(Config.SQLITE_PATH)
    try:
        yield conn
    finally:
        conn.close()

def upsert_match(match_data: dict, dataset: str, conn: sqlite3.Connection = None):
    """
    Upsert a match into the matches table.
    """
    query = '''
    INSERT INTO matches (
        match_id, dataset, start_time, duration, radiant_win, 
        game_mode, lobby_type, patch_id, radiant_team_id, 
        dire_team_id, leagueid, parse_version
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ON CONFLICT(match_id) DO UPDATE SET
        dataset=excluded.dataset,
        start_time=excluded.start_time,
        duration=excluded.duration,
        radiant_win=excluded.radiant_win,
        game_mode=excluded.game_mode,
        lobby_type=excluded.lobby_type,
        patch_id=excluded.patch_id,
        radiant_team_id=excluded.radiant_team_id,
        dire_team_id=excluded.dire_team_id,
        leagueid=excluded.leagueid,
        parse_version=excluded.parse_version,
        ingested_at=CURRENT_TIMESTAMP
    '''
    
    params = (
        match_data.get('match_id'),
        dataset,
        match_data.get('start_time'),
        match_data.get('duration'),
        match_data.get('radiant_win'),
        match_data.get('game_mode'),
        match_data.get('lobby_type'),
        match_data.get('patch'),
        match_data.get('radiant_team_id'),
        match_data.get('dire_team_id'),
        match_data.get('leagueid'),
        match_data.get('version'),
    )
    
    if conn is not None:
        conn.execute(query, params)
    else:
        with get_db_connection() as c:
            c.execute(query, params)
            c.commit()

def insert_draft_events(match_id: int, events: list, conn: sqlite3.Connection = None):
    """
    Insert a list of draft events for a match.
    Ensures idempotency by deleting any existing events for this match first.
    """
    delete_query = 'DELETE FROM draft_events WHERE match_id = ?'
    query = '''
    INSERT INTO draft_events (match_id, "order", is_pick, team, hero_id)
    VALUES (?, ?, ?, ?, ?)
    '''
    
    data = []
    for ev in events:
        data.append((
            match_id,
            ev.get('order'),
            ev.get('is_pick'),
            ev.get('team'),
            ev.get('hero_id')
        ))
        
    if conn is not None:
        conn.execute(delete_query, (match_id,))
        conn.executemany(query, data)
    else:
        with get_db_connection() as c:
            c.execute(delete_query, (match_id,))
            c.executemany(query, data)
            c.commit()

def insert_players(match_id: int, players: list, conn: sqlite3.Connection = None):
    """
    Insert players metadata including rank_tier and computed_mmr.
    Ensures idempotency by deleting any existing players for this match first.
    """
    delete_query = 'DELETE FROM match_players WHERE match_id = ?'
    query = '''
    INSERT INTO match_players (match_id, player_slot, account_id, leaver_status, team_id, rank_tier, computed_mmr)
    VALUES (?, ?, ?, ?, ?, ?, ?)
    '''
    
    data = []
    for p in players:
        data.append((
            match_id,
            p.get('player_slot'),
            p.get('account_id'),
            p.get('leaver_status'),
            p.get('team_id'),
            p.get('rank_tier'),
            p.get('computed_mmr')
        ))
        
    if conn is not None:
        conn.execute(delete_query, (match_id,))
        conn.executemany(query, data)
    else:
        with get_db_connection() as c:
            c.execute(delete_query, (match_id,))
            c.executemany(query, data)
            c.commit()

def save_full_match(match_data: dict, dataset: str) -> bool:
    """
    Persists match, draft_events, and match_players in a single atomic transaction.
    If any insert fails, the transaction is rolled back completely.
    """
    match_id = match_data.get('match_id')
    if not match_id:
        return False

    with get_db_connection() as conn:
        try:
            with conn: # Begins SQLite transaction, commits on exit, rollbacks on error
                upsert_match(match_data, dataset, conn=conn)
                
                pbs = match_data.get('picks_bans')
                if pbs:
                    insert_draft_events(match_id, pbs, conn=conn)
                    
                players = match_data.get('players')
                if players:
                    insert_players(match_id, players, conn=conn)
                    
            return True
        except sqlite3.Error as e:
            logger.error(f"Failed to atomically persist match {match_id}: {e}")
            return False

def get_ingestion_state(dataset: str) -> int:
    """Gets the last ingested match_id for a dataset."""
    with get_db_connection() as conn:
        row = conn.execute(
            "SELECT last_match_id FROM ingestion_state WHERE dataset = ?", 
            (dataset,)
        ).fetchone()
        return row[0] if row else 0

def set_ingestion_state(dataset: str, last_match_id: int):
    """Updates ingestion checkpoint."""
    query = '''
    INSERT INTO ingestion_state (dataset, last_match_id, updated_at)
    VALUES (?, ?, CURRENT_TIMESTAMP)
    ON CONFLICT(dataset) DO UPDATE SET
        last_match_id=excluded.last_match_id,
        updated_at=CURRENT_TIMESTAMP
    '''
    with get_db_connection() as conn:
        conn.execute(query, (dataset, last_match_id))
        conn.commit()
