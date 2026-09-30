import os
import unittest
import logging
from src.config import Config
from src.db.schema import init_db
from src.db.repo import (
    upsert_match,
    insert_draft_events,
    insert_players,
    get_ingestion_state,
    set_ingestion_state,
    get_db_connection
)

class TestDBSmoke(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.orig_db_path = Config.SQLITE_PATH
        cls.test_db_path = "data/test_smoke.sqlite"
        Config.SQLITE_PATH = cls.test_db_path
        if os.path.exists(cls.test_db_path):
            os.remove(cls.test_db_path)
        init_db()

    @classmethod
    def tearDownClass(cls):
        Config.SQLITE_PATH = cls.orig_db_path
        if os.path.exists(cls.test_db_path):
            try:
                os.remove(cls.test_db_path)
            except OSError:
                pass

    def test_crud_and_idempotency(self):
        match_id = 99999
        dataset = 'smoke_test'

        # 1. Upsert Match
        match_data = {
            'match_id': match_id,
            'start_time': 1600000000,
            'duration': 2000,
            'radiant_win': True,
            'game_mode': 2,
            'lobby_type': 1,
            'patch': 50,
            'radiant_team_id': 10,
            'dire_team_id': 20,
            'leagueid': 500,
            'version': 1
        }
        upsert_match(match_data, dataset)

        # 2. Insert Draft Events
        draft = [
            {'order': 1, 'is_pick': False, 'team': 0, 'hero_id': 100},
            {'order': 2, 'is_pick': True, 'team': 1, 'hero_id': 20}
        ]
        insert_draft_events(match_id, draft)

        # 3. Insert Players with MMR & Rank Tier
        players = [
            {
                'player_slot': 0,
                'account_id': 123456,
                'leaver_status': 0,
                'team_id': 10,
                'rank_tier': 72,
                'computed_mmr': 4350.5
            },
            {
                'player_slot': 128,
                'account_id': 654321,
                'leaver_status': 0,
                'team_id': 20,
                'rank_tier': 80,
                'computed_mmr': 4800.0
            }
        ]
        insert_players(match_id, players)

        # 4. Checkpoint
        set_ingestion_state(dataset, match_id)

        # 5. Verify Ingestion
        with get_db_connection() as conn:
            m = conn.execute("SELECT * FROM matches WHERE match_id=?", (match_id,)).fetchone()
            self.assertIsNotNone(m, "Match not found")
            self.assertEqual(m[1], dataset)

            d_count = conn.execute("SELECT count(*) FROM draft_events WHERE match_id=?", (match_id,)).fetchone()[0]
            self.assertEqual(d_count, 2)

            p_rows = conn.execute("SELECT account_id, rank_tier, computed_mmr FROM match_players WHERE match_id=? ORDER BY player_slot", (match_id,)).fetchall()
            self.assertEqual(len(p_rows), 2)
            self.assertEqual(p_rows[0][1], 72)
            self.assertEqual(p_rows[0][2], 4350.5)
            self.assertEqual(p_rows[1][1], 80)
            self.assertEqual(p_rows[1][2], 4800.0)

        self.assertEqual(get_ingestion_state(dataset), match_id)

        # 6. Verify Idempotency (re-insert should not duplicate rows)
        insert_draft_events(match_id, draft)
        insert_players(match_id, players)
        with get_db_connection() as conn:
            d_count_after = conn.execute("SELECT count(*) FROM draft_events WHERE match_id=?", (match_id,)).fetchone()[0]
            self.assertEqual(d_count_after, 2, "Draft events duplicated on re-insert")
            p_count_after = conn.execute("SELECT count(*) FROM match_players WHERE match_id=?", (match_id,)).fetchone()[0]
            self.assertEqual(p_count_after, 2, "Players duplicated on re-insert")

if __name__ == '__main__':
    unittest.main()
