import unittest
from src.features.filters import is_valid_match

class TestFilters(unittest.TestCase):
    def test_valid_match(self):
        match = {
            'match_id': 1,
            'game_mode': 2,
            'duration': 1801,
            'radiant_win': True,
            'picks_bans': [
                {'is_pick': True, 'team': 0, 'hero_id': i} for i in range(1, 6)
            ] + [
                {'is_pick': True, 'team': 1, 'hero_id': i} for i in range(6, 11)
            ],
            'players': [
                {'player_slot': i, 'leaver_status': 0} for i in range(5)
            ] + [
                {'player_slot': 128 + i, 'leaver_status': 0} for i in range(5)
            ]
        }
        self.assertTrue(is_valid_match(match))

    def test_invalid_game_mode(self):
        match = {
            'match_id': 1,
            'game_mode': 1, # All Pick
            'duration': 2000,
            'radiant_win': True,
            'picks_bans': [{'is_pick': True} for _ in range(10)],
            'players': [{'leaver_status': 0} for _ in range(10)]
        }
        self.assertFalse(is_valid_match(match))

    def test_short_duration(self):
        match = {
            'match_id': 1,
            'game_mode': 2,
            'duration': 1799,
            'radiant_win': True,
            'picks_bans': [{'is_pick': True} for _ in range(10)],
            'players': [{'leaver_status': 0} for _ in range(10)]
        }
        self.assertFalse(is_valid_match(match))

    def test_incomplete_draft(self):
        match = {
            'match_id': 1,
            'game_mode': 2,
            'duration': 2000,
            'radiant_win': True,
            'picks_bans': [{'is_pick': True} for _ in range(9)], # 9 picks
            'players': [{'leaver_status': 0} for _ in range(10)]
        }
        self.assertFalse(is_valid_match(match))

    def test_unbalanced_picks(self):
        match = {
            'match_id': 1,
            'game_mode': 2,
            'duration': 2000,
            'radiant_win': True,
            'picks_bans': [
                {'is_pick': True, 'team': 0, 'hero_id': i} for i in range(1, 7) # 6 radiant
            ] + [
                {'is_pick': True, 'team': 1, 'hero_id': i} for i in range(7, 11) # 4 dire
            ],
            'players': [{'player_slot': i, 'leaver_status': 0} for i in range(10)]
        }
        self.assertFalse(is_valid_match(match))

    def test_duplicate_hero_picks(self):
        match = {
            'match_id': 1,
            'game_mode': 2,
            'duration': 2000,
            'radiant_win': True,
            'picks_bans': [
                {'is_pick': True, 'team': 0, 'hero_id': 1} for _ in range(5) # All hero 1
            ] + [
                {'is_pick': True, 'team': 1, 'hero_id': i} for i in range(2, 7)
            ],
            'players': [{'player_slot': i, 'leaver_status': 0} for i in range(10)]
        }
        self.assertFalse(is_valid_match(match))

    def test_leaver(self):
        match = {
            'match_id': 1,
            'game_mode': 2,
            'duration': 2000,
            'radiant_win': True,
            'picks_bans': [{'is_pick': True} for _ in range(10)],
            'players': [
                {'player_slot': i, 'leaver_status': 0} for i in range(9)
            ] + [
                {'player_slot': 9, 'leaver_status': 1} # Leaver
            ]
        }
        self.assertFalse(is_valid_match(match))

if __name__ == '__main__':
    unittest.main()
