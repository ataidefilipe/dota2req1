import unittest
import torch
import os
import shutil
import tempfile
from src.features.neural_features import NeuralFeatureEncoder
from src.model.neural_dataset import DotaNeuralDataset, collate_dota_batch

class TestNeuralFeatures(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.mock_matches = [
            {
                'match_id': 1001,
                'radiant_win': True,
                'patch_id': 59,
                'start_time': 1000,
                'rad_picks': [1, 2, 3, 4, 5],
                'dire_picks': [6, 7, 8, 9, 10],
                'rad_bans': [11, 12, 13],
                'dire_bans': [14, 15, 16],
                'rad_players': [{'rank_tier': 70, 'computed_mmr': 4000.0} for _ in range(5)],
                'dire_players': [{'rank_tier': 80, 'computed_mmr': 4500.0} for _ in range(5)]
            },
            {
                'match_id': 1002,
                'radiant_win': False,
                'patch_id': 59,
                'start_time': 2000,
                'rad_picks': [2, 3, 4, 5, 6],
                'dire_picks': [7, 8, 9, 10, 11],
                'rad_bans': [1, 12],
                'dire_bans': [13, 14],
                'rad_players': [{'rank_tier': None, 'computed_mmr': None} for _ in range(5)],
                'dire_players': [{'rank_tier': 75, 'computed_mmr': 4200.0} for _ in range(5)]
            }
        ]

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_encoder_fit_and_transform(self):
        encoder = NeuralFeatureEncoder()
        encoder.fit(self.mock_matches)

        self.assertTrue(encoder.fitted)
        self.assertIn(1, encoder.hero_to_idx)
        self.assertIn(59, encoder.patch_to_idx)
        self.assertIn(70, encoder.rank_to_idx)

        # Transform match
        encoded = encoder.transform_match(self.mock_matches[0])
        self.assertEqual(len(encoded['rad_picks']), 5)
        self.assertEqual(len(encoded['dire_picks']), 5)
        self.assertEqual(len(encoded['rad_ranks']), 5)
        self.assertEqual(len(encoded['rad_mmrs']), 5)
        self.assertEqual(encoded['target'], 1.0)

    def test_unknown_hero_maps_to_unk(self):
        encoder = NeuralFeatureEncoder()
        encoder.fit(self.mock_matches)

        # Hero 999 was not in training
        tok = encoder.encode_hero(999)
        self.assertEqual(tok, encoder.UNK_TOKEN)

    def test_encoder_serialization(self):
        encoder = NeuralFeatureEncoder()
        encoder.fit(self.mock_matches)

        save_path = os.path.join(self.temp_dir, "encoder.json")
        encoder.save(save_path)

        loaded = NeuralFeatureEncoder.load(save_path)
        self.assertEqual(encoder.hero_to_idx, loaded.hero_to_idx)
        self.assertEqual(encoder.patch_to_idx, loaded.patch_to_idx)
        self.assertAlmostEqual(encoder.mmr_mean, loaded.mmr_mean)

    def test_collate_dota_batch(self):
        encoder = NeuralFeatureEncoder()
        encoder.fit(self.mock_matches)

        encoded_samples = [encoder.transform_match(m) for m in self.mock_matches]
        batch = collate_dota_batch(encoded_samples)

        # Validate Shapes
        self.assertEqual(list(batch['picks'].shape), [2, 2, 5])
        self.assertEqual(batch['picks'].dtype, torch.long)

        self.assertEqual(list(batch['bans'].shape[:2]), [2, 2])
        self.assertEqual(batch['bans'].dtype, torch.long)
        self.assertEqual(batch['bans_mask'].dtype, torch.bool)

        self.assertEqual(list(batch['ranks'].shape), [2, 2, 5])
        self.assertEqual(batch['ranks'].dtype, torch.long)

        self.assertEqual(list(batch['mmr'].shape), [2, 2, 5, 1])
        self.assertEqual(batch['mmr'].dtype, torch.float32)

        self.assertEqual(list(batch['skill_masks'].shape), [2, 2, 5, 2])
        self.assertEqual(batch['skill_masks'].dtype, torch.float32)

        self.assertEqual(list(batch['patch'].shape), [2])
        self.assertEqual(batch['patch'].dtype, torch.long)

        self.assertEqual(list(batch['target'].shape), [2])
        self.assertEqual(batch['target'].dtype, torch.float32)

        self.assertEqual(list(batch['composition_delta'].shape), [2, 21])
        self.assertEqual(batch['composition_delta'].dtype, torch.float32)

if __name__ == '__main__':
    unittest.main()
