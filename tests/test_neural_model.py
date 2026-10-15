import unittest
import torch
import os
import shutil
import tempfile
from src.model.neural_network import DotaWinPredictorNN
from src.model.calibration import TemperatureScaler
from src.model.predict_nn import NeuralWinPredictor

class TestNeuralModel(unittest.TestCase):
    def setUp(self):
        self.device = torch.device('cpu')
        self.model = DotaWinPredictorNN(
            num_heroes=50,
            num_patches=5,
            num_ranks=10,
            hero_dim=16,
            patch_dim=4,
            rank_dim=4,
            dropout=0.0
        )

    def test_forward_pass_shapes(self):
        B = 4
        batch = {
            'picks': torch.randint(1, 40, (B, 2, 5)),
            'bans': torch.randint(1, 40, (B, 2, 3)),
            'bans_mask': torch.ones((B, 2, 3), dtype=torch.bool),
            'bans_available': torch.ones((B, 2, 1), dtype=torch.float32),
            'ranks': torch.randint(1, 8, (B, 2, 5)),
            'mmr': torch.randn((B, 2, 5, 1)),
            'skill_masks': torch.ones((B, 2, 5, 2), dtype=torch.float32),
            'patch': torch.randint(1, 4, (B,)),
            'match_ids': torch.tensor([101, 102, 103, 104])
        }

        self.model.eval()
        with torch.no_grad():
            logits = self.model(batch)

        self.assertEqual(logits.shape, torch.Size([B]))
        self.assertFalse(torch.isnan(logits).any())

    def test_permutation_invariance_picks(self):
        """A ordem dos heróis em uma mesma equipe NÃO deve alterar o logit predito."""
        self.model.eval()

        picks_order1 = torch.tensor([[[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]], dtype=torch.long)
        picks_order2 = torch.tensor([[[5, 4, 3, 2, 1], [6, 7, 8, 9, 10]]], dtype=torch.long)

        batch1 = {
            'picks': picks_order1,
            'bans': torch.zeros((1, 2, 1), dtype=torch.long),
            'bans_mask': torch.zeros((1, 2, 1), dtype=torch.bool),
            'bans_available': torch.zeros((1, 2, 1), dtype=torch.float32),
            'ranks': torch.zeros((1, 2, 5), dtype=torch.long),
            'mmr': torch.zeros((1, 2, 5, 1), dtype=torch.float32),
            'skill_masks': torch.zeros((1, 2, 5, 2), dtype=torch.float32),
            'patch': torch.zeros(1, dtype=torch.long),
            'match_ids': torch.tensor([1])
        }

        batch2 = {
            'picks': picks_order2,
            'bans': torch.zeros((1, 2, 1), dtype=torch.long),
            'bans_mask': torch.zeros((1, 2, 1), dtype=torch.bool),
            'bans_available': torch.zeros((1, 2, 1), dtype=torch.float32),
            'ranks': torch.zeros((1, 2, 5), dtype=torch.long),
            'mmr': torch.zeros((1, 2, 5, 1), dtype=torch.float32),
            'skill_masks': torch.zeros((1, 2, 5, 2), dtype=torch.float32),
            'patch': torch.zeros(1, dtype=torch.long),
            'match_ids': torch.tensor([1])
        }

        with torch.no_grad():
            logit1 = self.model(batch1).item()
            logit2 = self.model(batch2).item()

        self.assertAlmostEqual(logit1, logit2, places=4)

    def test_temperature_scaling(self):
        scaler = TemperatureScaler()
        logits = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
        targets = torch.tensor([0.0, 0.0, 0.0, 1.0, 1.0])

        scaler.fit(logits, targets)
        self.assertTrue(scaler.fitted)
        self.assertGreater(scaler.temperature, 0.0)

        probs = scaler.predict_proba(logits)
        self.assertEqual(probs.shape, logits.shape)
        self.assertTrue((probs >= 0.0).all() and (probs <= 1.0).all())

    def test_composition_delta_forward(self):
        """Verifica que o modelo aceita e propaga tensores de composition_delta."""
        B = 2
        batch = {
            'picks': torch.randint(1, 40, (B, 2, 5)),
            'bans': torch.randint(1, 40, (B, 2, 1)),
            'bans_mask': torch.ones((B, 2, 1), dtype=torch.bool),
            'bans_available': torch.ones((B, 2, 1), dtype=torch.float32),
            'ranks': torch.randint(1, 8, (B, 2, 5)),
            'mmr': torch.randn((B, 2, 5, 1)),
            'skill_masks': torch.ones((B, 2, 5, 2), dtype=torch.float32),
            'patch': torch.randint(1, 4, (B,)),
            'match_ids': torch.tensor([1, 2]),
            'composition_delta': torch.randn((B, 21), dtype=torch.float32)
        }
        self.model.eval()
        with torch.no_grad():
            logits = self.model(batch)
        self.assertEqual(logits.shape, torch.Size([B]))
        self.assertFalse(torch.isnan(logits).any())

if __name__ == '__main__':
    unittest.main()

