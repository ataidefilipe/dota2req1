import unittest
import numpy as np
import os
import tempfile
from src.features.composition_features import (
    CompositionFeatureExtractor,
    CompositionScaler,
    FEATURE_NAMES,
    TEAM_FEATURE_NAMES
)

class TestCompositionFeatures(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.extractor = CompositionFeatureExtractor(constants_dir='data/constants')

    def test_feature_names_length(self):
        self.assertEqual(len(FEATURE_NAMES), 21)
        self.assertEqual(len(TEAM_FEATURE_NAMES), 20)
        self.assertIn("attack_type_diversity", FEATURE_NAMES)
        self.assertIn("avg_max_cooldown", FEATURE_NAMES)
        self.assertIn("long_cd_count", FEATURE_NAMES)
        self.assertIn("spammer_count", FEATURE_NAMES)
        self.assertIn("bkb_pierce_count", FEATURE_NAMES)
        self.assertIn("strong_dispel_count", FEATURE_NAMES)
        self.assertIn("hard_save_count", FEATURE_NAMES)
        self.assertIn("avg_synergy", FEATURE_NAMES)
        self.assertIn("max_synergy", FEATURE_NAMES)
        self.assertIn("min_synergy", FEATURE_NAMES)
        self.assertIn("early_power", FEATURE_NAMES)
        self.assertIn("late_power", FEATURE_NAMES)
        self.assertIn("scaling_factor", FEATURE_NAMES)
        self.assertIn("matchup_advantage", FEATURE_NAMES)

    def test_hero_profiles_loaded(self):
        self.assertGreaterEqual(len(self.extractor.profiles), 120)
        # Anti-Mage (ID 1)
        am = self.extractor.profiles.get(1)
        self.assertIsNotNone(am)
        self.assertEqual(am.is_ranged, 0.0)
        self.assertEqual(am.is_carry, 1.0)

        # Axe (ID 2): BKB-pierce disable (Berserker's Call)
        axe = self.extractor.profiles.get(2)
        self.assertIsNotNone(axe)
        self.assertEqual(axe.has_bkb_pierce, 1.0)

        # Abaddon (ID 102): Strong dispel
        aba = self.extractor.profiles.get(102)
        self.assertIsNotNone(aba)
        self.assertEqual(aba.has_strong_dispel, 1.0)

        # Dazzle (ID 50): Hard save (Shallow Grave)
        daz = self.extractor.profiles.get(50)
        self.assertIsNotNone(daz)
        self.assertEqual(daz.has_hard_save, 1.0)

        # Spectre (ID 67): Hard scaling (scaling_factor > 0)
        spec = self.extractor.profiles.get(67)
        self.assertIsNotNone(spec)
        self.assertGreater(spec.scaling_factor, 0.05)

        # Chen (ID 66): Early tempo (scaling_factor < 0)
        chen = self.extractor.profiles.get(66)
        self.assertIsNotNone(chen)
        self.assertLess(chen.scaling_factor, -0.05)

    def test_team_vector_permutation_invariance(self):
        team_a = [1, 18, 5, 33, 99]
        team_b = [99, 5, 1, 33, 18]

        vec_a = self.extractor.team_vector(team_a)
        vec_b = self.extractor.team_vector(team_b)

        self.assertEqual(vec_a.shape, (20,))
        np.testing.assert_allclose(vec_a, vec_b, atol=1e-6)

    def test_matchup_advantage_symmetry(self):
        rad = [1, 18, 5, 33, 99]
        dire = [2, 14, 25, 40, 74]

        adv_rad_dire = self.extractor.calc_matchup_advantage(rad, dire)
        adv_dire_rad = self.extractor.calc_matchup_advantage(dire, rad)

        # Rigorosamente antissimétrico
        self.assertAlmostEqual(adv_dire_rad, -adv_rad_dire, places=6)

        # Vantagem contra si mesmo deve ser identicamente 0.0
        adv_self = self.extractor.calc_matchup_advantage(rad, rad)
        self.assertAlmostEqual(adv_self, 0.0, places=6)

    def test_team_differential_symmetry(self):
        rad = [1, 18, 5, 33, 99]
        dire = [2, 14, 25, 40, 74]

        delta_rad_dire = self.extractor.transform(rad, dire)
        delta_dire_rad = self.extractor.transform(dire, rad)

        self.assertEqual(delta_rad_dire.shape, (21,))
        self.assertEqual(delta_dire_rad.shape, (21,))

        # Delta(Dire, Rad) deve ser exatamente -Delta(Rad, Dire)
        np.testing.assert_allclose(delta_dire_rad, -delta_rad_dire, atol=1e-6)

        # Delta(Rad, Rad) deve ser 0 em todas as 21 posições
        delta_zero = self.extractor.transform(rad, rad)
        np.testing.assert_allclose(delta_zero, np.zeros(21), atol=1e-6)

    def test_scaler_preserves_sign_inversion(self):
        X_mock = np.random.randn(5, 21).astype(np.float32)

        scaler = CompositionScaler()
        scaler.fit(X_mock)

        sample = X_mock[0:1]
        sample_inverted = -sample

        norm_sample = scaler.transform(sample)
        norm_inverted = scaler.transform(sample_inverted)

        # Deve cumprir rigorosamente: transform(-x) == -transform(x)
        np.testing.assert_allclose(norm_inverted, -norm_sample, atol=1e-6)

    def test_scaler_save_load(self):
        X_mock = np.random.randn(5, 21).astype(np.float32)

        scaler = CompositionScaler().fit(X_mock)
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as tmp:
            tmp_path = tmp.name

        try:
            scaler.save(tmp_path)
            loaded = CompositionScaler.load(tmp_path)

            np.testing.assert_allclose(scaler.scale_, loaded.scale_, atol=1e-6)
            self.assertTrue(loaded.fitted)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

if __name__ == '__main__':
    unittest.main()
