import argparse
import logging
import json
import joblib
import os
import numpy as np
from scipy import sparse
from ..config import Config

logger = logging.getLogger(__name__)

class WinPredictor:
    def __init__(self, model_dir='models', data_dir='data/processed', constants_file='data/heroes.json'):
        self.model_path = os.path.join(model_dir, 'draft_win_lr.joblib')
        self.scaler_path = os.path.join(model_dir, 'draft_win_scaler.joblib')
        self.feature_map_path = os.path.join(data_dir, 'feature_map.json')
        self.heroes_path = constants_file
        
        self.model = None
        self.scaler = None
        self.comp_extractor = None
        self.feature_map = None
        self.heroes_name_to_id = {}
        self.heroes_id_to_name = {}
        
    def load(self):
        # Load Model
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found at {self.model_path}. Run train.py first.")
        self.model = joblib.load(self.model_path)
        
        # Load Scaler if present
        if os.path.exists(self.scaler_path):
            self.scaler = joblib.load(self.scaler_path)
            from ..features.composition_features import CompositionFeatureExtractor
            self.comp_extractor = CompositionFeatureExtractor()

        # Load Feature Map
        if not os.path.exists(self.feature_map_path):
             raise FileNotFoundError(f"Feature map not found at {self.feature_map_path}. Run build_features.py first.")
        with open(self.feature_map_path, 'r') as f:
            data = json.load(f)
            self.feature_map = data.get('feature_map', {})
            
        # Load Heroes (optional but needed for name resolution)
        if os.path.exists(self.heroes_path):
            with open(self.heroes_path, 'r') as f:
                heroes_list = json.load(f)
                for h in heroes_list:
                    # Map 'Anti-Mage' -> 1
                    self.heroes_name_to_id[h['localized_name']] = h['id']
                    # Map 'npc_dota_hero_antimage' -> 1
                    self.heroes_name_to_id[h['name']] = h['id']
                    self.heroes_id_to_name[h['id']] = h['localized_name']
        else:
            logger.warning(f"Heroes file not found at {self.heroes_path}. Name resolution limited.")

    def resolve_hero(self, identifier):
        """Resolves a hero name or ID to an ID (int)."""
        # If it's already an ID
        try:
            return int(identifier)
        except ValueError:
            pass
            
        # Try exact match
        if identifier in self.heroes_name_to_id:
            return self.heroes_name_to_id[identifier]
            
        # Try fuzzy or case-insensitive (simple version)
        for name, hid in self.heroes_name_to_id.items():
            if name.lower() == str(identifier).lower():
                return hid
                
        logger.warning(f"Hero '{identifier}' not found.")
        return None

    def predict(self, radiant_heroes, dire_heroes, radiant_bans=None, dire_bans=None, patch_id=None):
        """
        Predicts radiant win probability.
        Args:
            radiant_heroes: list of hero names/ids
            dire_heroes: list of hero names/ids
            radiant_bans: list of hero names/ids (optional)
            dire_bans: list of hero names/ids (optional)
            patch_id: int (optional)
        """
        if not self.model:
            self.load()
            
        # 1. Build Feature Vector
        # We need to construct a sparse row vector of shape (1, n_features)
        
        # Get max feature index to know size
        n_features = len(self.feature_map)
        
        rows = []
        cols = []
        data = []
        
        def add_feature(feat_name):
            if feat_name in self.feature_map:
                idx = self.feature_map[feat_name]
                rows.append(0)
                cols.append(idx)
                data.append(1)
            # else: ignored (not in training set)
            
        # Process Picks
        # Format: rad_pick_{id}, dire_pick_{id}
        for h in radiant_heroes:
            hid = self.resolve_hero(h)
            if hid: add_feature(f"rad_pick_{hid}")
            
        for h in dire_heroes:
            hid = self.resolve_hero(h)
            if hid: add_feature(f"dire_pick_{hid}")
            
        # Process Bans
        if radiant_bans:
            for h in radiant_bans:
                hid = self.resolve_hero(h)
                if hid: add_feature(f"rad_ban_{hid}")
                
        if dire_bans:
            for h in dire_bans:
                hid = self.resolve_hero(h)
                if hid: add_feature(f"dire_ban_{hid}")
                
        # Process Patch
        if patch_id:
            add_feature(f"patch_{patch_id}")
            
        # Create Sparse Matrix
        X = sparse.coo_matrix((data, (rows, cols)), shape=(1, n_features)).tocsr()
        
        # Se o modelo requer as features de composição
        if hasattr(self.model, 'n_features_in_') and self.model.n_features_in_ > n_features:
            rad_ids = [self.resolve_hero(h) for h in radiant_heroes if self.resolve_hero(h)]
            dire_ids = [self.resolve_hero(h) for h in dire_heroes if self.resolve_hero(h)]
            if len(rad_ids) == 5 and len(dire_ids) == 5 and self.comp_extractor and self.scaler:
                raw_delta = self.comp_extractor.transform(rad_ids, dire_ids)
                scaled_delta = self.scaler.transform(raw_delta.reshape(1, -1))
                X = sparse.hstack([X, sparse.csr_matrix(scaled_delta)], format='csr')
            else:
                extra_dim = self.model.n_features_in_ - n_features
                X = sparse.hstack([X, sparse.csr_matrix((1, extra_dim))], format='csr')

        # Predict
        prob = self.model.predict_proba(X)[0, 1]
        return prob

def main():
    parser = argparse.ArgumentParser(description="Predict Dota 2 Win Probability")
    parser.add_argument("--backend", choices=["lr", "nn"], default="lr", help="Backend preditivo (lr=Regressão Logística, nn=Rede Neural PyTorch)")
    parser.add_argument("--rad", nargs='+', help="Radiant heroes (names or IDs)")
    parser.add_argument("--dire", nargs='+', help="Dire heroes (names or IDs)")
    parser.add_argument("--rad-bans", nargs='+', help="Radiant bans")
    parser.add_argument("--dire-bans", nargs='+', help="Dire bans")
    parser.add_argument("--patch", type=int, help="Patch ID")
    parser.add_argument("--json", type=str, help="Path to JSON payload file (alternative to args)")
    
    args = parser.parse_args()

    # Validação de argumentos
    if args.json:
        with open(args.json, 'r', encoding='utf-8') as f:
            data = json.load(f)
        rad = data.get('radiant_heroes', [])
        dire = data.get('dire_heroes', [])
        rbans = data.get('radiant_bans', [])
        dbans = data.get('dire_bans', [])
        patch = data.get('patch_id')
        rad_p = data.get('rad_players', [])
        dire_p = data.get('dire_players', [])
    else:
        if not args.rad or not args.dire:
            parser.error("É necessário fornecer --rad e --dire, ou passar um payload com --json.")
        rad = args.rad
        dire = args.dire
        rbans = args.rad_bans
        dbans = args.dire_bans
        patch = args.patch
        rad_p = None
        dire_p = None

    if args.backend == "nn":
        from .predict_nn import NeuralWinPredictor
        predictor = NeuralWinPredictor()
        predictor.load()
        prob = predictor.predict(
            radiant_heroes=rad,
            dire_heroes=dire,
            radiant_bans=rbans,
            dire_bans=dbans,
            rad_players=rad_p,
            dire_players=dire_p,
            patch_id=patch
        )
    else:
        predictor = WinPredictor()
        try:
            predictor.load()
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return
        prob = predictor.predict(rad, dire, rbans, dbans, patch)

    print(f"Backend: {args.backend.upper()}")
    print(f"P(Radiant Win): {prob:.4f}")
    if prob >= 0.5:
        print(f"Prediction: Radiant Victory ({prob * 100:.1f}%)")
    else:
        print(f"Prediction: Dire Victory ({(1.0 - prob) * 100:.1f}%)")


if __name__ == "__main__":
    # Minimal logging for existing
    logging.basicConfig(level=logging.ERROR) 
    # But we want to see errors if any
    main()
