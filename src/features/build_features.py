import logging
import sqlite3
import json
import numpy as np
from scipy import sparse
import joblib
import os
from collections import defaultdict
from ..db.repo import get_db_connection
from ..config import Config
from .composition_features import CompositionFeatureExtractor, FEATURE_NAMES

logger = logging.getLogger(__name__)

class DatasetBuilder:
    def __init__(self):
        self.feature_map = {} # feature_name -> index
        self.feature_names = [] # index -> feature_name (for verification)
        self.comp_extractor = CompositionFeatureExtractor()
        
    def _get_feature_index(self, feature_name):
        if feature_name not in self.feature_map:
            idx = len(self.feature_names)
            self.feature_map[feature_name] = idx
            self.feature_names.append(feature_name)
        return self.feature_map[feature_name]

    def load_raw_data(self, patch_id: int = None):
        """
        Loads all matches that pass the basic valid flag (target patch, CM, duration >= 1800s).
        We fetch match_id, radiant_win, patch_id, start_time.
        Then we fetch all draft events for these matches.
        """
        target_patch = patch_id if patch_id is not None else Config.TARGET_PATCH
        query_matches = """
            SELECT match_id, radiant_win, patch_id, start_time
            FROM matches
            WHERE patch_id = ? AND duration >= 1800 AND game_mode = 2 AND radiant_win IS NOT NULL
            ORDER BY start_time ASC, match_id ASC
        """
        
        # We need draft events. It's often faster to fetch all and process in memory 
        # if dataset fits (MVP), or join in SQL.
        # Let's do a join to ensure we only get matches with draft info.
        # But a massive join might be slow.
        # Optimization: Fetch matches first, then fetch draft events for those matches.
        
        with get_db_connection() as conn:
            # 1. Fetch Matches
            cursor = conn.execute(query_matches, (target_patch,))
            matches = cursor.fetchall()
            
            # matches structure: [(match_id, win, patch, start), ...]
            if not matches:
                logger.warning(f"No matches found in DB for patch {target_patch}.")
                return [], [], [], {}

            match_ids = [m[0] for m in matches]
            y_list = [1 if m[1] else 0 for m in matches]
            meta_list = [{'match_id': m[0], 'start_time': m[3], 'patch_id': m[2]} for m in matches]
            
            logger.info(f"Loaded {len(matches)} matches for patch {target_patch}. Fetching draft events...")
            
            # 2. Fetch Draft Events
            # Getting all events at once might be heavy ? 
            # For MVP with <100k matches, it's fine.
            # IN clause might fail if too many IDs.
            # Let's just select ALL draft events and filter in python if needed, 
            # or rely on the fact that we only saved events for saved matches.
            
            cursor = conn.execute("SELECT match_id, team, hero_id, is_pick FROM draft_events")
            # We need to group by match_id
            match_events = defaultdict(list)
            for row in cursor:
                # row: match_id, team, hero_id, is_pick
                match_events[row[0]].append(row)
                
            return match_ids, y_list, meta_list, match_events

    def build_dataset(self, output_dir='data/processed', patch_id: int = None):
        match_ids, y_list, meta_list, match_events = self.load_raw_data(patch_id=patch_id)
        
        if not match_ids:
            return

        # Sparse matrix construction using COO format (rows, cols, data)
        rows = []
        cols = []
        data = []
        
        valid_indices = []
        
        for i, mid in enumerate(match_ids):
            events = match_events.get(mid)
            if not events:
                # Should have been filtered, but safe check
                continue
                
            # Valid match for dataset
            valid_indices.append(i)
            # current_row = len(valid_indices) - 1 # Logic error if we skip? 
            # Ideally y_list should be aligned. 
            # Let's rebuild y based on valid matches
            
            # Features
            # Patch
            patch_id = meta_list[i]['patch_id']
            if patch_id:
                f_name = f"patch_{patch_id}"
                col_idx = self._get_feature_index(f_name)
                rows.append(len(valid_indices)-1)
                cols.append(col_idx)
                data.append(1)
            
            # Draft
            # team: 0 (Radiant), 1 (Dire)
            # is_pick: 1 (Pick), 0 (Ban)
            for _, team, hero_id, is_pick in events:
                # prefix: rad/dire _ pick/ban _ hero_id
                side = "rad" if team == 0 else "dire"
                type_ = "pick" if is_pick else "ban"
                f_name = f"{side}_{type_}_{hero_id}"
                
                col_idx = self._get_feature_index(f_name)
                rows.append(len(valid_indices)-1)
                cols.append(col_idx)
                data.append(1)

        # Re-align Y and Meta
        y_final = np.array([y_list[i] for i in valid_indices], dtype=np.int8)
        meta_final = [meta_list[i] for i in valid_indices]
        
        # Compute Composition Features (11 features)
        comp_deltas = []
        for i in valid_indices:
            mid = match_ids[i]
            events = match_events.get(mid, [])
            rad_picks = [ev[2] for ev in events if ev[3] == 1 and ev[1] == 0]
            dire_picks = [ev[2] for ev in events if ev[3] == 1 and ev[1] == 1]
            if len(rad_picks) == 5 and len(dire_picks) == 5:
                delta = self.comp_extractor.transform(rad_picks, dire_picks)
            else:
                delta = np.zeros(len(FEATURE_NAMES), dtype=np.float32)
            comp_deltas.append(delta)
        comp_deltas = np.array(comp_deltas, dtype=np.float32)

        # Build Sparse Matrix
        n_samples = len(valid_indices)
        n_features = len(self.feature_names)
        
        X = sparse.coo_matrix((data, (rows, cols)), shape=(n_samples, n_features))
        X_csr = X.tocsr()
        
        # Save
        os.makedirs(output_dir, exist_ok=True)
        sparse.save_npz(self._path(output_dir, 'X.npz'), X_csr)
        np.save(self._path(output_dir, 'y.npy'), y_final)
        np.save(self._path(output_dir, 'composition_delta.npy'), comp_deltas)
        
        with open(self._path(output_dir, 'composition_feature_names.json'), 'w', encoding='utf-8') as f:
            json.dump(list(FEATURE_NAMES), f, indent=2)

        with open(self._path(output_dir, 'feature_map.json'), 'w') as f:
            json.dump({'feature_map': self.feature_map, 'feature_names': self.feature_names}, f)
            
        # Also save meta for temporal split
        joblib.dump(meta_final, self._path(output_dir, 'meta.joblib'))
        
        logger.info(f"Dataset built. Sparse Shape: {X_csr.shape} | Comp Shape: {comp_deltas.shape}. Saved to {output_dir}")

    def _path(self, dir, fname):
        return os.path.join(dir, fname)

if __name__ == "__main__":
    from ..logger import setup_logging
    import argparse
    setup_logging()

    parser = argparse.ArgumentParser(description="Construção de Features Tabulares / Esparsas")
    parser.add_argument("--patch", type=int, default=None, help=f"ID do patch (default: {Config.TARGET_PATCH})")
    parser.add_argument("--output-dir", type=str, default="data/processed", help="Diretório de saída")
    args = parser.parse_args()

    builder = DatasetBuilder()
    builder.build_dataset(output_dir=args.output_dir, patch_id=args.patch)

