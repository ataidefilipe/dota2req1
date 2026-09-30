import os
import json
import logging
import sqlite3
import numpy as np
import joblib
from typing import List, Dict, Any, Optional, Tuple
from ..config import Config
from ..db.repo import get_db_connection

logger = logging.getLogger(__name__)

class NeuralFeatureEncoder:
    """
    Encoder determinístico para tensores neurais:
    - Mapeia hero_id para tokens contíguos (PAD=0, UNK=1).
    - Mapeia patch_id para tokens contíguos (MISSING=0, UNK=1).
    - Mapeia rank_tier para tokens categóricos (MISSING=0, UNK=1).
    - Normaliza computed_mmr com média, desvio e clipping calculados estritamente no treino.
    """
    PAD_TOKEN = 0
    UNK_TOKEN = 1
    MISSING_TOKEN = 0

    def __init__(self):
        self.hero_to_idx: Dict[int, int] = {}
        self.idx_to_hero: Dict[int, int] = {}
        self.patch_to_idx: Dict[int, int] = {}
        self.rank_to_idx: Dict[int, int] = {}

        self.mmr_mean: float = 0.0
        self.mmr_std: float = 1.0
        self.mmr_q01: float = 0.0
        self.mmr_q99: float = 8000.0

        self.fitted: bool = False

    def fit(self, train_matches: List[Dict[str, Any]]) -> "NeuralFeatureEncoder":
        """
        Ajusta os vocabulários e estatísticas de normalização exclusivamente no treino.
        """
        logger.info(f"Ajustando NeuralFeatureEncoder com {len(train_matches)} partidas de treino...")

        # 1. Vocabulário de Heróis (PAD=0, UNK=1)
        hero_ids = set()
        patch_ids = set()
        rank_tiers = set()
        mmrs = []

        for m in train_matches:
            # Picks & Bans
            for p in m.get('rad_picks', []) + m.get('dire_picks', []):
                if p is not None:
                    hero_ids.add(int(p))
            for b in m.get('rad_bans', []) + m.get('dire_bans', []):
                if b is not None:
                    hero_ids.add(int(b))

            # Patch
            patch = m.get('patch_id')
            if patch is not None:
                patch_ids.add(int(patch))

            # Jogadores (Rank & MMR)
            for player in m.get('rad_players', []) + m.get('dire_players', []):
                rank = player.get('rank_tier')
                if rank is not None:
                    rank_tiers.add(int(rank))
                mmr = player.get('computed_mmr')
                if mmr is not None and not np.isnan(mmr):
                    mmrs.append(float(mmr))

        # Construir mapa de heróis
        self.hero_to_idx = {"PAD": 0, "UNK": 1}
        self.idx_to_hero = {0: 0, 1: 0}
        for idx, hid in enumerate(sorted(hero_ids), start=2):
            self.hero_to_idx[hid] = idx
            self.idx_to_hero[idx] = hid

        # Construir mapa de patches (MISSING=0, UNK=1)
        self.patch_to_idx = {"MISSING": 0, "UNK": 1}
        for idx, pid in enumerate(sorted(patch_ids), start=2):
            self.patch_to_idx[pid] = idx

        # Construir mapa de ranks (MISSING=0, UNK=1)
        self.rank_to_idx = {"MISSING": 0, "UNK": 1}
        for idx, rk in enumerate(sorted(rank_tiers), start=2):
            self.rank_to_idx[rk] = idx

        # Estatísticas de MMR
        if mmrs:
            mmr_array = np.array(mmrs, dtype=np.float32)
            self.mmr_q01 = float(np.percentile(mmr_array, 1))
            self.mmr_q99 = float(np.percentile(mmr_array, 99))
            clipped = np.clip(mmr_array, self.mmr_q01, self.mmr_q99)
            self.mmr_mean = float(np.mean(clipped))
            self.mmr_std = float(np.std(clipped))
            if self.mmr_std < 1e-4:
                self.mmr_std = 1.0
        else:
            self.mmr_mean = 0.0
            self.mmr_std = 1.0
            self.mmr_q01 = 0.0
            self.mmr_q99 = 8000.0

        self.fitted = True
        logger.info(
            f"Encoder ajustado: {len(self.hero_to_idx)} tokens de heróis, "
            f"{len(self.patch_to_idx)} tokens de patches, {len(self.rank_to_idx)} tokens de ranks. "
            f"MMR: média={self.mmr_mean:.1f}, std={self.mmr_std:.1f}"
        )
        return self

    def encode_hero(self, hero_id: Optional[int]) -> int:
        if hero_id is None or hero_id == 0:
            return self.PAD_TOKEN
        return self.hero_to_idx.get(hero_id, self.UNK_TOKEN)

    def encode_patch(self, patch_id: Optional[int]) -> int:
        if patch_id is None:
            return self.patch_to_idx.get("MISSING", 0)
        return self.patch_to_idx.get(patch_id, self.patch_to_idx.get("UNK", 1))

    def encode_rank(self, rank_tier: Optional[int]) -> int:
        if rank_tier is None:
            return self.rank_to_idx.get("MISSING", 0)
        return self.rank_to_idx.get(rank_tier, self.rank_to_idx.get("UNK", 1))

    def normalize_mmr(self, mmr: Optional[float]) -> Tuple[float, bool]:
        """Retorna (mmr_normalizado, observado_bool). Ausentes retornam (0.0, False)."""
        if mmr is None or np.isnan(mmr):
            return 0.0, False
        clipped = np.clip(float(mmr), self.mmr_q01, self.mmr_q99)
        norm = (clipped - self.mmr_mean) / max(self.mmr_std, 1e-4)
        return float(norm), True

    def transform_match(self, match: Dict[str, Any]) -> Dict[str, Any]:
        """
        Converte uma partida estruturada em arrays/listas codificados.
        """
        if not self.fitted:
            raise RuntimeError("O encoder precisa ser ajustado (fit) antes do transform.")

        # 1. Picks (5 Radiant, 5 Dire)
        rad_picks = [self.encode_hero(h) for h in match.get('rad_picks', [])[:5]]
        while len(rad_picks) < 5:
            rad_picks.append(self.PAD_TOKEN)

        dire_picks = [self.encode_hero(h) for h in match.get('dire_picks', [])[:5]]
        while len(dire_picks) < 5:
            dire_picks.append(self.PAD_TOKEN)

        # 2. Bans (lista de tamanho variável)
        rad_bans_raw = match.get('rad_bans', []) or []
        dire_bans_raw = match.get('dire_bans', []) or []
        rad_bans = [self.encode_hero(h) for h in rad_bans_raw if h is not None]
        dire_bans = [self.encode_hero(h) for h in dire_bans_raw if h is not None]

        # 3. Jogadores (Habilidade)
        def process_team_players(players_list):
            ranks = []
            mmrs = []
            rank_masks = []
            mmr_masks = []
            players = players_list or []
            for i in range(5):
                if i < len(players):
                    p = players[i]
                    r_raw = p.get('rank_tier')
                    r_tok = self.encode_rank(r_raw)
                    r_obs = r_raw is not None

                    m_raw = p.get('computed_mmr')
                    m_norm, m_obs = self.normalize_mmr(m_raw)
                else:
                    r_tok = self.rank_to_idx.get("MISSING", 0)
                    r_obs = False
                    m_norm = 0.0
                    m_obs = False

                ranks.append(r_tok)
                mmrs.append(m_norm)
                rank_masks.append(r_obs)
                mmr_masks.append(m_obs)

            return ranks, mmrs, rank_masks, mmr_masks

        rad_ranks, rad_mmrs, rad_r_masks, rad_m_masks = process_team_players(match.get('rad_players'))
        dire_ranks, dire_mmrs, dire_r_masks, dire_m_masks = process_team_players(match.get('dire_players'))

        # 4. Patch
        patch_tok = self.encode_patch(match.get('patch_id'))

        # 5. Target
        y = match.get('radiant_win')
        target = float(1.0 if y else 0.0) if y is not None else None

        return {
            'match_id': match.get('match_id'),
            'start_time': match.get('start_time'),
            'rad_picks': rad_picks,
            'dire_picks': dire_picks,
            'rad_bans': rad_bans,
            'dire_bans': dire_bans,
            'rad_bans_available': len(rad_bans_raw) > 0,
            'dire_bans_available': len(dire_bans_raw) > 0,
            'rad_ranks': rad_ranks,
            'dire_ranks': dire_ranks,
            'rad_mmrs': rad_mmrs,
            'dire_mmrs': dire_mmrs,
            'rad_rank_masks': rad_r_masks,
            'dire_rank_masks': dire_r_masks,
            'rad_mmr_masks': rad_m_masks,
            'dire_mmr_masks': dire_m_masks,
            'patch_token': patch_tok,
            'target': target
        }

    def save(self, filepath: str):
        data = {
            'hero_to_idx': self.hero_to_idx,
            'idx_to_hero': self.idx_to_hero,
            'patch_to_idx': self.patch_to_idx,
            'rank_to_idx': self.rank_to_idx,
            'mmr_mean': self.mmr_mean,
            'mmr_std': self.mmr_std,
            'mmr_q01': self.mmr_q01,
            'mmr_q99': self.mmr_q99,
            'num_heroes': len(self.hero_to_idx),
            'num_patches': len(self.patch_to_idx),
            'num_ranks': len(self.rank_to_idx)
        }
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        logger.info(f"NeuralFeatureEncoder salvo em {filepath}")

    @classmethod
    def load(cls, filepath: str) -> "NeuralFeatureEncoder":
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        encoder = cls()
        encoder.hero_to_idx = {
            (int(k) if k.isdigit() else k): v for k, v in data['hero_to_idx'].items()
        }
        encoder.idx_to_hero = {int(k): v for k, v in data['idx_to_hero'].items()}
        encoder.patch_to_idx = {
            (int(k) if k.isdigit() else k): v for k, v in data['patch_to_idx'].items()
        }
        encoder.rank_to_idx = {
            (int(k) if k.isdigit() else k): v for k, v in data['rank_to_idx'].items()
        }
        encoder.mmr_mean = data['mmr_mean']
        encoder.mmr_std = data['mmr_std']
        encoder.mmr_q01 = data['mmr_q01']
        encoder.mmr_q99 = data['mmr_q99']
        encoder.fitted = True
        return encoder


def extract_raw_matches_from_db(db_path: str = None) -> List[Dict[str, Any]]:
    """
    Extrai do SQLite todas as partidas válidas com seus respectivos draft events e players,
    ordenadas por start_time ASC para split temporal estrito.
    """
    db_file = db_path or Config.SQLITE_PATH
    logger.info(f"Extraindo partidas do banco SQLite: {db_file}")

    with sqlite3.connect(db_file) as conn:
        # 1. Partidas
        query_matches = """
            SELECT match_id, radiant_win, patch_id, start_time, duration, dataset
            FROM matches
            ORDER BY start_time ASC
        """
        matches_rows = conn.execute(query_matches).fetchall()
        if not matches_rows:
            logger.warning("Nenhuma partida encontrada no banco.")
            return []

        # 2. Draft Events
        events_rows = conn.execute("""
            SELECT match_id, [order], is_pick, team, hero_id
            FROM draft_events
            ORDER BY [order] ASC
        """).fetchall()

        events_by_match = {}
        for mid, order, is_pick, team, hero_id in events_rows:
            if mid not in events_by_match:
                events_by_match[mid] = {'rad_picks': [], 'dire_picks': [], 'rad_bans': [], 'dire_bans': []}
            if is_pick:
                if team == 0:
                    events_by_match[mid]['rad_picks'].append(hero_id)
                else:
                    events_by_match[mid]['dire_picks'].append(hero_id)
            else:
                if team == 0:
                    events_by_match[mid]['rad_bans'].append(hero_id)
                else:
                    events_by_match[mid]['dire_bans'].append(hero_id)

        # 3. Match Players
        players_rows = conn.execute("""
            SELECT match_id, player_slot, rank_tier, computed_mmr
            FROM match_players
            ORDER BY player_slot ASC
        """).fetchall()

        players_by_match = {}
        for mid, slot, rank_tier, computed_mmr in players_rows:
            if mid not in players_by_match:
                players_by_match[mid] = {'rad_players': [], 'dire_players': []}
            p_dict = {'player_slot': slot, 'rank_tier': rank_tier, 'computed_mmr': computed_mmr}
            if slot < 128:
                players_by_match[mid]['rad_players'].append(p_dict)
            else:
                players_by_match[mid]['dire_players'].append(p_dict)

    # 4. Consolidar lista de dicionários
    results = []
    for mid, radiant_win, patch_id, start_time, duration, dataset in matches_rows:
        draft = events_by_match.get(mid, {'rad_picks': [], 'dire_picks': [], 'rad_bans': [], 'dire_bans': []})
        players = players_by_match.get(mid, {'rad_players': [], 'dire_players': []})

        # Apenas partidas com 5 picks de cada lado
        if len(draft['rad_picks']) == 5 and len(draft['dire_picks']) == 5:
            results.append({
                'match_id': mid,
                'radiant_win': bool(radiant_win),
                'patch_id': patch_id,
                'start_time': start_time,
                'duration': duration,
                'dataset': dataset,
                'rad_picks': draft['rad_picks'],
                'dire_picks': draft['dire_picks'],
                'rad_bans': draft['rad_bans'],
                'dire_bans': draft['dire_bans'],
                'rad_players': players['rad_players'],
                'dire_players': players['dire_players']
            })

    logger.info(f"Total de {len(results)} partidas completas e válidas extraídas com sucesso.")
    return results


def prepare_neural_dataset(
    output_dir: str = 'data/processed_neural',
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15
) -> Dict[str, Any]:
    """
    Executa o pipeline completo da Fase 2:
    - Extrai snapshot do banco
    - Aplica split temporal 70 / 15 / 15 por start_time
    - Fit do encoder unicamente no treino
    - Serializa encoder.json, splits.json e amostras codificadas (.joblib)
    """
    matches = extract_raw_matches_from_db()
    if not matches:
        raise ValueError("Nenhuma partida válida disponível para processar.")

    n_total = len(matches)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    train_raw = matches[:n_train]
    val_raw = matches[n_train:n_train + n_val]
    test_raw = matches[n_train + n_val:]

    logger.info(
        f"Split Temporal: Total={n_total} | "
        f"Treino={len(train_raw)} (70%) | "
        f"Validação={len(val_raw)} (15%) | "
        f"Teste={len(test_raw)} (15%)"
    )

    # Fit apenas no treino
    encoder = NeuralFeatureEncoder()
    encoder.fit(train_raw)

    # Transformar cada partição
    train_encoded = [encoder.transform_match(m) for m in train_raw]
    val_encoded = [encoder.transform_match(m) for m in val_raw]
    test_encoded = [encoder.transform_match(m) for m in test_raw]

    # Salvar artefatos
    os.makedirs(output_dir, exist_ok=True)
    encoder_path = os.path.join(output_dir, 'encoder.json')
    encoder.save(encoder_path)

    splits_meta = {
        'n_total': n_total,
        'train': {
            'count': len(train_raw),
            'start_time_min': train_raw[0]['start_time'] if train_raw else None,
            'start_time_max': train_raw[-1]['start_time'] if train_raw else None,
            'match_ids': [m['match_id'] for m in train_raw]
        },
        'val': {
            'count': len(val_raw),
            'start_time_min': val_raw[0]['start_time'] if val_raw else None,
            'start_time_max': val_raw[-1]['start_time'] if val_raw else None,
            'match_ids': [m['match_id'] for m in val_raw]
        },
        'test': {
            'count': len(test_raw),
            'start_time_min': test_raw[0]['start_time'] if test_raw else None,
            'start_time_max': test_raw[-1]['start_time'] if test_raw else None,
            'match_ids': [m['match_id'] for m in test_raw]
        }
    }
    with open(os.path.join(output_dir, 'splits.json'), 'w', encoding='utf-8') as f:
        json.dump(splits_meta, f, indent=2)

    joblib.dump(train_encoded, os.path.join(output_dir, 'train_encoded.joblib'))
    joblib.dump(val_encoded, os.path.join(output_dir, 'val_encoded.joblib'))
    joblib.dump(test_encoded, os.path.join(output_dir, 'test_encoded.joblib'))

    logger.info(f"Dataset neural salvo com sucesso em '{output_dir}'.")
    return {
        'splits': splits_meta,
        'encoder': encoder,
        'counts': {'train': len(train_encoded), 'val': len(val_encoded), 'test': len(test_encoded)}
    }

if __name__ == "__main__":
    from ..logger import setup_logging
    setup_logging()
    prepare_neural_dataset()
