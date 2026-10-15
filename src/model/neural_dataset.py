import os
import torch
from torch.utils.data import Dataset, DataLoader
import joblib
from typing import List, Dict, Any, Optional, Tuple
from ..features.neural_features import NeuralFeatureEncoder

class DotaNeuralDataset(Dataset):
    """
    PyTorch Dataset que encapsula amostras de partidas codificadas pelo NeuralFeatureEncoder.
    """
    def __init__(self, data: Any):
        if isinstance(data, str):
            if not os.path.exists(data):
                raise FileNotFoundError(f"Arquivo de dados não encontrado: {data}")
            self.samples = joblib.load(data)
        elif isinstance(data, list):
            self.samples = data
        else:
            raise TypeError("Data deve ser um caminho de arquivo (.joblib) ou lista de amostras.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.samples[idx]


def collate_dota_batch(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """
    Empacota uma lista de amostras individuais nos tensores especificados pela arquitetura:
    - picks: [B, 2, 5] (torch.long)
    - bans: [B, 2, max_bans_len] (torch.long)
    - bans_mask: [B, 2, max_bans_len] (torch.bool)
    - bans_available: [B, 2, 1] (torch.float32)
    - ranks: [B, 2, 5] (torch.long)
    - mmr: [B, 2, 5, 1] (torch.float32)
    - skill_masks: [B, 2, 5, 2] (torch.float32)
    - patch: [B] (torch.long)
    - target: [B] (torch.float32)
    - match_ids: [B] (torch.long)
    """
    B = len(batch)

    # 1. Picks: [B, 2, 5]
    picks = torch.zeros((B, 2, 5), dtype=torch.long)
    for i, s in enumerate(batch):
        picks[i, 0] = torch.tensor(s['rad_picks'], dtype=torch.long)
        picks[i, 1] = torch.tensor(s['dire_picks'], dtype=torch.long)

    # 2. Bans: tamanho dinâmico no batch com padding=0
    max_bans = 0
    for s in batch:
        max_bans = max(max_bans, len(s['rad_bans']), len(s['dire_bans']))
    max_bans = max(max_bans, 1) # Pelo menos 1 para manter shape consistente

    bans = torch.zeros((B, 2, max_bans), dtype=torch.long)
    bans_mask = torch.zeros((B, 2, max_bans), dtype=torch.bool)
    bans_available = torch.zeros((B, 2, 1), dtype=torch.float32)

    for i, s in enumerate(batch):
        # Radiant bans
        rb = s.get('rad_bans', [])
        if rb:
            bans[i, 0, :len(rb)] = torch.tensor(rb, dtype=torch.long)
            bans_mask[i, 0, :len(rb)] = True
        bans_available[i, 0, 0] = 1.0 if s.get('rad_bans_available', False) else 0.0

        # Dire bans
        db = s.get('dire_bans', [])
        if db:
            bans[i, 1, :len(db)] = torch.tensor(db, dtype=torch.long)
            bans_mask[i, 1, :len(db)] = True
        bans_available[i, 1, 0] = 1.0 if s.get('dire_bans_available', False) else 0.0

    # 3. Ranks: [B, 2, 5]
    ranks = torch.zeros((B, 2, 5), dtype=torch.long)
    for i, s in enumerate(batch):
        ranks[i, 0] = torch.tensor(s['rad_ranks'], dtype=torch.long)
        ranks[i, 1] = torch.tensor(s['dire_ranks'], dtype=torch.long)

    # 4. MMR: [B, 2, 5, 1]
    mmr = torch.zeros((B, 2, 5, 1), dtype=torch.float32)
    for i, s in enumerate(batch):
        mmr[i, 0, :, 0] = torch.tensor(s['rad_mmrs'], dtype=torch.float32)
        mmr[i, 1, :, 0] = torch.tensor(s['dire_mmrs'], dtype=torch.float32)

    # 5. Skill masks: [B, 2, 5, 2] (col 0: rank_obs, col 1: mmr_obs)
    skill_masks = torch.zeros((B, 2, 5, 2), dtype=torch.float32)
    for i, s in enumerate(batch):
        # Radiant
        skill_masks[i, 0, :, 0] = torch.tensor(s['rad_rank_masks'], dtype=torch.float32)
        skill_masks[i, 0, :, 1] = torch.tensor(s['rad_mmr_masks'], dtype=torch.float32)
        # Dire
        skill_masks[i, 1, :, 0] = torch.tensor(s['dire_rank_masks'], dtype=torch.float32)
        skill_masks[i, 1, :, 1] = torch.tensor(s['dire_mmr_masks'], dtype=torch.float32)

    # 6. Patch: [B]
    patch = torch.tensor([s['patch_token'] for s in batch], dtype=torch.long)

    # 7. Match IDs: [B]
    match_ids = torch.tensor([(s.get('match_id') or 0) for s in batch], dtype=torch.long)

    # 8. Target: [B] (se presente)
    has_target = batch[0].get('target') is not None
    if has_target:
        target = torch.tensor([s['target'] for s in batch], dtype=torch.float32)
    else:
        target = torch.zeros(B, dtype=torch.float32)

    # 9. Composition Delta: [B, 11]
    has_comp = 'composition_delta' in batch[0] and batch[0]['composition_delta'] is not None
    if has_comp:
        composition_delta = torch.tensor([s['composition_delta'] for s in batch], dtype=torch.float32)
    else:
        composition_delta = torch.zeros((B, 11), dtype=torch.float32)

    return {
        'picks': picks,
        'bans': bans,
        'bans_mask': bans_mask,
        'bans_available': bans_available,
        'ranks': ranks,
        'mmr': mmr,
        'skill_masks': skill_masks,
        'patch': patch,
        'target': target,
        'match_ids': match_ids,
        'composition_delta': composition_delta
    }


def create_data_loaders(
    data_dir: str = 'data/processed_neural',
    batch_size: int = 32,
    num_workers: int = 0
) -> Tuple[DataLoader, DataLoader, DataLoader, NeuralFeatureEncoder]:
    """
    Cria os DataLoaders de Treino, Validação e Teste com o encoder correspondente.
    """
    train_path = os.path.join(data_dir, 'train_encoded.joblib')
    val_path = os.path.join(data_dir, 'val_encoded.joblib')
    test_path = os.path.join(data_dir, 'test_encoded.joblib')
    encoder_path = os.path.join(data_dir, 'encoder.json')

    if not all(os.path.exists(p) for p in [train_path, val_path, test_path, encoder_path]):
        raise FileNotFoundError(
            f"Artefatos neurais não encontrados em '{data_dir}'. Execute neural_features.py primeiro."
        )

    encoder = NeuralFeatureEncoder.load(encoder_path)

    train_ds = DotaNeuralDataset(train_path)
    val_ds = DotaNeuralDataset(val_path)
    test_ds = DotaNeuralDataset(test_path)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True, # Shuffle apenas dentro do split de treino
        collate_fn=collate_dota_batch,
        num_workers=num_workers
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_dota_batch,
        num_workers=num_workers
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_dota_batch,
        num_workers=num_workers
    )

    return train_loader, val_loader, test_loader, encoder

if __name__ == "__main__":
    from ..logger import setup_logging
    setup_logging()
    tl, vl, testl, enc = create_data_loaders()
    batch = next(iter(tl))
    print("--- Verificação do Batch Neural ---")
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            print(f"{k}: shape={list(v.shape)}, dtype={v.dtype}")
        else:
            print(f"{k}: {type(v)}")
