import os
import json
import logging
import argparse
import torch
from typing import List, Dict, Any, Optional, Union

from ..config import Config
from .neural_network import DotaWinPredictorNN
from ..features.neural_features import NeuralFeatureEncoder
from .neural_dataset import collate_dota_batch

logger = logging.getLogger(__name__)

class NeuralWinPredictor:
    """
    Motor de Inferência Neural desacoplado e autocontido:
    - Carrega modelo, encoder e temperatura calibrada de models/draft_win_nn/
    - Resolve nomes amigáveis em inglês para IDs
    - Processa picks, bans, patch e habilidade opcional
    - Retorna a probabilidade calibrada P(Radiant Win)
    """
    def __init__(self, model_dir: str = 'models/draft_win_nn', constants_file: str = 'data/heroes.json'):
        self.model_dir = model_dir
        self.constants_file = constants_file

        self.model = None
        self.encoder = None
        self.temperature = 1.0
        self.heroes_name_to_id = {}
        self.heroes_id_to_name = {}

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def load(self):
        weights_path = os.path.join(self.model_dir, 'weights.pt')
        config_path = os.path.join(self.model_dir, 'model_config.json')
        encoder_path = os.path.join(self.model_dir, 'encoder.json')
        calib_path = os.path.join(self.model_dir, 'calibration.json')

        if not os.path.exists(weights_path) or not os.path.exists(config_path):
            raise FileNotFoundError(
                f"Artefatos do modelo neural não encontrados em '{self.model_dir}'. Execute train_nn.py primeiro."
            )

        # 1. Carregar Configuração e Modelo
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)

        self.model = DotaWinPredictorNN(**config).to(self.device)
        self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
        self.model.eval()

        # 2. Carregar Encoder
        if os.path.exists(encoder_path):
            self.encoder = NeuralFeatureEncoder.load(encoder_path)
        else:
            fallback_enc = 'data/processed_neural/encoder.json'
            self.encoder = NeuralFeatureEncoder.load(fallback_enc)

        # 3. Carregar Calibração
        if os.path.exists(calib_path):
            with open(calib_path, 'r', encoding='utf-8') as f:
                cdata = json.load(f)
                self.temperature = float(cdata.get('temperature', 1.0))
        else:
            self.temperature = 1.0

        # 4. Carregar Constantes de Heróis
        if os.path.exists(self.constants_file):
            with open(self.constants_file, 'r', encoding='utf-8') as f:
                heroes_list = json.load(f)
                for h in heroes_list:
                    self.heroes_name_to_id[h['localized_name']] = h['id']
                    self.heroes_name_to_id[h['name']] = h['id']
                    self.heroes_id_to_name[h['id']] = h['localized_name']
        else:
            logger.warning(f"Arquivo de heróis '{self.constants_file}' não encontrado. Resolução limitada a IDs numéricos.")

        logger.info(f"Modelo neural carregado com sucesso (Temperatura={self.temperature:.4f})")

    def resolve_hero(self, identifier: Union[str, int]) -> int:
        """Resolve nome ou ID para hero_id int."""
        try:
            return int(identifier)
        except ValueError:
            pass

        if identifier in self.heroes_name_to_id:
            return self.heroes_name_to_id[identifier]

        for name, hid in self.heroes_name_to_id.items():
            if name.lower() == str(identifier).lower():
                return hid

        raise ValueError(f"Herói '{identifier}' não encontrado no catálogo oficial.")

    @torch.inference_mode()
    def predict(
        self,
        radiant_heroes: List[Union[str, int]],
        dire_heroes: List[Union[str, int]],
        radiant_bans: Optional[List[Union[str, int]]] = None,
        dire_bans: Optional[List[Union[str, int]]] = None,
        rad_players: Optional[List[Dict[str, Any]]] = None,
        dire_players: Optional[List[Dict[str, Any]]] = None,
        patch_id: Optional[int] = None
    ) -> float:
        if self.model is None:
            self.load()

        # Resolver heróis
        rad_picks = [self.resolve_hero(h) for h in radiant_heroes]
        dire_picks = [self.resolve_hero(h) for h in dire_heroes]

        rad_bans_resolved = [self.resolve_hero(h) for h in radiant_bans] if radiant_bans else []
        dire_bans_resolved = [self.resolve_hero(h) for h in dire_bans] if dire_bans else []

        # Montar match estruturado
        match_dict = {
            'rad_picks': rad_picks,
            'dire_picks': dire_picks,
            'rad_bans': rad_bans_resolved,
            'dire_bans': dire_bans_resolved,
            'rad_players': rad_players or [],
            'dire_players': dire_players or [],
            'patch_id': patch_id,
            'radiant_win': None
        }

        # Codificar
        encoded = self.encoder.transform_match(match_dict)

        # Montar lote com collate_dota_batch
        batch = collate_dota_batch([encoded])
        batch_dev = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        # Forward pass
        logit = self.model(batch_dev) # [1]
        scaled_logit = logit / max(self.temperature, 1e-4)
        prob = torch.sigmoid(scaled_logit).item()
        return float(prob)


def main():
    parser = argparse.ArgumentParser(description="Inferência com Rede Neural PyTorch para Dota 2")
    parser.add_argument("--rad", nargs='+', help="Heróis Radiant (5 nomes ou IDs)")
    parser.add_argument("--dire", nargs='+', help="Heróis Dire (5 nomes ou IDs)")
    parser.add_argument("--rad-bans", nargs='+', help="Bans Radiant (opcional)")
    parser.add_argument("--dire-bans", nargs='+', help="Bans Dire (opcional)")
    parser.add_argument("--patch", type=int, default=Config.TARGET_PATCH, help=f"ID do patch (default: {Config.TARGET_PATCH})")
    parser.add_argument("--json", type=str, help="Caminho para arquivo JSON de payload")
    args = parser.parse_args()

    predictor = NeuralWinPredictor()
    predictor.load()

    if args.json:
        with open(args.json, 'r', encoding='utf-8') as f:
            data = json.load(f)
        rad = data.get('radiant_heroes', [])
        dire = data.get('dire_heroes', [])
        rbans = data.get('radiant_bans', [])
        dbans = data.get('dire_bans', [])
        patch = data.get('patch_id', Config.TARGET_PATCH)
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

    prob = predictor.predict(
        radiant_heroes=rad,
        dire_heroes=dire,
        radiant_bans=rbans,
        dire_bans=dbans,
        rad_players=rad_p,
        dire_players=dire_p,
        patch_id=patch
    )

    print("=" * 45)
    print(f"Probabilidade Radiant Win: {prob * 100:.2f}%")
    if prob >= 0.5:
        print(f"Predição: Vitória da RADIANT ({prob * 100:.1f}%)")
    else:
        print(f"Predição: Vitória da DIRE ({(1.0 - prob) * 100:.1f}%)")
    print("=" * 45)

if __name__ == "__main__":
    main()
