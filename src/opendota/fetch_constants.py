import argparse
import json
import os
import logging
from .client import OpenDotaClient

logger = logging.getLogger(__name__)

def fetch_all_constants(output_dir: str = 'data'):
    """
    Baixa e salva todas as constantes estruturadas do Dota 2 via OpenDota:
    1. heroes.json: Atributos base, ganhos por nível, HP, Mana, Armadura, Alcance, Funções e Velocidade.
    2. hero_abilities.json: Mapeamento de cada herói para suas habilidades, talentos e facetas.
    3. abilities.json: Detalhes das magias (dano, custo de mana, cooldown, pierce BKB, tipo de dano, duração).
    """
    client = OpenDotaClient()
    os.makedirs(output_dir, exist_ok=True)
    constants_dir = os.path.join(output_dir, 'constants')
    os.makedirs(constants_dir, exist_ok=True)

    # 1. Hero Stats & Attributes
    logger.info("Baixando atributos e estatísticas dos heróis (/heroStats)...")
    hero_stats = client.get("/heroStats")
    if hero_stats:
        # Salva o arquivo principal data/heroes.json (compatível com predict e dashboard)
        heroes_path = os.path.join(output_dir, 'heroes.json')
        with open(heroes_path, 'w', encoding='utf-8') as f:
            json.dump(hero_stats, f, indent=2, ensure_ascii=False)
        logger.info(f"[+] Salvos {len(hero_stats)} heróis com atributos completos em {heroes_path}")

        # Salva também cópia em data/constants/heroes.json
        constants_heroes_path = os.path.join(constants_dir, 'heroes.json')
        with open(constants_heroes_path, 'w', encoding='utf-8') as f:
            json.dump(hero_stats, f, indent=2, ensure_ascii=False)
    else:
        logger.error("Falha ao baixar /heroStats.")

    # 2. Hero Abilities Mapping
    logger.info("Baixando mapeamento de habilidades dos heróis (/constants/hero_abilities)...")
    hero_abilities = client.get("/constants/hero_abilities")
    if hero_abilities:
        hab_path = os.path.join(constants_dir, 'hero_abilities.json')
        with open(hab_path, 'w', encoding='utf-8') as f:
            json.dump(hero_abilities, f, indent=2, ensure_ascii=False)
        logger.info(f"[+] Salvo mapeamento de habilidades para {len(hero_abilities)} heróis em {hab_path}")
    else:
        logger.error("Falha ao baixar /constants/hero_abilities.")

    # 3. Ability Details (Magias, dano, custo, cooldown, efeitos)
    logger.info("Baixando catálogo detalhado de magias e habilidades (/constants/abilities)...")
    abilities = client.get("/constants/abilities")
    if abilities:
        ab_path = os.path.join(constants_dir, 'abilities.json')
        with open(ab_path, 'w', encoding='utf-8') as f:
            json.dump(abilities, f, indent=2, ensure_ascii=False)
        logger.info(f"[+] Salvas {len(abilities)} habilidades e magias com dano/custo/duração em {ab_path}")
    else:
        logger.error("Falha ao baixar /constants/abilities.")

if __name__ == "__main__":
    from ..logger import setup_logging
    setup_logging()
    fetch_all_constants()
