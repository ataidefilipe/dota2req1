import argparse
import json
import os
import logging
from .client import OpenDotaClient
from ..db import repo

logger = logging.getLogger(__name__)

def fetch_all_constants(output_dir: str = 'data'):
    """
    Baixa e salva todas as constantes estruturadas do Dota 2 via OpenDota:
    1. heroes.json: Atributos base, ganhos por nível, HP, Mana, Armadura, Alcance, Funções e Velocidade.
    2. hero_abilities.json: Mapeamento de cada herói para suas habilidades, talentos e facetas.
    3. abilities.json: Detalhes das magias (dano, custo de mana, cooldown, pierce BKB, tipo de dano, duração).
    4. patch.json: Metadados históricos de versões do jogo (nomes, timestamps, patch_ids).
    5. Persistência em SQLite nas tabelas hero_ref e patch_ref.
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
            
        try:
            repo.upsert_heroes(hero_stats)
            logger.info(f"[+] Populados {len(hero_stats)} heróis na tabela hero_ref do SQLite.")
        except Exception as e:
            logger.warning(f"Não foi possível salvar heróis no SQLite: {e}")
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

    # 4. Patch Metadata
    logger.info("Baixando referências de patches (/constants/patch)...")
    patches = client.get("/constants/patch")
    if patches:
        patch_path = os.path.join(constants_dir, 'patch.json')
        with open(patch_path, 'w', encoding='utf-8') as f:
            json.dump(patches, f, indent=2, ensure_ascii=False)
        logger.info(f"[+] Salvos {len(patches)} patches em {patch_path}")
        
        try:
            repo.upsert_patches(patches)
            logger.info(f"[+] Populados {len(patches)} patches na tabela patch_ref do SQLite.")
        except Exception as e:
            logger.warning(f"Não foi possível salvar patches no SQLite: {e}")
    else:
        logger.error("Falha ao baixar /constants/patch.")

    # 5. Hero Matchups Matrix
    logger.info("Baixando matriz global de confrontos (/heroes/{id}/matchups)...")
    matchups_matrix = {}
    if hero_stats:
        for i, h in enumerate(hero_stats):
            hid = h.get('id')
            if not hid:
                continue
            m = client.get(f"/heroes/{hid}/matchups")
            if m:
                matchups_matrix[str(hid)] = {
                    str(x['hero_id']): {
                        'games_played': x.get('games_played', 0),
                        'wins': x.get('wins', 0)
                    }
                    for x in m if 'hero_id' in x
                }
            if (i + 1) % 25 == 0 or (i + 1) == len(hero_stats):
                logger.info(f"Progresso de confrontos: {i + 1}/{len(hero_stats)} heróis processados...")

        matchups_path = os.path.join(constants_dir, 'matchups.json')
        with open(matchups_path, 'w', encoding='utf-8') as f:
            json.dump(matchups_matrix, f, indent=2, ensure_ascii=False)
        logger.info(f"[+] Salva matriz global de confrontos para {len(matchups_matrix)} heróis em {matchups_path}")

    # 6. Hero Durations Curves
    durations_path = os.path.join(constants_dir, 'durations.json')
    if not os.path.exists(durations_path):
        logger.info("Baixando curvas de duração de partida (/heroes/{id}/durations)...")
        durations_matrix = {}
        if hero_stats:
            for i, h in enumerate(hero_stats):
                hid = h.get('id')
                if not hid:
                    continue
                d = client.get(f"/heroes/{hid}/durations")
                if d:
                    durations_matrix[str(hid)] = d
                if (i + 1) % 25 == 0 or (i + 1) == len(hero_stats):
                    logger.info(f"Progresso de durações: {i + 1}/{len(hero_stats)} heróis processados...")

            with open(durations_path, 'w', encoding='utf-8') as f:
                json.dump(durations_matrix, f, indent=2, ensure_ascii=False)
            logger.info(f"[+] Salvas curvas de duração para {len(durations_matrix)} heróis em {durations_path}")


if __name__ == "__main__":
    from ..logger import setup_logging
    setup_logging()
    fetch_all_constants()

