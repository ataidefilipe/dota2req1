import json
import os
import logging
from dataclasses import dataclass
from typing import Sequence, Dict, Any, Tuple
import numpy as np

logger = logging.getLogger(__name__)

FEATURE_NAMES = (
    "attack_type_diversity",  # 4 * n_ranged * n_melee / 25
    "attribute_gain",         # mean(str_gain + agi_gain + int_gain)
    "move_speed",             # mean(move_speed)
    "initiator_count",        # sum(1 if Initiator in roles)
    "durable_count",          # sum(1 if Durable in roles)
    "carry_overlap",          # n_carry * (n_carry - 1) / 2
    "pusher_count",           # sum(1 if Pusher in roles)
    "disabler_count",         # sum(1 if Disabler in roles)
    "avg_max_cooldown",       # mean(max_cd per hero)
    "long_cd_count",          # sum(1 if max_cd >= 120s)
    "spammer_count",          # sum(1 if min_cd <= 4s and min_cd > 0)
    "bkb_pierce_count",       # sum(1 if hero has BKB-piercing disable/ultimate)
    "strong_dispel_count",    # sum(1 if hero has strong dispel ability)
    "hard_save_count",        # sum(1 if hero has hard save ability)
    "avg_synergy",            # mean intra-team pairwise synergy across 10 teammate pairs
    "max_synergy",            # max intra-team pairwise synergy (best combo)
    "min_synergy",            # min intra-team pairwise synergy (worst anti-synergy)
    "early_power",            # mean hero early winrate (<= 30 min)
    "late_power",             # mean hero late winrate (> 45 min)
    "scaling_factor",         # mean hero scaling slope (late_power - early_power)
    "matchup_advantage",      # mean pairwise head-to-head winrate differential vs opponent
)

TEAM_FEATURE_NAMES = FEATURE_NAMES[:20]

BKB_PIERCE_HEROES = {
    'axe', 'bane', 'batrider', 'beastmaster', 'bloodseeker',
    'rattletrap', 'doom_bringer', 'enigma', 'faceless_void',
    'legion_commander', 'magnataur', 'necrolyte', 'primal_beast',
    'pudge', 'shadow_demon', 'silencer', 'slardar', 'spirit_breaker',
    'treant', 'tusk', 'vengefulspirit', 'viper', 'winter_wyvern',
    'marci', 'muerta', 'huskar'
}

STRONG_DISPEL_HEROES = {
    'abaddon', 'legion_commander', 'oracle', 'slark',
    'tidehunter', 'ursa', 'omniknight', 'life_stealer'
}

HARD_SAVE_HEROES = {
    'dazzle', 'oracle', 'shadow_demon', 'obsidian_destroyer',
    'winter_wyvern', 'pugna', 'vengefulspirit'
}

@dataclass(frozen=True)
class HeroProfile:
    hero_id: int
    name: str
    is_ranged: float
    total_attr_gain: float
    move_speed: float
    is_initiator: float
    is_durable: float
    is_carry: float
    is_pusher: float
    is_disabler: float
    max_cd: float
    has_long_cd: float
    has_spam_cd: float
    has_bkb_pierce: float
    has_strong_dispel: float
    has_hard_save: float
    early_power: float
    late_power: float
    scaling_factor: float


class CompositionFeatureExtractor:
    """
    Extrator estático, determinístico e auditável de features de composição de equipe
    para o Dota 2 Win Predictor (Etapa 1 + Etapa 2 + Etapa 3).
    
    Lê os metadados de heróis, magias, matriz de matchups, sinergias e durações de data/constants
    e pré-computa perfis fixos por herói e matrizes anti-simétricas / simétricas.
    Permite extrair vetores por time (20 features) e vetores diferenciais (21 features):
        Delta = [V(Radiant) - V(Dire); MatchupAdvantage(Radiant vs Dire)]
    garantindo que Delta(Dire, Radiant) == -Delta(Radiant, Dire).
    """
    schema_version: str = "1.3.0"
    feature_names: Tuple[str, ...] = FEATURE_NAMES
    team_feature_names: Tuple[str, ...] = TEAM_FEATURE_NAMES

    def __init__(self, constants_dir: str = 'data/constants'):
        self.constants_dir = constants_dir
        self.profiles: Dict[int, HeroProfile] = {}
        self.matchup_matrix = np.zeros((250, 250), dtype=np.float32)
        self.synergy_matrix = np.zeros((250, 250), dtype=np.float32)
        self._load_and_build_profiles()
        self._load_and_build_matchups()
        self._load_and_build_synergies()

    def _load_and_build_profiles(self):
        heroes_path = os.path.join(self.constants_dir, 'heroes.json')
        if not os.path.exists(heroes_path):
            heroes_path = 'data/heroes.json'

        if not os.path.exists(heroes_path):
            raise FileNotFoundError(f"Arquivo de constantes de heróis não encontrado em '{heroes_path}'.")

        with open(heroes_path, 'r', encoding='utf-8') as f:
            heroes_data = json.load(f)

        if isinstance(heroes_data, dict):
            heroes_list = list(heroes_data.values())
        else:
            heroes_list = heroes_data

        hab_path = os.path.join(self.constants_dir, 'hero_abilities.json')
        abs_path = os.path.join(self.constants_dir, 'abilities.json')

        hero_abilities = {}
        if os.path.exists(hab_path):
            with open(hab_path, 'r', encoding='utf-8') as f:
                hero_abilities = json.load(f)

        abilities_catalog = {}
        if os.path.exists(abs_path):
            with open(abs_path, 'r', encoding='utf-8') as f:
                abilities_catalog = json.load(f)

        durations_path = os.path.join(self.constants_dir, 'durations.json')
        durations_data = {}
        if os.path.exists(durations_path):
            with open(durations_path, 'r', encoding='utf-8') as f:
                durations_data = json.load(f)

        for h in heroes_list:
            hid = int(h['id'])
            hname = h.get('name', '')
            short_name = hname.replace('npc_dota_hero_', '')
            attack_type = h.get('attack_type', 'Melee')
            is_ranged = 1.0 if attack_type == 'Ranged' else 0.0

            str_gain = float(h.get('str_gain', 0.0) or 0.0)
            agi_gain = float(h.get('agi_gain', 0.0) or 0.0)
            int_gain = float(h.get('int_gain', 0.0) or 0.0)
            total_attr_gain = str_gain + agi_gain + int_gain

            move_speed = float(h.get('move_speed', 300.0) or 300.0)
            roles = set(h.get('roles', []))

            # Extrair cooldowns das magias
            h_abils_info = hero_abilities.get(hname, {})
            raw_abils = []
            for a in h_abils_info.get('abilities', []):
                if isinstance(a, list):
                    raw_abils.extend(a)
                elif isinstance(a, str):
                    raw_abils.append(a)

            cds = []
            for aname in raw_abils:
                if aname in abilities_catalog:
                    ad = abilities_catalog[aname]
                    beh = ad.get('behavior')
                    if beh == 'Hidden' or (isinstance(beh, list) and 'Hidden' in beh):
                        continue
                    cd = ad.get('cd')
                    if cd is not None:
                        if isinstance(cd, list):
                            vals = [float(x) for x in cd if str(x).replace('.', '', 1).isdigit()]
                            if vals:
                                cds.append(vals[0])
                        elif isinstance(cd, (int, float)):
                            cds.append(float(cd))
                        elif isinstance(cd, str) and cd.replace('.', '', 1).isdigit():
                            cds.append(float(cd))

            max_cd = float(max(cds)) if cds else 0.0
            positive_cds = [c for c in cds if c > 0]
            min_cd = float(min(positive_cds)) if positive_cds else 0.0

            has_long_cd = 1.0 if max_cd >= 120.0 else 0.0
            has_spam_cd = 1.0 if (0.0 < min_cd <= 4.0) else 0.0

            has_bkb_pierce = 1.0 if short_name in BKB_PIERCE_HEROES else 0.0
            has_strong_dispel = 1.0 if short_name in STRONG_DISPEL_HEROES else 0.0
            has_hard_save = 1.0 if short_name in HARD_SAVE_HEROES else 0.0

            # Extrair curvas de duração e scaling (Etapa 3)
            h_durations = durations_data.get(str(hid), [])
            g_early, w_early = 0, 0
            g_late, w_late = 0, 0
            for b in h_durations:
                sec = b.get('duration_bin', 0)
                g = b.get('games_played', 0)
                w = b.get('wins', 0)
                if sec <= 1800:
                    g_early += g
                    w_early += w
                elif sec > 2700:
                    g_late += g
                    w_late += w

            C_dur = 20.0
            early_power = float((w_early + C_dur * 0.5) / (g_early + C_dur)) if (g_early + C_dur) > 0 else 0.50
            late_power = float((w_late + C_dur * 0.5) / (g_late + C_dur)) if (g_late + C_dur) > 0 else 0.50
            scaling_factor = float(late_power - early_power)

            self.profiles[hid] = HeroProfile(
                hero_id=hid,
                name=hname,
                is_ranged=is_ranged,
                total_attr_gain=total_attr_gain,
                move_speed=move_speed,
                is_initiator=1.0 if 'Initiator' in roles else 0.0,
                is_durable=1.0 if 'Durable' in roles else 0.0,
                is_carry=1.0 if 'Carry' in roles else 0.0,
                is_pusher=1.0 if 'Pusher' in roles else 0.0,
                is_disabler=1.0 if 'Disabler' in roles else 0.0,
                max_cd=max_cd,
                has_long_cd=has_long_cd,
                has_spam_cd=has_spam_cd,
                has_bkb_pierce=has_bkb_pierce,
                has_strong_dispel=has_strong_dispel,
                has_hard_save=has_hard_save,
                early_power=early_power,
                late_power=late_power,
                scaling_factor=scaling_factor,
            )

        logger.info(f"CompositionFeatureExtractor inicializado com {len(self.profiles)} perfis de heróis.")

    def _load_and_build_matchups(self):
        matchups_path = os.path.join(self.constants_dir, 'matchups.json')
        if not os.path.exists(matchups_path):
            logger.warning(f"Arquivo de matchups não encontrado em '{matchups_path}'. Matchup advantage será neutro (0).")
            return

        with open(matchups_path, 'r', encoding='utf-8') as f:
            matchups_data = json.load(f)

        C = 20.0
        prior_wr = 0.50
        for r_str, opps in matchups_data.items():
            try:
                r_id = int(r_str)
            except ValueError:
                continue
            if r_id >= 250:
                continue

            for d_str, stat in opps.items():
                try:
                    d_id = int(d_str)
                except ValueError:
                    continue
                if d_id >= 250:
                    continue

                games_r = float(stat.get('games_played', 0))
                wins_r = float(stat.get('wins', 0))
                wr_r = (wins_r + C * prior_wr) / (games_r + C) if (games_r + C) > 0 else prior_wr

                opp_stat = matchups_data.get(str(d_id), {}).get(str(r_id), {})
                games_d = float(opp_stat.get('games_played', games_r))
                wins_d = float(opp_stat.get('wins', max(0.0, games_r - wins_r)))
                wr_d = (wins_d + C * prior_wr) / (games_d + C) if (games_d + C) > 0 else prior_wr

                net_h2h = 0.5 * (wr_r - wr_d)
                self.matchup_matrix[r_id, d_id] = net_h2h

        logger.info("Matriz de matchups pairwise pré-computada com priors Bayesianos (C=20).")

    def _load_and_build_synergies(self):
        synergies_path = os.path.join(self.constants_dir, 'synergies.json')
        if not os.path.exists(synergies_path):
            logger.warning(f"Arquivo de sinergias não encontrado em '{synergies_path}'. Sinergias serão neutras (0).")
            return

        with open(synergies_path, 'r', encoding='utf-8') as f:
            synergies_data = json.load(f)

        for h1_str, opps in synergies_data.items():
            try:
                h1_id = int(h1_str)
            except ValueError:
                continue
            if h1_id >= 250:
                continue

            for h2_str, stat in opps.items():
                try:
                    h2_id = int(h2_str)
                except ValueError:
                    continue
                if h2_id >= 250:
                    continue

                syn_val = float(stat.get('synergy', 0.0))
                self.synergy_matrix[h1_id, h2_id] = syn_val
                self.synergy_matrix[h2_id, h1_id] = syn_val

        logger.info("Matriz de sinergias intra-equipe pré-computada com sucesso.")

    def calc_matchup_advantage(self, radiant_ids: Sequence[int], dire_ids: Sequence[int]) -> float:
        """
        Calcula a vantagem média de confronto direto (head-to-head) entre os 5 heróis da Radiant
        e os 5 heróis da Dire usando a matriz bayesiana pré-computada.
        Garante rigorosa antissimetria:
            calc_matchup_advantage(dire, rad) == -calc_matchup_advantage(rad, dire)
        """
        if len(radiant_ids) != 5 or len(dire_ids) != 5:
            return 0.0
        rad_arr = [int(h) for h in radiant_ids if int(h) < 250]
        dire_arr = [int(h) for h in dire_ids if int(h) < 250]
        if len(rad_arr) != 5 or len(dire_arr) != 5:
            return 0.0
        submatrix = self.matchup_matrix[np.ix_(rad_arr, dire_arr)]
        return float(np.mean(submatrix))

    def team_vector(self, hero_ids: Sequence[int]) -> np.ndarray:
        """
        Calcula o vetor de composição absoluta V(T) in R^20 para um time de 5 heróis.
        Ordem dos heróis é irrelevante (invariante a permutação).
        """
        if len(hero_ids) != 5:
            raise ValueError(f"Esperado exatamente 5 heróis por equipe, recebido {len(hero_ids)}: {hero_ids}")

        profiles = [self.profiles.get(int(hid)) for hid in hero_ids]
        valid_profiles = [p for p in profiles if p is not None]

        if not valid_profiles:
            return np.zeros(len(self.team_feature_names), dtype=np.float32)

        n_ranged = sum(p.is_ranged for p in valid_profiles)
        n_melee = len(valid_profiles) - n_ranged
        attack_type_diversity = float((4.0 * n_ranged * n_melee) / 25.0)

        attribute_gain = float(np.mean([p.total_attr_gain for p in valid_profiles]))
        move_speed = float(np.mean([p.move_speed for p in valid_profiles]))

        initiator_count = float(sum(p.is_initiator for p in valid_profiles))
        durable_count = float(sum(p.is_durable for p in valid_profiles))

        n_carry = sum(p.is_carry for p in valid_profiles)
        carry_overlap = float((n_carry * (n_carry - 1.0)) / 2.0)

        pusher_count = float(sum(p.is_pusher for p in valid_profiles))
        disabler_count = float(sum(p.is_disabler for p in valid_profiles))

        avg_max_cooldown = float(np.mean([p.max_cd for p in valid_profiles]))
        long_cd_count = float(sum(p.has_long_cd for p in valid_profiles))
        spammer_count = float(sum(p.has_spam_cd for p in valid_profiles))

        bkb_pierce_count = float(sum(p.has_bkb_pierce for p in valid_profiles))
        strong_dispel_count = float(sum(p.has_strong_dispel for p in valid_profiles))
        hard_save_count = float(sum(p.has_hard_save for p in valid_profiles))

        # Sinergias intra-equipe (10 pares C(5, 2))
        pair_synergies = []
        for i in range(len(hero_ids)):
            hid_i = int(hero_ids[i])
            for j in range(i + 1, len(hero_ids)):
                hid_j = int(hero_ids[j])
                if hid_i < 250 and hid_j < 250:
                    pair_synergies.append(float(self.synergy_matrix[hid_i, hid_j]))
                else:
                    pair_synergies.append(0.0)

        avg_synergy = float(np.mean(pair_synergies)) if pair_synergies else 0.0
        max_synergy = float(np.max(pair_synergies)) if pair_synergies else 0.0
        min_synergy = float(np.min(pair_synergies)) if pair_synergies else 0.0

        # Timing e Curvas de Poder (Etapa 3)
        early_power = float(np.mean([p.early_power for p in valid_profiles]))
        late_power = float(np.mean([p.late_power for p in valid_profiles]))
        scaling_factor = float(np.mean([p.scaling_factor for p in valid_profiles]))

        return np.array([
            attack_type_diversity,
            attribute_gain,
            move_speed,
            initiator_count,
            durable_count,
            carry_overlap,
            pusher_count,
            disabler_count,
            avg_max_cooldown,
            long_cd_count,
            spammer_count,
            bkb_pierce_count,
            strong_dispel_count,
            hard_save_count,
            avg_synergy,
            max_synergy,
            min_synergy,
            early_power,
            late_power,
            scaling_factor
        ], dtype=np.float32)

    def transform(self, radiant_ids: Sequence[int], dire_ids: Sequence[int]) -> np.ndarray:
        """
        Calcula o vetor diferencial Delta in R^21:
            Delta[:20] = V(Radiant) - V(Dire)  (atributos, cooldowns, mecânicas, sinergias e timing)
            Delta[20]  = MatchupAdvantage(Radiant vs Dire)
            
        Propriedade garantida: Delta(Dire, Radiant) == -Delta(Radiant, Dire).
        """
        v_rad = self.team_vector(radiant_ids)
        v_dire = self.team_vector(dire_ids)
        diff_team = v_rad - v_dire
        matchup_adv = self.calc_matchup_advantage(radiant_ids, dire_ids)
        return np.append(diff_team, np.float32(matchup_adv))


class CompositionScaler:
    """
    Escalonador RMS que preserva a antissimetria perfeita de inversão de lados.
    
    Calcula por coluna s_j = sqrt(mean(Delta_j^2)) sem subtrair média, de modo que:
        Delta / s == -( -Delta / s )
    """
    def __init__(self):
        self.scale_: np.ndarray = None
        self.fitted: bool = False

    def fit(self, X_deltas: np.ndarray) -> "CompositionScaler":
        if X_deltas.ndim == 1:
            X_deltas = X_deltas.reshape(1, -1)

        # RMS por coluna: sqrt(mean(x^2))
        rms = np.sqrt(np.mean(X_deltas ** 2, axis=0))
        # Evitar divisão por zero caso a feature seja constante 0
        rms[rms < 1e-6] = 1.0

        self.scale_ = rms.astype(np.float32)
        self.fitted = True
        return self

    def transform(self, X_deltas: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("CompositionScaler precisa ser ajustado com fit() antes de transform().")
        return (X_deltas / self.scale_).astype(np.float32)

    def fit_transform(self, X_deltas: np.ndarray) -> np.ndarray:
        return self.fit(X_deltas).transform(X_deltas)

    def save(self, filepath: str):
        if not self.fitted:
            raise RuntimeError("Não é possível salvar scaler não ajustado.")
        data = {
            "scale": self.scale_.tolist(),
            "feature_names": list(FEATURE_NAMES)
        }
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, filepath: str) -> "CompositionScaler":
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        scaler = cls()
        scaler.scale_ = np.array(data["scale"], dtype=np.float32)
        scaler.fitted = True
        return scaler
