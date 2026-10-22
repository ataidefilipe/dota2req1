import sys
import os

# Garantir que a raiz do projeto esteja no sys.path e o cwd seja a raiz
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

import streamlit as st
import pandas as pd
import sqlite3
import json
import numpy as np
from src.config import Config
from src.features.composition_features import CompositionFeatureExtractor
from src.betting.engine import evaluate_betting_opportunity, simulate_bankroll_growth

# Configuração da página
st.set_page_config(
    page_title="Aegis Draft • Dota 2 Win Predictor",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Estilização visual temática (Dota 2 Radiant vs Dire)
st.markdown("""
<style>
    .radiant-box {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.08) 0%, rgba(5, 150, 105, 0.15) 100%);
        border: 2px solid #10b981;
        border-radius: 12px;
        padding: 16px;
        margin-bottom: 12px;
    }
    .dire-box {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.08) 0%, rgba(185, 28, 28, 0.15) 100%);
        border: 2px solid #ef4444;
        border-radius: 12px;
        padding: 16px;
        margin-bottom: 12px;
    }
    .metric-card {
        background-color: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 10px;
        padding: 14px;
        text-align: center;
    }
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        height: 48px;
        font-weight: 700;
        font-size: 16px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    matches = pd.DataFrame()
    events = pd.DataFrame()
    if os.path.exists('data/opendota.sqlite'):
        try:
            conn = sqlite3.connect('data/opendota.sqlite')
            matches = pd.read_sql("SELECT * FROM matches", conn)
            events = pd.read_sql("SELECT * FROM draft_events", conn)
            conn.close()
        except Exception:
            pass

    heroes_map = {}
    heroes_name_to_id = {}
    heroes_list = []
    if os.path.exists('data/constants/heroes.json'):
        with open('data/constants/heroes.json', 'r', encoding='utf-8') as f:
            raw = json.load(f)
            heroes = raw.values() if isinstance(raw, dict) else raw
            for h in heroes:
                hid = int(h['id'])
                name = h['localized_name']
                heroes_map[hid] = name
                heroes_name_to_id[name] = hid
                heroes_list.append(name)
    heroes_list.sort()

    return matches, events, heroes_map, heroes_name_to_id, heroes_list


@st.cache_resource
def load_extractor():
    return CompositionFeatureExtractor(constants_dir='data/constants')


@st.cache_resource
def load_predictor():
    from src.model.predict_nn import NeuralWinPredictor
    return NeuralWinPredictor()


@st.cache_data
def load_meta_insights(heroes_map):
    scaling_data = []
    if os.path.exists('data/constants/durations.json'):
        with open('data/constants/durations.json', 'r', encoding='utf-8') as f:
            dur = json.load(f)
            for hid_str, bins in dur.items():
                hid = int(hid_str)
                hname = heroes_map.get(hid, f"Hero {hid}")
                g_early, w_early = 0, 0
                g_late, w_late = 0, 0
                for b in bins:
                    sec = b.get('duration_bin', 0)
                    g = b.get('games_played', 0)
                    w = b.get('wins', 0)
                    if sec <= 1800:
                        g_early += g
                        w_early += w
                    elif sec > 2700:
                        g_late += g
                        w_late += w
                C = 20.0
                pe = (w_early + C * 0.5) / (g_early + C) if (g_early + C) > 0 else 0.5
                pl = (w_late + C * 0.5) / (g_late + C) if (g_late + C) > 0 else 0.5
                scaling_data.append({
                    'hero': hname,
                    'early_wr': pe * 100,
                    'late_wr': pl * 100,
                    'scaling': (pl - pe) * 100,
                    'games': g_early + g_late
                })

    synergies_data = []
    if os.path.exists('data/constants/synergies.json'):
        with open('data/constants/synergies.json', 'r', encoding='utf-8') as f:
            syn = json.load(f)
            for h1_str, opps in syn.items():
                h1 = int(h1_str)
                for h2_str, stat in opps.items():
                    h2 = int(h2_str)
                    if h1 < h2 and stat.get('games', 0) >= 150:
                        synergies_data.append({
                            'combo': f"{heroes_map.get(h1, str(h1))} + {heroes_map.get(h2, str(h2))}",
                            'games': stat.get('games', 0),
                            'winrate': stat.get('wr', 0.5) * 100,
                            'synergy': stat.get('synergy', 0.0) * 100
                        })

    return scaling_data, synergies_data


def main():
    # 1. Carregamento de Recursos
    try:
        matches, events, heroes_map, heroes_name_to_id, sorted_hero_names = load_data()
        extractor = load_extractor()
        pred_nn = load_predictor()
        scaling_data, synergies_data = load_meta_insights(heroes_map)
    except Exception as e:
        st.error(f"Erro ao inicializar recursos do dashboard: {e}")
        return

    # Presets para experimentação rápida
    PRESETS = {
        "— Selecionar Preset Rápido —": ([], []),
        "Late-Game Scaling vs Zoo Tempo Push": (
            ["Spectre", "Anti-Mage", "Faceless Void", "Dazzle", "Crystal Maiden"],
            ["Chen", "Broodmother", "Lycan", "Beastmaster", "Nature's Prophet"]
        ),
        "Combos de Teamfight vs Pick-off Gank": (
            ["Magnus", "Sven", "Enigma", "Witch Doctor", "Jakiro"],
            ["Batrider", "Doom", "Slardar", "Bounty Hunter", "Spirit Breaker"]
        ),
        "Draft Competitivo Equilibrado": (
            ["Phantom Assassin", "Bloodseeker", "Ogre Magi", "Rubick", "Centaur Warrunner"],
            ["Axe", "Lion", "Sniper", "Viper", "Tidehunter"]
        )
    }

    # Inicializar estado do rascunho de draft se necessário
    if "rad_picks" not in st.session_state:
        st.session_state["rad_picks"] = ["Anti-Mage", "Crystal Maiden", "Axe", "Lion", "Zeus"]
    if "dire_picks" not in st.session_state:
        st.session_state["dire_picks"] = ["Pudge", "Sniper", "Invoker", "Techies", "Chen"]

    # 2. Header & Branding
    st.title("🛡️ Aegis Draft • Simulador e Oráculo de Vitórias")
    st.caption("Motor Preditivo para Dota 2 • Patch 60 (v7.41) • Rede Neural Deep Sets & Cross-Attention com Calibração")

    # 3. Seção Hero: O Simulador de Draft Interativo
    preset_choice = st.selectbox(
        "⚡ Escolha um rascunho de exemplo para testar imediatamente ou monte o seu abaixo:",
        list(PRESETS.keys())
    )
    if preset_choice != "— Selecionar Preset Rápido —":
        p_rad, p_dire = PRESETS[preset_choice]
        st.session_state["rad_picks"] = p_rad
        st.session_state["dire_picks"] = p_dire

    col_rad, col_dire = st.columns(2)

    # Lado Radiant (Verde)
    with col_rad:
        st.markdown("""
        <div class="radiant-box">
            <h3 style="color: #10b981; margin: 0 0 8px 0;">🟢 Faction: RADIANT</h3>
            <span style="font-size: 13px; opacity: 0.85;">Selecione os 5 heróis da equipe da Luz</span>
        </div>
        """, unsafe_allow_html=True)

        # Prevenção de duplicidade: heróis já no Dire não aparecem aqui
        avail_rad = [h for h in sorted_hero_names if h not in st.session_state.get("dire_picks", [])]
        selected_rad = st.multiselect(
            "Heróis da Radiant (máx. 5):",
            avail_rad,
            default=[h for h in st.session_state["rad_picks"] if h in avail_rad][:5],
            max_selections=5,
            key="multiselect_rad"
        )
        st.session_state["rad_picks"] = selected_rad

    # Lado Dire (Vermelho)
    with col_dire:
        st.markdown("""
        <div class="dire-box">
            <h3 style="color: #ef4444; margin: 0 0 8px 0;">🔴 Faction: DIRE</h3>
            <span style="font-size: 13px; opacity: 0.85;">Selecione os 5 heróis da equipe das Trevas</span>
        </div>
        """, unsafe_allow_html=True)

        # Prevenção de duplicidade: heróis já no Radiant não aparecem aqui
        avail_dire = [h for h in sorted_hero_names if h not in st.session_state.get("rad_picks", [])]
        selected_dire = st.multiselect(
            "Heróis da Dire (máx. 5):",
            avail_dire,
            default=[h for h in st.session_state["dire_picks"] if h in avail_dire][:5],
            max_selections=5,
            key="multiselect_dire"
        )
        st.session_state["dire_picks"] = selected_dire

    # Validação e Botão de Ação
    is_valid_draft = (len(selected_rad) == 5 and len(selected_dire) == 5)

    btn_predict = st.button("🔮 Calcular Probabilidade de Vitória & Diagnóstico de Draft", type="primary")

    # Executar previsão se botão clicado ou se já temos 5v5 válidos
    if btn_predict or is_valid_draft:
        if not is_valid_draft:
            st.warning("⚠️ Selecione exatamente 5 heróis em cada equipe para calcular a probabilidade.")
        else:
            rad_ids = [heroes_name_to_id[h] for h in selected_rad]
            dire_ids = [heroes_name_to_id[h] for h in selected_dire]

            # 1. Inferência Neural Calibrada
            prob_rad = pred_nn.predict(radiant_heroes=selected_rad, dire_heroes=selected_dire, patch_id=Config.TARGET_PATCH)
            rad_pct = prob_rad * 100.0
            dire_pct = (1.0 - prob_rad) * 100.0

            # 2. Barra de Duelo Estilizada
            st.markdown("---")
            st.markdown("### 🏆 Placar de Probabilidade de Vitória Calibrada")

            # Barra visual de confronto
            dual_bar_html = f"""
            <div style="display: flex; height: 38px; border-radius: 8px; overflow: hidden; font-weight: 800; font-size: 15px; box-shadow: 0 4px 12px rgba(0,0,0,0.3); margin-bottom: 12px;">
                <div style="width: {rad_pct}%; background: linear-gradient(90deg, #059669, #10b981); color: white; display: flex; align-items: center; justify-content: center; text-shadow: 0 1px 2px rgba(0,0,0,0.5);">
                    RADIANT {rad_pct:.1f}%
                </div>
                <div style="width: {dire_pct}%; background: linear-gradient(90deg, #ef4444, #b91c1c); color: white; display: flex; align-items: center; justify-content: center; text-shadow: 0 1px 2px rgba(0,0,0,0.5);">
                    DIRE {dire_pct:.1f}%
                </div>
            </div>
            """
            st.markdown(dual_bar_html, unsafe_allow_html=True)

            # Veredito
            diff = abs(rad_pct - 50.0)
            if diff < 2.5:
                st.info("⚖️ **Confronto Altamente Equilibrado:** As duas equipes apresentam chances técnicas equivalentes. O resultado será decidido puramente na execução mecânica dentro de jogo.")
            elif rad_pct >= 50.0:
                st.success(f"🏆 **Favorito: RADIANT** com **{rad_pct:.1f}%** de probabilidade estimada (+{rad_pct - 50:.1f}% de vantagem técnica de draft).")
            else:
                st.error(f"🏆 **Favorito: DIRE** com **{dire_pct:.1f}%** de probabilidade estimada (+{dire_pct - 50:.1f}% de vantagem técnica de draft).")

            # 3. Painel de Explicabilidade nos 4 Pilares
            st.markdown("### 🔬 Diagnóstico Estratégico do Draft (Os 4 Pilares)")

            v_rad = extractor.team_vector(rad_ids)
            v_dire = extractor.team_vector(dire_ids)
            matchup_adv = extractor.calc_matchup_advantage(rad_ids, dire_ids)

            card_col1, card_col2 = st.columns(2)

            # Pilar 1: Confrontos Diretos (Counters 1v1)
            with card_col1:
                with st.container(border=True):
                    st.markdown("#### 1. ⚔️ Confrontos Diretos (Counters)")
                    adv_str = f"+{matchup_adv*100:.2f}% (Radiant)" if matchup_adv >= 0 else f"+{abs(matchup_adv)*100:.2f}% (Dire)"
                    st.markdown(f"**Balanço Geral de Confronto:** `{adv_str}`")

                    counters = []
                    for r in rad_ids:
                        for d in dire_ids:
                            edge = extractor.matchup_matrix[r, d]
                            counters.append((heroes_map[r], heroes_map[d], edge))
                    counters.sort(key=lambda x: abs(x[2]), reverse=True)

                    st.markdown("**Top 3 Counters Individuais Decisivos:**")
                    for r_name, d_name, edge in counters[:3]:
                        if edge > 0:
                            st.write(f"- 🟢 **{r_name}** anula **{d_name}** (`+{edge*100:.1f}%` vantagem Radiant)")
                        else:
                            st.write(f"- 🔴 **{d_name}** supera **{r_name}** (`+{abs(edge)*100:.1f}%` vantagem Dire)")

            # Pilar 2: Sinergias e Combos de Cada Time
            with card_col2:
                with st.container(border=True):
                    st.markdown("#### 2. 🤝 Química e Sinergias Aliadas")
                    rad_combos = []
                    for i in range(5):
                        for j in range(i+1, 5):
                            rad_combos.append((heroes_map[rad_ids[i]], heroes_map[rad_ids[j]], extractor.synergy_matrix[rad_ids[i], rad_ids[j]]))
                    rad_combos.sort(key=lambda x: x[2], reverse=True)

                    dire_combos = []
                    for i in range(5):
                        for j in range(i+1, 5):
                            dire_combos.append((heroes_map[dire_ids[i]], heroes_map[dire_ids[j]], extractor.synergy_matrix[dire_ids[i], dire_ids[j]]))
                    dire_combos.sort(key=lambda x: x[2], reverse=True)

                    st.markdown(f"- 🟢 **Melhor Combo Radiant:** **{rad_combos[0][0]} + {rad_combos[0][1]}** (`{rad_combos[0][2]*100:+.2f}%`)")
                    st.markdown(f"- 🟢 **Maior Atrito Radiant:** **{rad_combos[-1][0]} + {rad_combos[-1][1]}** (`{rad_combos[-1][2]*100:+.2f}%`)")
                    st.markdown(f"- 🔴 **Melhor Combo Dire:** **{dire_combos[0][0]} + {dire_combos[0][1]}** (`{dire_combos[0][2]*100:+.2f}%`)")
                    st.markdown(f"- 🔴 **Maior Atrito Dire:** **{dire_combos[-1][0]} + {dire_combos[-1][1]}** (`{dire_combos[-1][2]*100:+.2f}%`)")

            card_col3, card_col4 = st.columns(2)

            # Pilar 3: Timing e Escalonamento (Early vs Late)
            with card_col3:
                with st.container(border=True):
                    st.markdown("#### 3. ⏱️ O Relógio da Partida (Timing & Scaling)")
                    rad_early = float(v_rad[17]) * 100
                    dire_early = float(v_dire[17]) * 100
                    rad_late = float(v_rad[18]) * 100
                    dire_late = float(v_dire[18]) * 100
                    rad_slope = float(v_rad[19]) * 100
                    dire_slope = float(v_dire[19]) * 100

                    st.write(f"- **Poder no Early Game (<30 min):** Radiant `{rad_early:.1f}%` vs Dire `{dire_early:.1f}%`")
                    st.write(f"- **Poder no Late Game (>45 min):** Radiant `{rad_late:.1f}%` vs Dire `{dire_late:.1f}%`")
                    st.write(f"- **Curva de Escalonamento:** Radiant `{rad_slope:+.1f}%` vs Dire `{dire_slope:+.1f}%`")

                    if rad_slope > dire_slope + 2.5:
                        st.info("🕒 **Vantagem de Tempo:** O draft da Radiant escala melhor para o late game. A Dire está 'no relógio' e precisa pressionar antes dos 30 minutos.")
                    elif dire_slope > rad_slope + 2.5:
                        st.info("🕒 **Vantagem de Tempo:** O draft da Dire escala melhor para o late game. A Radiant precisa acelerar suas vantagens de mapa e fechar cedo.")
                    else:
                        st.info("⚖️ **Ritmo Equilibrado:** As duas equipes possuem janelas de força muito próximas ao longo do tempo.")

            # Pilar 4: Mecânicas Críticas de Teamfight
            with card_col4:
                with st.container(border=True):
                    st.markdown("#### 4. 🛡️ Ferramentas Críticas de Teamfight")
                    rad_profiles = [extractor.profiles[hid] for hid in rad_ids]
                    dire_profiles = [extractor.profiles[hid] for hid in dire_ids]

                    rad_pierce = [p.name.replace('npc_dota_hero_', '').replace('_', ' ').title() for p in rad_profiles if p.has_bkb_pierce]
                    dire_pierce = [p.name.replace('npc_dota_hero_', '').replace('_', ' ').title() for p in dire_profiles if p.has_bkb_pierce]

                    rad_dispel = [p.name.replace('npc_dota_hero_', '').replace('_', ' ').title() for p in rad_profiles if p.has_strong_dispel]
                    dire_dispel = [p.name.replace('npc_dota_hero_', '').replace('_', ' ').title() for p in dire_profiles if p.has_strong_dispel]

                    rad_save = [p.name.replace('npc_dota_hero_', '').replace('_', ' ').title() for p in rad_profiles if p.has_hard_save]
                    dire_save = [p.name.replace('npc_dota_hero_', '').replace('_', ' ').title() for p in dire_profiles if p.has_hard_save]

                    st.write(f"- **Pierces de BKB:** 🟢 `{len(rad_pierce)}` ({', '.join(rad_pierce) or 'Nenhum'}) | 🔴 `{len(dire_pierce)}` ({', '.join(dire_pierce) or 'Nenhum'})")
                    st.write(f"- **Strong Dispels:** 🟢 `{len(rad_dispel)}` ({', '.join(rad_dispel) or 'Nenhum'}) | 🔴 `{len(dire_dispel)}` ({', '.join(dire_dispel) or 'Nenhum'})")
                    st.write(f"- **Hard Saves:** 🟢 `{len(rad_save)}` ({', '.join(rad_save) or 'Nenhum'}) | 🔴 `{len(dire_save)}` ({', '.join(dire_save) or 'Nenhum'})")

            # 5. Calculadora Quantitativa de Apostas & Gestão de Banca (+EV)
            st.markdown("---")
            st.markdown("### 💰 Análise Quantitativa de Apostas & Gestão de Banca (+EV)")
            st.caption("Compare as odds da casa com a probabilidade calibrada do modelo para identificar valor esperado (+EV) e dimensionar a aposta via Critério de Kelly Fracionário.")

            col_o1, col_o2, col_b, col_r = st.columns([1, 1, 1.2, 1.2])
            with col_o1:
                odd_rad_input = st.number_input("Odd Radiant (Casa)", min_value=1.01, max_value=30.0, value=1.90, step=0.05, format="%.2f")
            with col_o2:
                odd_dire_input = st.number_input("Odd Dire (Casa)", min_value=1.01, max_value=30.0, value=1.90, step=0.05, format="%.2f")
            with col_b:
                bankroll_input = st.number_input("Sua Banca Total (R$)", min_value=10.0, max_value=1000000.0, value=1000.0, step=50.0, format="%.2f")
            with col_r:
                risk_profile = st.selectbox(
                    "Perfil de Gestão (Kelly)",
                    options=[
                        "Conservador (Quarter Kelly 25%)",
                        "Moderado (Half Kelly 50%)",
                        "Agressivo (Full Kelly 100%)"
                    ],
                    index=0
                )

            kelly_mult = 0.25 if "Conservador" in risk_profile else (0.50 if "Moderado" in risk_profile else 1.00)

            opp = evaluate_betting_opportunity(
                p_model_radiant=prob_rad,
                odd_radiant=odd_rad_input,
                odd_dire=odd_dire_input,
                bankroll=bankroll_input,
                kelly_multiplier=kelly_mult,
                max_stake_pct=0.05,
                min_ev=0.03
            )

            if opp.is_actionable:
                st.success(f"🟢 **OPORTUNIDADE +EV IDENTIFICADA: APOSTAR NO {opp.side_name.upper()}** (Odd: `{opp.odd:.2f}`)")

                b_m1, b_m2, b_m3, b_m4 = st.columns(4)
                with b_m1:
                    st.metric("Aposta Recomendada (Stake)", f"R$ {opp.recommended_stake:.2f}", f"{opp.stake_percentage:.2f}% da banca")
                with b_m2:
                    st.metric("Retorno Esperado (+EV)", f"+{opp.ev*100:.1f}%", "Por real apostado")
                with b_m3:
                    st.metric("Vantagem s/ Casa (Edge)", f"+{opp.edge*100:.1f}%", f"Modelo {opp.model_prob*100:.1f}% vs Casa {opp.market_prob_fair*100:.1f}%")
                with b_m4:
                    lucro_liq = opp.recommended_stake * (opp.odd - 1.0)
                    st.metric("Lucro Líquido Estimado", f"R$ {lucro_liq:.2f}", f"Retorno Bruto: R$ {opp.recommended_stake * opp.odd:.2f}")

                st.caption(f"ℹ️ **Diagnóstico:** {opp.reason}")
            else:
                st.warning("🔴 **SEM VALOR MATEMÁTICO: NENHUMA APOSTA RECOMENDADA**")
                st.write(f"ℹ️ {opp.reason}")

            with st.expander("📈 Ver Simulação de Monte Carlo da Banca (Projeção de 200 apostas)"):
                st.markdown("Projeção probabilística de evolução do patrimônio repetindo este padrão de vantagem ao longo de 200 apostas sequenciais:")

                sim_target_prob = opp.model_prob if opp.is_actionable else max(prob_rad, 1.0 - prob_rad)
                sim_target_odd = opp.odd if opp.is_actionable else (odd_rad_input if prob_rad >= 0.5 else odd_dire_input)

                mc = simulate_bankroll_growth(
                    p_win=sim_target_prob,
                    odd=sim_target_odd,
                    initial_bankroll=bankroll_input,
                    kelly_multiplier=kelly_mult,
                    max_stake_pct=0.05,
                    n_bets=200,
                    n_simulations=100
                )

                mc_col1, mc_col2, mc_col3 = st.columns(3)
                with mc_col1:
                    st.metric("Banca Mediana Projetada", f"R$ {mc.median_final_bankroll:.2f}", f"{((mc.median_final_bankroll / bankroll_input) - 1.0)*100:+.1f}% de ROI")
                with mc_col2:
                    st.metric("Pior Drawdown Mediano", f"{mc.max_drawdown_median:.1f}%", "Oscilação máxima esperada")
                with mc_col3:
                    st.metric("Risco de Perda Severa (>50%)", f"{mc.risk_of_ruin:.1f}%", "Probabilidade de perda >50%")

                df_trajs = pd.DataFrame({f"Trajetória {i+1}": traj for i, traj in enumerate(mc.trajectories_sample)})
                st.line_chart(df_trajs, height=250)

    # 4. Abas de Apoio e Storytelling (Abaixo da Seção Principal)
    st.markdown("---")
    st.subheader("📚 Inteligência de Domínio, História & Auditoria do Modelo")
    story_tab_hist, story_tab1, story_tab2, story_tab3, story_tab4 = st.tabs([
        "📖 A Jornada: O que Aprendemos (Jogo & Números)",
        "🦸 Meta-Game & Arquétipos do Patch 60",
        "🔬 A Ciência do Modelo (Auditoria)",
        "💰 Teoria dos Jogos & Gestão de Banca",
        "📊 Dados & Volume de Partidas"
    ])

    # Aba 0: A Jornada Científica — Evolução, Jogo e Números
    with story_tab_hist:
        st.markdown("#### 📖 Da Moeda ao Nível Profissional: A História da Nossa Inteligência")
        st.caption("Como evoluímos o modelo ao longo do tempo, o que a IA descobriu sobre o funcionamento real do Dota 2 e como interpretar os números com rigor estatístico.")

        # 1. Linha do Tempo Narrativa em Colunas/Cards
        st.markdown("##### 1. 🚀 Linha do Tempo: A Evolução conforme os Ajustes")
        tl1, tl2, tl3 = st.columns(3)
        with tl1:
            with st.container(border=True):
                st.markdown("###### Fase 1: O Chute Cego (50.4% → 55.2%)")
                st.write(
                    "**O começo:** Tentamos prever o vencedor olhando apenas os heróis isolados (One-Hot Encoding de 509 colunas). "
                    "O modelo acertava **50.4%** — o mesmo que jogar uma moeda para o alto.\n\n"
                    "**1º Ajuste:** Adicionamos atributos físicos e *cooldowns* de magias. Subimos para **55.2%**, provando que ultimates longos importam, mas ainda não sabíamos como um time interagia com o outro."
                )
        with tl2:
            with st.container(border=True):
                st.markdown("###### Fase 2: A Virada dos Counters (71.2% → 74.8%)")
                st.write(
                    "**A grande descoberta:** Extraímos os confrontos 1v1 históricos (Matchups), a química de **1,6 milhão de duplas** aliadas e as curvas de tempo de jogo (Early vs Late).\n\n"
                    "**O salto:** A acurácia explodiu de 55% para **74.77%** (e **0.8065 de ROC-AUC**). O modelo finalmente entendeu quem anula quem, quais duplas combam e quem ganha se o jogo passar de 45 minutos."
                )
        with tl3:
            with st.container(border=True):
                st.markdown("###### Fase 3: Escala Total & O Segredo do 'GG' (6.087 Jogos)")
                st.write(
                    "**O mistério:** Mil jogos profissionais estavam sendo barrados pelos filtros. Descobrimos que pros dão *disconnect* 5s antes do trono cair após digitar 'GG' (`leaver_status=1`).\n\n"
                    "**A consolidação definitiva:** Destravamos **100% de todas as partidas de Captain's Mode do Patch 60** na história da OpenDota (**6.087 jogos**). Em testes cegos rigorosos com mais de 1.200 jogos nunca vistos, os modelos bateram recordes: **68.93% de acurácia**, **0.7568 de ROC-AUC** e Log Loss abaixo de **0.59**."
                )

        st.markdown("---")

        # 2. O que Aprendemos sobre o Jogo (Os Pesos do Dota 2)
        st.markdown("##### 2. 🎮 O que a IA Aprendeu sobre Como se Ganha no Dota 2")
        st.caption("Ao analisar as 6.087 partidas competitivas do Patch 60, o modelo atribuiu pesos matemáticos para cada aspecto de um draft. Estas são as 4 leis consolidadas pela máquina:")

        g_col1, g_col2 = st.columns([1.2, 1])
        with g_col1:
            st.markdown(
                """
                1. **⚔️ Anular o inimigo vale 8x mais do que qualquer atributo isolado (`Peso: +1.8677`):**
                   Ter respostas diretas (counters 1v1) contra as peças do adversário é, com folga, o fator número 1 de vitória no Dota 2 competitivo.
                2. **⏱️ Quem escala pro Late Game tem o relógio a seu favor (`Peso: +0.2277`):**
                   Times de pressão precoce (*zoo/push*) estão 'no relógio': se cometerem um único erro na subida de *High Ground*, a equipe com melhor curva de escalonamento pós-45 minutos vira o jogo com alta probabilidade.
                3. **🤝 Química aliada sustenta o plano de jogo (`Peso: +0.1530`):**
                   Duplas com sinergia histórica comprovada ganharam ainda mais peso à medida que adicionamos jogos, combinando controle de grupo (*lockdown*) com sincronia de iniciação.
                4. **🏃 Mobilidade e Cooldowns de Teamfight (`Peso: +0.1005` e `+0.1025`):**
                   Velocidade de movimento base e disponibilidade de habilidades de alto impacto determinam quem dita o ritmo das lutas pelo mapa.
                """
            )
        with g_col2:
            weights_chart_df = pd.DataFrame({
                "Fator Tático do Draft": [
                    "1. Vantagem de Counter 1v1",
                    "2. Escalonamento Late Game",
                    "3. Sinergia entre Aliados",
                    "4. Cooldown de Ultimates",
                    "5. Velocidade de Movimento",
                    "6. Sobreposição de Carries"
                ],
                "Impacto no Modelo (Peso)": [1.8677, 0.2277, 0.1530, 0.1025, 0.1005, 0.0432]
            }).set_index("Fator Tático do Draft")
            st.bar_chart(weights_chart_df, height=290)

        st.markdown("---")

        # 3. O que Aprendemos sobre os Números (A Verdade Estatística)
        st.markdown("##### 3. 📊 O que Aprendemos sobre os Números: Por que 68% em 4.200 jogos é mais forte que 75% em 2.200?")
        st.caption("À primeira vista, ver a acurácia nominal ir de 74.7% (em 2.2k jogos) para 67.8% (em 4.2k jogos) pode parecer uma queda. Na ciência de dados, foi o nosso maior ganho de maturidade:")

        n_col1, n_col2, n_col3 = st.columns(3)
        with n_col1:
            with st.container(border=True):
                st.markdown("###### 🏋️ 1. A Prova Dobrou e Incluiu o 'Erro Humano'")
                st.write(
                    "O teste cego saltou de **442 para 854 partidas**, incluindo ligas amadoras e Tier 2/3. "
                    "Nessas ligas, ocorrem mais erros humanos (*throws* com vantagem). Como a IA só vê os 10 heróis antes do jogo começar, os **~32% restantes são o espaço exato onde o talento humano e o clique no mouse decidem a partida**."
                )
        with n_col2:
            with st.container(border=True):
                st.markdown("###### 📈 2. A Vantagem contra o Mercado Continua +16%")
                st.write(
                    "Para medir a força real da IA, olhamos quanto ela bate o modelo básico (sem nossas features) na mesma prova difícil:\n"
                    "- **Modelo Básico em 4.2k jogos:** `51.87%` (moeda)\n"
                    "- **Nosso Modelo em 4.2k jogos:** `67.86%`\n\n"
                    "Mantivemos **+16 pontos percentuais de vantagem pura** sobre o mercado."
                )
        with n_col3:
            with st.container(border=True):
                st.markdown("###### 🎯 3. Honestidade Calibrada ($T = 1.0409$)")
                st.write(
                    "Modelos pequenos costumam ser 'mentirosos': gritam 90% de certeza quando a chance real é 60%. "
                    "Com 4.267 partidas, nossa temperatura de calibração chegou a **$1.0409$** (onde $1.00$ é a perfeição matemática). "
                    "Se a barra acima mostra **65%**, você pode confiar seu capital que a chance real é 65%."
                )

    # Aba 1: Meta-Game & Tendências
    with story_tab1:
        st.markdown("##### Dinâmica de Poder dos Heróis no Patch Atual (v7.41)")
        m_col1, m_col2 = st.columns(2)

        with m_col1:
            st.markdown("##### ⏱️ Top Hiper-Carries (Late-Game Scaling)")
            st.caption("Heróis que mais ganham taxa de vitória conforme a partida passa dos 45 minutos:")
            if scaling_data:
                df_scale = pd.DataFrame(scaling_data)
                top_late = df_scale.sort_values(by='scaling', ascending=False).head(8)
                st.dataframe(
                    top_late[['hero', 'early_wr', 'late_wr', 'scaling']].rename(
                        columns={'hero': 'Herói', 'early_wr': 'Win% (<30m)', 'late_wr': 'Win% (>45m)', 'scaling': 'Escalonamento'}
                    ).style.format({'Win% (<30m)': '{:.1f}%', 'Win% (>45m)': '{:.1f}%', 'Escalonamento': '+{:.1f}%'}),
                    use_container_width=True
                )

        with m_col2:
            st.markdown("##### ⚡ Top Heróis de Early Tempo / Zoo Push")
            st.caption("Heróis que atingem seu ápice antes dos 30 minutos e precisam finalizar cedo:")
            if scaling_data:
                top_early = df_scale.sort_values(by='scaling', ascending=True).head(8)
                st.dataframe(
                    top_early[['hero', 'early_wr', 'late_wr', 'scaling']].rename(
                        columns={'hero': 'Herói', 'early_wr': 'Win% (<30m)', 'late_wr': 'Win% (>45m)', 'scaling': 'Escalonamento'}
                    ).style.format({'Win% (<30m)': '{:.1f}%', 'Win% (>45m)': '{:.1f}%', 'Escalonamento': '{:.1f}%'}),
                    use_container_width=True
                )

        st.markdown("##### 🤝 Top 10 Maiores Sinergias (Combos Aliados do Dota 2)")
        if synergies_data:
            df_syn = pd.DataFrame(synergies_data)
            top_syn = df_syn.sort_values(by='synergy', ascending=False).head(10)
            st.dataframe(
                top_syn[['combo', 'games', 'winrate', 'synergy']].rename(
                    columns={'combo': 'Dupla Aliada', 'games': 'Partidas Analisadas', 'winrate': 'Win Rate Real', 'synergy': 'Bônus Sinergia'}
                ).style.format({'Win Rate Real': '{:.1f}%', 'Bônus Sinergia': '+{:.2f}%'}),
                use_container_width=True
            )

    # Aba 2: Ciência do Modelo & Auditoria
    with story_tab2:
        st.markdown("##### 📈 Evolução Histórica dos Modelos no Projeto")
        st.caption("Evolução rigorosa no conjunto de teste congelado (20% das partidas mais recentes, nunca vistas pelo treino):")

        evolution_df = pd.DataFrame([
            {"Iteração": "1. Baseline (Picks/Bans puros)", "Features": "509 esparsas", "Amostra": "2.211", "ROC-AUC": "0.5553", "Acurácia": "55.08%", "Log Loss": "0.7178", "Brier Score": "0.2601"},
            {"Iteração": "2. NN V1 (Cooldowns & Atributos)", "Features": "11 comp", "Amostra": "3.449", "ROC-AUC": "0.5595", "Acurácia": "55.20%", "Log Loss": "0.6875", "Brier Score": "0.2472"},
            {"Iteração": "3. Etapa 1 (Matchups H2H + Mecânicas)", "Features": "15 comp", "Amostra": "2.211", "ROC-AUC": "0.7849", "Acurácia": "71.17%", "Log Loss": "0.5608", "Brier Score": "0.1895"},
            {"Iteração": "4. Etapa 2 (Sinergias Intra-Equipe)", "Features": "18 comp", "Amostra": "2.211", "ROC-AUC": "0.7819", "Acurácia": "70.87%", "Log Loss": "0.5577", "Brier Score": "0.1888"},
            {"Iteração": "5. Etapa 3 (Curvas de Duração & Timing)", "Features": "21 comp", "Amostra": "2.211", "ROC-AUC": "0.8065", "Acurácia": "74.77%", "Log Loss": "0.5375", "Brier Score": "0.1803"},
            {"Iteração": "6. Regressão Logística (Base 4.267)", "Features": "509 + 21 comp", "Amostra": "4.267", "ROC-AUC": "0.7432", "Acurácia": "67.80%", "Log Loss": "0.6028", "Brier Score": "0.2071"},
            {"Iteração": "7. Rede Neural Calibrada (Base 4.267)", "Features": "Attention + 21 comp", "Amostra": "4.267", "ROC-AUC": "0.7438", "Acurácia": "67.86%", "Log Loss": "0.5976", "Brier Score": "0.2050"},
            {"Iteração": "8. Regressão Logística (Base 4.593)", "Features": "509 + 21 comp", "Amostra": "4.593", "ROC-AUC": "0.7324", "Acurácia": "65.94%", "Log Loss": "0.6169", "Brier Score": "0.2129"},
            {"Iteração": "9. Rede Neural Calibrada (Base 4.593)", "Features": "Attention + 21 comp", "Amostra": "4.593", "ROC-AUC": "0.7356", "Acurácia": "67.10%", "Log Loss": "0.6083", "Brier Score": "0.2094"},
            {"Iteração": "10. Regressão Logística (Base 6.087)", "Features": "509 + 21 comp", "Amostra": "6.087", "ROC-AUC": "0.7566", "Acurácia": "68.88%", "Log Loss": "0.5894", "Brier Score": "0.2017"},
            {"Iteração": "★ Rede Neural Calibrada (Base 6.087)", "Features": "Attention + 21 comp", "Amostra": "6.087", "ROC-AUC": "0.7568", "Acurácia": "68.93%", "Log Loss": "0.5858", "Brier Score": "0.2005"}
        ])
        st.dataframe(evolution_df, use_container_width=True)

        col_proof1, col_proof2, col_proof3 = st.columns(3)
        with col_proof1:
            st.markdown("###### 🛡️ Invariância à Ordem")
            st.write("A ordem em que os 5 heróis são escolhidos na equipe não afeta a predição. Garantido por operações simétricas (Deep Sets e Pooling).")
        with col_proof2:
            st.markdown("###### ⚖️ Antissimetria Perfeita")
            st.write("Inverter os lados (Radiant ↔ Dire) inverte rigorosamente o sinal da predição: $P(\text{Rad}) + P(\text{Dire}) = 1.00$.")
        with col_proof3:
            st.markdown("###### 🎯 Calibração de Probabilidade")
            st.write("Temperatura calibrada ($T=0.8194$, $ECE=0.0413$) refinando as probabilidades para o cálculo de valor esperado (+EV) no Critério de Kelly.")

    # Aba 3: Teoria dos Jogos & Gestão de Banca
    with story_tab3:
        st.markdown("##### 🎲 Matemática Quantitativa: Por que o Longo Prazo Ganha das Casas de Apostas")
        st.caption("Apostar com sucesso não é tentar adivinhar todas as partidas, mas sim encontrar assimetrias de preço (+EV) e gerenciar a banca para eliminar o risco de ruína.")

        tg_col1, tg_col2 = st.columns(2)
        with tg_col1:
            with st.container(border=True):
                st.markdown("###### 1. O que é o Valor Esperado (+EV)?")
                st.write("Se uma moeda justa paga odd 2.20 para sair cara, você tem uma aposta com **+10% de EV**. Mesmo perdendo algumas rodadas pontuais, em centenas de repetições seu patrimônio crescerá de forma previsível.")
                st.latex(r"\text{EV} = (P_{\text{modelo}} \times \text{Odd}) - 1")

            with st.container(border=True):
                st.markdown("###### 2. O Vigorish (A Taxa da Casa)")
                st.write("As casas de apostas embutem uma comissão invisível (overround) de 5% a 10% nas cotações. Apostar 'no feeling' significa lutar contra essa taxa constante. Nosso motor desconta essa margem para revelar a probabilidade justa.")
                st.latex(r"\text{Vig} = \left(\frac{1}{\text{Odd}_{\text{Rad}}} + \frac{1}{\text{Odd}_{\text{Dire}}}\right) - 1")

        with tg_col2:
            with st.container(border=True):
                st.markdown("###### 3. O Critério de Kelly Fracionário")
                st.write("Criado por John Kelly no Bell Labs e adotado por fundos quantitativos, calcula o tamanho exato da aposta. Usamos o **Quarter Kelly (25%)**, que oferece ~85% do crescimento teórico reduzindo a volatilidade em 75%.")
                st.latex(r"f^* = \frac{P \times O - 1}{O - 1} \implies \text{Stake} = \text{Banca} \times \min(0.25 \times f^*, 5\%)")

            with st.container(border=True):
                st.markdown("###### 4. A Assimetria do Live Draft no Dota 2")
                st.write("Três quartos da partida dependem da dinâmica dos 10 heróis. Enquanto a maioria do mercado aposta na fama dos times, nosso motor analisa matchups 1v1, sinergias e escalonamento em milissegundos.")

    # Aba 4: Dados & Histórico do Dataset
    with story_tab4:
        st.markdown("##### 📊 Metadados do Banco SQLite de Treinamento")
        c1, c2, c3 = st.columns(3)
        c1.metric("Partidas Válidas", f"{len(matches):,}")
        c2.metric("Eventos de Draft", f"{len(events):,}")
        c3.metric("Heróis no Catálogo", len(heroes_map))

        st.markdown("##### Histórico de Picks e Bans Mais Frequentes do Torneio")
        filtered_events = events.copy()
        if not filtered_events.empty:
            filtered_events['hero_name'] = filtered_events['hero_id'].map(heroes_map).fillna("Unknown")
            b_col1, b_col2 = st.columns(2)
            with b_col1:
                st.write("**Top 10 Heróis Mais Pickados:**")
                st.bar_chart(filtered_events[filtered_events['is_pick'] == 1]['hero_name'].value_counts().head(10))
            with b_col2:
                st.write("**Top 10 Heróis Mais Banidos:**")
                st.bar_chart(filtered_events[filtered_events['is_pick'] == 0]['hero_name'].value_counts().head(10))


if __name__ == "__main__":
    main()
