import sqlite3
import json
import os
import sys
import pandas as pd
import numpy as np
from itertools import combinations
from collections import Counter, defaultdict

# Fix Windows console encoding for UTF-8 and emojis
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

def run_analysis(patch_id=59):
    conn = sqlite3.connect('data/opendota.sqlite')
    
    # 1. Carregar Dados
    matches = pd.read_sql(f"SELECT * FROM matches WHERE patch_id = {patch_id} AND dataset = 'pro'", conn)
    events = pd.read_sql(f"""
        SELECT de.*, m.radiant_win 
        FROM draft_events de
        JOIN matches m ON de.match_id = m.match_id
        WHERE m.patch_id = {patch_id} AND m.dataset = 'pro'
    """, conn)
    players = pd.read_sql(f"""
        SELECT mp.*, m.radiant_win
        FROM match_players mp
        JOIN matches m ON mp.match_id = m.match_id
        WHERE m.patch_id = {patch_id} AND m.dataset = 'pro'
    """, conn)
    conn.close()

    # Mapeamento de heróis
    heroes_map = {}
    if os.path.exists('data/heroes.json'):
        with open('data/heroes.json', 'r', encoding='utf-8') as f:
            heroes_list = json.load(f)
            heroes_map = {h['id']: h['localized_name'] for h in heroes_list}

    total_matches = len(matches)
    if total_matches == 0:
        print("Nenhuma partida encontrada para este patch.")
        return

    print("=================================================================")
    print(f"📊 RELATÓRIO ANALÍTICO DO META PROFISSIONAL - PATCH {patch_id}")
    print("=================================================================\n")

    # 1. Visão Geral e Temporal
    start_date = pd.to_datetime(matches['start_time'].min(), unit='s')
    end_date = pd.to_datetime(matches['start_time'].max(), unit='s')
    rad_wins = matches['radiant_win'].sum()
    dire_wins = total_matches - rad_wins
    rad_wr = (rad_wins / total_matches) * 100
    dire_wr = (dire_wins / total_matches) * 100

    dur_min = matches['duration'] / 60.0
    print("--- 1. OVERVIEW & EQUILÍBRIO DE LADOS ---")
    print(f"Total de Partidas Pro Analisadas: {total_matches}")
    print(f"Período Temporal: {start_date.strftime('%d/%m/%Y')} até {end_date.strftime('%d/%m/%Y')} ({(end_date - start_date).days} dias)")
    print(f"Radiant Win Rate: {rad_wr:.2f}% ({rad_wins} vitórias)")
    print(f"Dire Win Rate:    {dire_wr:.2f}% ({dire_wins} vitórias)")
    print(f"Duração Média: {dur_min.mean():.1f} min | Mediana: {dur_min.median():.1f} min (Mín: {dur_min.min():.1f}m, Máx: {dur_min.max():.1f}m)")
    print(f"Duração Quartis: 25%: {dur_min.quantile(0.25):.1f}m | 75%: {dur_min.quantile(0.75):.1f}m")

    # Lados por faixa de duração
    matches['dur_bucket'] = pd.cut(
        dur_min, 
        bins=[0, 30, 40, 50, 200], 
        labels=['< 30 min (Stomp)', '30-40 min (Mid-Game)', '40-50 min (Late-Game)', '> 50 min (Ultra-Late)']
    )
    print("\nTaxa de Vitória por Duração:")
    for bucket, grp in matches.groupby('dur_bucket', observed=False):
        b_total = len(grp)
        if b_total > 0:
            b_rad_wr = grp['radiant_win'].mean() * 100
            print(f"  - {bucket}: {b_total} jogos ({b_total/total_matches*100:.1f}%) -> Radiant WR: {b_rad_wr:.1f}% | Dire WR: {100-b_rad_wr:.1f}%")

    # First Pick vs Second Pick
    # Em Captain's Mode, a primeira escolha de herói (is_pick == 1 com menor order)
    first_picks = events[events['is_pick'] == 1].sort_values(['match_id', 'order']).groupby('match_id').first().reset_index()
    if not first_picks.empty:
        # team 0 = Radiant, team 1 = Dire
        # Se team == 0 e radiant_win == 1, FP venceu. Se team == 1 e radiant_win == 0, FP venceu.
        fp_won = ((first_picks['team'] == 0) & (first_picks['radiant_win'] == 1)) | \
                 ((first_picks['team'] == 1) & (first_picks['radiant_win'] == 0))
        fp_wr = fp_won.mean() * 100
        fp_rad_count = (first_picks['team'] == 0).sum()
        fp_dire_count = (first_picks['team'] == 1).sum()
        print(f"\nFirst Pick vs Second Pick:")
        print(f"  - First Pick Win Rate:  {fp_wr:.2f}%")
        print(f"  - Second Pick Win Rate: {100 - fp_wr:.2f}%")
        print(f"  - First Pick Escolhido por: Radiant {fp_rad_count}x ({fp_rad_count/total_matches*100:.1f}%) | Dire {fp_dire_count}x ({fp_dire_count/total_matches*100:.1f}%)")

    # 2. Hero Meta Analysis (Picks & Bans)
    print("\n--- 2. META DE HERÓIS (PICKS & BANS) ---")
    picks_df = events[events['is_pick'] == 1].copy()
    bans_df = events[events['is_pick'] == 0].copy()

    # Contagens
    pick_counts = picks_df['hero_id'].value_counts()
    ban_counts = bans_df['hero_id'].value_counts()
    all_heroes = set(pick_counts.index).union(set(ban_counts.index))

    meta_stats = []
    for hid in all_heroes:
        h_name = heroes_map.get(hid, f"Hero_{hid}")
        p_count = int(pick_counts.get(hid, 0))
        b_count = int(ban_counts.get(hid, 0))
        total_contest = p_count + b_count
        contest_rate = (total_contest / total_matches) * 100
        pick_rate = (p_count / total_matches) * 100
        ban_rate = (b_count / total_matches) * 100

        # Win rate
        h_picks = picks_df[picks_df['hero_id'] == hid]
        if len(h_picks) > 0:
            # vitória se (team == 0 e radiant_win == 1) ou (team == 1 e radiant_win == 0)
            wins = (((h_picks['team'] == 0) & (h_picks['radiant_win'] == 1)) | \
                    ((h_picks['team'] == 1) & (h_picks['radiant_win'] == 0))).sum()
            wr = (wins / len(h_picks)) * 100
            
            # Radiant vs Dire WR do herói
            rad_h = h_picks[h_picks['team'] == 0]
            rad_h_wr = (rad_h['radiant_win'] == 1).mean() * 100 if len(rad_h) > 0 else np.nan
            dire_h = h_picks[h_picks['team'] == 1]
            dire_h_wr = (dire_h['radiant_win'] == 0).mean() * 100 if len(dire_h) > 0 else np.nan
        else:
            wins = 0
            wr = np.nan
            rad_h_wr = np.nan
            dire_h_wr = np.nan

        meta_stats.append({
            'hero_id': hid,
            'name': h_name,
            'picks': p_count,
            'bans': b_count,
            'contest': total_contest,
            'contest_pct': contest_rate,
            'pick_pct': pick_rate,
            'ban_pct': ban_rate,
            'wins': wins,
            'win_rate': wr,
            'rad_wr': rad_h_wr,
            'dire_wr': dire_h_wr,
            'rad_picks': len(h_picks[h_picks['team'] == 0]) if len(h_picks) > 0 else 0,
            'dire_picks': len(h_picks[h_picks['team'] == 1]) if len(h_picks) > 0 else 0
        })

    meta_df = pd.DataFrame(meta_stats)

    print(f"Total de heróis distintos no Dota 2 representados: {len(meta_df)} (de {len(heroes_map)} cadastrados)")
    print(f"Heróis não contestados (0 picks e 0 bans): {len(heroes_map) - len(meta_df)}")

    # Top 10 Mais Contestados
    print("\n🔥 TOP 12 HERÓIS MAIS CONTESTADOS (Picks + Bans):")
    top_contest = meta_df.sort_values('contest', ascending=False).head(12)
    print(f"{'Herói':<20} | {'Contest':<8} | {'Taxa %':<7} | {'Picks':<6} | {'Bans':<6} | {'Win Rate %':<10}")
    print("-" * 68)
    for _, r in top_contest.iterrows():
        wr_str = f"{r['win_rate']:.1f}%" if pd.notna(r['win_rate']) else "N/A"
        print(f"{r['name']:<20} | {r['contest']:<8} | {r['contest_pct']:<6.1f}% | {r['picks']:<6} | {r['bans']:<6} | {wr_str:<10}")

    # Top 10 Mais Banidos
    print("\n🚫 TOP 10 HERÓIS MAIS BANIDOS (Respeito / Medo do Meta):")
    top_bans = meta_df.sort_values('bans', ascending=False).head(10)
    for i, (_, r) in enumerate(top_bans.iterrows(), 1):
        print(f" {i:2d}. {r['name']:<20} -> {r['bans']} bans ({r['ban_pct']:.1f}% dos jogos)")

    # Top 10 Mais Escolhidos (Picks)
    print("\n✅ TOP 10 HERÓIS MAIS ESCOLHIDOS (Volume de Jogo):")
    top_picks = meta_df.sort_values('picks', ascending=False).head(10)
    print(f"{'Herói':<20} | {'Picks':<6} | {'Pick %':<7} | {'Win Rate %':<10} | {'Rad WR':<8} | {'Dire WR':<8}")
    print("-" * 68)
    for _, r in top_picks.iterrows():
        print(f"{r['name']:<20} | {r['picks']:<6} | {r['pick_pct']:<6.1f}% | {r['win_rate']:<9.1f}% | {r['rad_wr']:<7.1f}% | {r['dire_wr']:<7.1f}%")

    # Maior Win Rate (mínimo 30 picks)
    relevant_heroes = meta_df[meta_df['picks'] >= 30]
    print(f"\n🏆 HERÓIS COM MAIOR TAXA DE VITÓRIA (Mínimo 30 picks, n={len(relevant_heroes)} heróis elegíveis):")
    top_wr = relevant_heroes.sort_values('win_rate', ascending=False).head(8)
    for i, (_, r) in enumerate(top_wr.iterrows(), 1):
        print(f" {i:2d}. {r['name']:<20} -> {r['win_rate']:.1f}% WR ({r['wins']}/{r['picks']} vitórias) | Bans: {r['bans']}")

    # Menor Win Rate (Traps do Meta, mínimo 30 picks)
    print(f"\n⚠️ 'ARMADILHAS' DO META (Menor Win Rate com >= 30 picks):")
    low_wr = relevant_heroes.sort_values('win_rate', ascending=True).head(8)
    for i, (_, r) in enumerate(low_wr.iterrows(), 1):
        print(f" {i:2d}. {r['name']:<20} -> {r['win_rate']:.1f}% WR ({r['wins']}/{r['picks']} vitórias) | Bans: {r['bans']}")

    # Maior Disparidade de Lado (Radiant vs Dire)
    side_disp = relevant_heroes[relevant_heroes['rad_picks'] >= 10][relevant_heroes['dire_picks'] >= 10].copy()
    side_disp['delta_rad_dire'] = side_disp['rad_wr'] - side_disp['dire_wr']
    print("\n⚖️ MAIORES VIESES DE LADO (Diferença de WR Radiant vs Dire, min 10 picks por lado):")
    print("Heróis que performam MUITO melhor no Radiant:")
    for _, r in side_disp.sort_values('delta_rad_dire', ascending=False).head(4).iterrows():
        print(f"  - {r['name']:<18}: Radiant {r['rad_wr']:.1f}% vs Dire {r['dire_wr']:.1f}% (Delta: +{r['delta_rad_dire']:.1f}%)")
    print("Heróis que performam MUITO melhor no Dire:")
    for _, r in side_disp.sort_values('delta_rad_dire', ascending=True).head(4).iterrows():
        print(f"  - {r['name']:<18}: Dire {r['dire_wr']:.1f}% vs Radiant {r['rad_wr']:.1f}% (Delta: {-r['delta_rad_dire']:.1f}%)")

    # 3. Sinergias (Duplas de Heróis)
    print("\n--- 3. SINERGIAS (DUPLAS DE HERÓIS NA MESMA EQUIPE) ---")
    pair_counts = Counter()
    pair_wins = Counter()

    for mid, group in picks_df.groupby('match_id'):
        rad_heroes = sorted(group[group['team'] == 0]['hero_id'].tolist())
        dire_heroes = sorted(group[group['team'] == 1]['hero_id'].tolist())
        rad_win = group['radiant_win'].iloc[0]

        for p in combinations(rad_heroes, 2):
            pair_counts[p] += 1
            if rad_win == 1:
                pair_wins[p] += 1
                
        for p in combinations(dire_heroes, 2):
            pair_counts[p] += 1
            if rad_win == 0:
                pair_wins[p] += 1

    pair_list = []
    for pair, count in pair_counts.items():
        if count >= 12: # mínimo 12 jogos juntos
            w = pair_wins[pair]
            wr = (w / count) * 100
            pair_list.append({
                'hero1': heroes_map.get(pair[0], f"Hero_{pair[0]}"),
                'hero2': heroes_map.get(pair[1], f"Hero_{pair[1]}"),
                'games': count,
                'wins': w,
                'win_rate': wr
            })

    pair_df = pd.DataFrame(pair_list)
    if not pair_df.empty:
        print("Top 6 Duplas Mais Frequentes:")
        for _, r in pair_df.sort_values('games', ascending=False).head(6).iterrows():
            print(f"  - {r['hero1']} + {r['hero2']}: {r['games']} jogos -> {r['win_rate']:.1f}% WR")
        
        print("\nTop 6 Duplas com Maior Taxa de Vitória (mínimo 12 jogos juntos):")
        for _, r in pair_df.sort_values('win_rate', ascending=False).head(6).iterrows():
            print(f"  - {r['hero1']} + {r['hero2']}: {r['win_rate']:.1f}% WR ({r['wins']}/{r['games']} vitórias)")

    # 4. Contexto Competitivo & Times
    print("\n--- 4. CONTEXTO COMPETITIVO & TIMES ---")
    rad_teams = matches[['radiant_team_id', 'radiant_win']].dropna()
    dire_teams = matches[['dire_team_id', 'radiant_win']].dropna()
    
    team_games = defaultdict(int)
    team_wins = defaultdict(int)

    for _, r in rad_teams.iterrows():
        tid = int(r['radiant_team_id'])
        if tid > 0:
            team_games[tid] += 1
            if r['radiant_win'] == 1:
                team_wins[tid] += 1

    for _, r in dire_teams.iterrows():
        tid = int(r['dire_team_id'])
        if tid > 0:
            team_games[tid] += 1
            if r['radiant_win'] == 0:
                team_wins[tid] += 1

    team_stats = []
    for tid, games in team_games.items():
        if games >= 10:
            w = team_wins[tid]
            team_stats.append({
                'team_id': tid,
                'games': games,
                'wins': w,
                'win_rate': (w / games) * 100
            })
    team_df = pd.DataFrame(team_stats)
    print(f"Total de Times Profissionais Identificados: {len(team_games)}")
    print(f"Total de Ligas/Torneios Distintos: {matches['leagueid'].nunique()}")
    if not team_df.empty:
        print("\nTimes Mais Ativos no Dataset (Top 5 por partidas):")
        for _, r in team_df.sort_values('games', ascending=False).head(5).iterrows():
            print(f"  - Team ID {int(r['team_id'])}: {int(r['games'])} jogos | {r['win_rate']:.1f}% WR")

    # 5. Telemetria de Jogadores & MMR
    print("\n--- 5. TELEMETRIA DE JOGADORES & MMR ---")
    valid_mmr = players['computed_mmr'].dropna()
    valid_ranks = players['rank_tier'].dropna()
    unique_players = players['account_id'].dropna().nunique()
    print(f"Jogadores Únicos (account_id): {unique_players}")
    print(f"Distribuição de MMR Calculado (n={len(valid_mmr)}):")
    print(f"  - Média: {valid_mmr.mean():.0f} | Mediana: {valid_mmr.median():.0f}")
    print(f"  - Mín: {valid_mmr.min():.0f} | Máx: {valid_mmr.max():.0f}")
    print(f"  - 25% Quartil: {valid_mmr.quantile(0.25):.0f} | 75% Quartil: {valid_mmr.quantile(0.75):.0f}")

    if len(valid_ranks) > 0:
        # No Dota 2: rank_tier 80+ é Immortal, 70-79 Divine, etc.
        immortals = (valid_ranks >= 80).sum()
        divines = ((valid_ranks >= 70) & (valid_ranks < 80)).sum()
        others = (valid_ranks < 70).sum()
        print(f"Distribuição de Patentes (Rank Tier):")
        print(f"  - Imortal (Tier 80+): {immortals} ({immortals/len(valid_ranks)*100:.1f}%)")
        print(f"  - Divino (Tier 70-79): {divines} ({divines/len(valid_ranks)*100:.1f}%)")
        print(f"  - Outros (Tier < 70): {others} ({others/len(valid_ranks)*100:.1f}%)")

    print("\n=================================================================")
    print("Análise concluída com sucesso.")

if __name__ == '__main__':
    run_analysis(59)
