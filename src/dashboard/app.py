import streamlit as st
import pandas as pd
import sqlite3
import json
import os

# Set page config
st.set_page_config(page_title="Dota 2 Win Predictor Dashboard", layout="wide")

@st.cache_data
def load_data():
    conn = sqlite3.connect('data/opendota.sqlite')
    matches = pd.read_sql("SELECT * FROM matches", conn)
    events = pd.read_sql("SELECT * FROM draft_events", conn)
    conn.close()

    heroes_map = {}
    heroes_list = []
    if os.path.exists('data/heroes.json'):
        with open('data/heroes.json', 'r', encoding='utf-8') as f:
            heroes = json.load(f)
            for h in heroes:
                heroes_map[h['id']] = h['localized_name']
                heroes_list.append(h['localized_name'])
    heroes_list.sort()

    return matches, events, heroes_map, heroes_list

def load_metrics():
    lr_metrics = {}
    nn_metrics = {}
    if os.path.exists('reports/metrics.json'):
        with open('reports/metrics.json', 'r', encoding='utf-8') as f:
            lr_metrics = json.load(f)
    if os.path.exists('reports/nn_metrics.json'):
        with open('reports/nn_metrics.json', 'r', encoding='utf-8') as f:
            nn_metrics = json.load(f)
    return lr_metrics, nn_metrics

def main():
    st.title("🛡️ Dota 2 Win Predictor Dashboard")

    try:
        matches, events, heroes_map, sorted_hero_names = load_data()
        lr_metrics, nn_metrics = load_metrics()
    except Exception as e:
        st.error(f"Erro ao carregar dados: {e}")
        return

    # Sidebar Filters
    st.sidebar.header("Filtros do Dataset")
    dataset_filter = st.sidebar.multiselect(
        "Dataset",
        matches['dataset'].unique(),
        default=matches['dataset'].unique()
    )

    filtered_matches = matches[matches['dataset'].isin(dataset_filter)]

    # KPI Row
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total de Partidas", len(filtered_matches))

    win_rate = filtered_matches['radiant_win'].mean() * 100 if not filtered_matches.empty else 0.0
    col2.metric("Taxa de Vitória Radiant", f"{win_rate:.1f}%")

    avg_duration = (filtered_matches['duration'].mean() / 60) if not filtered_matches.empty else 0.0
    col3.metric("Duração Média", f"{avg_duration:.1f} min")

    patches_count = filtered_matches['patch_id'].nunique() if not filtered_matches.empty else 0
    col4.metric("Patches Registrados", patches_count)

    # Tabs
    tab1, tab2, tab3 = st.tabs(["📊 Visão Geral", "🦸 Heróis & Meta", "🤖 Preditor & Modelos"])

    with tab1:
        st.subheader("Taxa de Vitória por Data")
        if not filtered_matches.empty:
            df = filtered_matches.copy()
            df['date'] = pd.to_datetime(df['start_time'], unit='s')
            daily_wins = df.set_index('date').resample('D')['radiant_win'].mean()
            st.line_chart(daily_wins)
        else:
            st.info("Nenhuma partida encontrada.")

    with tab2:
        st.subheader("Top Picks & Bans")
        filtered_ids = filtered_matches['match_id']
        filtered_events = events[events['match_id'].isin(filtered_ids)].copy()

        if not filtered_events.empty:
            filtered_events['hero_name'] = filtered_events['hero_id'].map(heroes_map).fillna("Unknown")

            col_p, col_b = st.columns(2)
            with col_p:
                st.write("**Top 10 Heróis Mais Escolhidos (Picks)**")
                picks = filtered_events[filtered_events['is_pick'] == 1]['hero_name'].value_counts().head(10)
                st.bar_chart(picks)

            with col_b:
                st.write("**Top 10 Heróis Mais Banidos (Bans)**")
                bans = filtered_events[filtered_events['is_pick'] == 0]['hero_name'].value_counts().head(10)
                st.bar_chart(bans)
        else:
            st.info("Nenhum evento de draft encontrado.")

    with tab3:
        st.subheader("Comparação de Modelos de Machine Learning")

        m_col1, m_col2 = st.columns(2)
        with m_col1:
            st.markdown("#### Baseline: Regressão Logística (One-Hot)")
            if lr_metrics:
                st.write(f"- **Log Loss no Teste:** `{lr_metrics.get('test_log_loss', 0.0):.4f}`")
                st.write(f"- **Log Loss no Treino:** `{lr_metrics.get('train_log_loss', 0.0):.4f}`")
                st.write(f"- **Acurácia no Teste:** `{lr_metrics.get('test_accuracy', 0.0) * 100:.1f}%`")
            else:
                st.info("Métricas da Regressão Logística não encontradas.")

        with m_col2:
            st.markdown("#### Rede Neural PyTorch (Deep Sets + Atenção)")
            if nn_metrics:
                st.write(f"- **Log Loss Calibrado (Teste):** `{nn_metrics.get('test_log_loss', 0.0):.4f}`")
                st.write(f"- **ROC-AUC (Teste):** `{nn_metrics.get('test_roc_auc', 0.0):.4f}`")
                st.write(f"- **Brier Score:** `{nn_metrics.get('test_brier_score', 0.0):.4f}`")
                st.write(f"- **Temperatura Calibrada:** `T = {nn_metrics.get('optimal_temperature', 1.0):.4f}`")
            else:
                st.info("Métricas da Rede Neural não encontradas.")

        st.markdown("---")
        st.subheader("🎮 Simulador de Draft Interativo")

        default_rad = ["Anti-Mage", "Crystal Maiden", "Axe", "Lion", "Zeus"]
        default_dire = ["Pudge", "Sniper", "Invoker", "Techies", "Chen"]

        # Garantir defaults válidos
        valid_rad = [h for h in default_rad if h in sorted_hero_names] or sorted_hero_names[:5]
        valid_dire = [h for h in default_dire if h in sorted_hero_names] or sorted_hero_names[5:10]

        d_col1, d_col2 = st.columns(2)
        with d_col1:
            st.markdown("##### 🟢 Picks Radiant (5 Heróis)")
            selected_rad = st.multiselect("Selecione exatamente 5 heróis:", sorted_hero_names, default=valid_rad, max_selections=5)

        with d_col2:
            st.markdown("##### 🔴 Picks Dire (5 Heróis)")
            selected_dire = st.multiselect("Selecione exatamente 5 heróis:", sorted_hero_names, default=valid_dire, max_selections=5)

        model_backend = st.radio("Selecione o Modelo:", ["Rede Neural PyTorch (Recomendado)", "Regressão Logística (Baseline)"], horizontal=True)

        if st.button("🔮 Calcular Probabilidade de Vitória"):
            if len(selected_rad) != 5 or len(selected_dire) != 5:
                st.warning("Selecione exatamente 5 heróis para cada equipe!")
            else:
                if "Rede Neural" in model_backend:
                    from src.model.predict_nn import NeuralWinPredictor
                    pred = NeuralWinPredictor()
                    prob = pred.predict(radiant_heroes=selected_rad, dire_heroes=selected_dire, patch_id=59)
                else:
                    from src.model.predict import WinPredictor
                    pred = WinPredictor()
                    prob = pred.predict(radiant_heroes=selected_rad, dire_heroes=selected_dire, patch_id=59)

                st.markdown("### Resultado da Predição")
                rad_pct = prob * 100
                dire_pct = (1.0 - prob) * 100

                st.progress(float(prob))
                res_col1, res_col2 = st.columns(2)
                res_col1.metric("Probabilidade Radiant", f"{rad_pct:.2f}%")
                res_col2.metric("Probabilidade Dire", f"{dire_pct:.2f}%")

                if prob >= 0.5:
                    st.success(f"🏆 Vitória Provável: **RADIANT** ({rad_pct:.1f}%)")
                else:
                    st.error(f"🏆 Vitória Provável: **DIRE** ({dire_pct:.1f}%)")

if __name__ == "__main__":
    main()
