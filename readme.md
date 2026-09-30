# Dota 2 Win Predictor (MVP)

Um sistema MVP escrito em Python para prever a probabilidade de vitória do time Radiant em partidas de Dota 2 com base apenas nos **picks e bans** e no **lado** (Radiant/Dire). O projeto utiliza dados históricos partidas profissionais e ranqueadas de alto nível obtidos via OpenDota API.

## 📋 Funcionalidades

*   **Ingestão de Dados:** Coleta partidas da API OpenDota (Pro Matches e Public Matches High Rank).
*   **Filtros de Qualidade:** Apenas Captain's Mode, Duração > 30min, Sem abandonos (leavers), Draft Completo (10 picks).
*   **Armazenamento:** Persistência em SQLite relacional (`matches`, `draft_events`).
*   **Feature Engineering:** Criação de matriz esparsa com One-Hot Encoding para Picks/Bans e Patch.
*   **Machine Learning:** Treinamento de Regressão Logística (Baseline) com split temporal.
*   **Inferência:** Script CLI para prever vitória com base em nomes de heróis.

## 🛠️ Instalação

1.  **Clone o repositório:**
    ```bash
    git clone <repo-url>
    cd dota2req1
    ```

2.  **Crie e ative um ambiente virtual:**
    ```bash
    python -m venv .venv
    # Windows
    .venv\Scripts\activate
    # Linux/Mac
    source .venv/bin/activate
    ```

3.  **Instale as dependências:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Configure o ambiente:**
    Copie o exemplo e (opcionalmente) adicione sua API Key do OpenDota.
    ```bash
    copy .env.example .env
    ```

5.  **Inicialize o Banco de Dados:**
    ```bash
    python -m src.db.schema
    ```

## 🚀 Como Usar

### 1. Ingestão de Dados (Coleta Contínua e Backfill)
Baixe partidas profissionais ou ranqueadas. O pipeline suporta paginação contínua e checkpoints incrementais:

```bash
# Coletar partidas recentes (Forward - a partir do último checkpoint)
python -m src.opendota.ingest --dataset pro --direction forward --limit 100

# Coletar histórico retroativo (Backfill - descendo cronologicamente no passado)
python -m src.opendota.ingest --dataset pro --direction backfill --limit 500
```


### 2. Preparar Dataset
Baixe os nomes dos heróis (necessário para inferência e tradução de IDs) e construa as features.

```bash
# Baixar constantes de heróis
python -m src.opendota.fetch_constants

# Construir dataset (X.npz, y.npy)
python -m src.features.build_features
```

### 3. Treinar Modelos

#### Opção A: Rede Neural PyTorch (Deep Sets + Atenção + MMR) [Recomendado]
```bash
# 1. Preparar dataset estruturado e tensores neurais
python -m src.features.neural_features

# 2. Treinar rede neural com Early Stopping e Calibração
python -m src.model.train_nn --epochs 80 --batch-size 32
```

#### Opção B: Regressão Logística (Baseline One-Hot)
```bash
# 1. Construir matriz esparsa
python -m src.features.build_features

# 2. Treinar baseline
python -m src.model.train
```

### 4. Predição (Inferência)
Preveja a vitória dado um draft utilizando nomes de heróis em inglês ou IDs numéricos:

```bash
# Inferência com Rede Neural (Default pós-fase 3)
python -m src.model.predict --backend nn --rad "Anti-Mage" "Crystal Maiden" "Axe" "Lion" "Zeus" \
                                        --dire "Pudge" "Sniper" "Invoker" "Techies" "Chen"

# Inferência com Regressão Logística
python -m src.model.predict --backend lr --rad "Anti-Mage" "Crystal Maiden" "Axe" "Lion" "Zeus" \
                                        --dire "Pudge" "Sniper" "Invoker" "Techies" "Chen"
```

### 5. Dashboard Analítico e Simulador Interativo
Abra o painel web interativo para visualizar gráficos do meta e testar drafts visualmente:

```bash
streamlit run src/dashboard/app.py
```


## 📂 Estrutura do Projeto

```
dota2req1/
├── data/                   # SQLite, JSONs e arquivos processados (.npz)
├── models/                 # Modelos serializados (.joblib)
├── reports/                # Métricas e estatísticas
├── src/
│   ├── config.py           # Configurações globais
│   ├── db/                 # Schema e Repositório SQLite
│   ├── features/           # Filtros e Construção de Dataset
│   ├── model/              # Scripts de Treino e Predição
│   └── opendota/           # Cliente API e Scripts de Ingestão
└── ...
```

## 📊 Telemetria

Para ver estatísticas do dataset coletado:

```bash
python -m src.features.dataset_stats
```

## 📝 Decisões de Design (MVP)

*   **Anti-Leakage:** Nenhuma feature de "dentro do jogo" (ouro, XP, KDA) é usada. Apenas dados disponíveis no momento do draft.
*   **Modelo Simples:** Regressão Logística foi escolhida como baseline robusta para dados esparsos de alta dimensionalidade.
*   **SQLite:** Banco em arquivo único para facilidade de transporte e setup sem infraestrutura complexa.
