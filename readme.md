# 🛡️ Aegis Draft — Dota 2 Win Predictor & Quantitative Betting Engine

[![Python 3.10+](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-FF4B4B.svg)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3%2B-F7931E.svg)](https://scikit-learn.org/)
[![Test Suite](https://img.shields.io/badge/pytest-34%20passed-success.svg)](tests/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

O **Aegis Draft** é um sistema completo de inteligência artificial preditiva e análise quantitativa de apostas para o **Dota 2 (Patch 7.41 / Patch 60)**. O projeto combina modelos de Deep Learning e Machine Learning interpretável para prever a probabilidade de vitória de confrontos competitivos utilizando apenas a seleção de heróis (*draft*), além de fornecer um motor de **Teoria dos Jogos** que identifica assimetrias de preço (*+EV - Positive Expected Value*) contra casas de apostas com gestão de banca pelo **Critério de Kelly Fracionário**.

---

## 🌟 Principais Recursos

1. **Previsão de Vitória em Tempo Real (<50ms):**
   - **Deep Learning PyTorch:** Modelo com embeddings de heróis, camadas de Multi-Head Self-Attention (intra-equipe), Cross-Attention (inter-equipes) e calibração de temperatura ($T=1.0999$, $ECE=0.0504$). Atinge **74.77% de Acurácia** no conjunto de teste temporal congelado.
   - **Regressão Logística Calibrada:** Modelo linear interpretável que quebrou a barreira de **0.8065 de ROC-AUC** e **72.69% de Acurácia** com Brier Score de **0.1803**.

2. **Engenharia de Features Multidimensional (21 Atributos):**
   - **Etapa 1 (Confrontos Diretos & Mecânicas):** Matriz anti-simétrica de matchups 1v1 com regularização Bayesiana empírica ($C=20$), mais indicadores de habilidades críticas de combate (*BKB Pierce*, *Strong Dispel*, *Hard Save*).
   - **Etapa 2 (Sinergias Intra-Equipe):** Matriz de química de aliados construída a partir da agregação de mais de 1.600.000 pares de jogadores observados no OpenDota, extraindo `avg_synergy`, `max_synergy` e `min_synergy`.
   - **Etapa 3 (Curvas de Tempo & Escalonamento):** Curvas de taxa de vitória por faixa de tempo dos 127 heróis, mensurando `early_power` ($\le 30$m), `late_power` ($> 45$m) e o coeficiente de escalonamento `scaling_factor`.

3. **Garantias Matemáticas Invioláveis:**
   - **Antissimetria Estrita:** $\Delta(\text{Dire}, \text{Radiant}) = -\Delta(\text{Radiant}, \text{Dire}) \implies P(\text{Rad}) + P(\text{Dire}) = 1.00$.
   - **Invariância à Permutação:** A ordem de clique/seleção dos 5 heróis na equipe não altera o resultado da predição (*Deep Sets Pooling*).

4. **Motor Quantitativo de Apostas & Teoria dos Jogos (`src/betting`):**
   - **Removedor de Vigorish:** Extrai o *overround* invisível (5% a 10%) cobrado pelas casas para obter a probabilidade justa implícita ($P_{fair}$).
   - **Cálculo de +EV e Edge:** Identifica desajustes de cotação contra o mercado onde o retorno esperado compensa o risco.
   - **Critério de Kelly Fracionário:** Recomenda a alocação exata da banca (*Stake*) com teto prudencial de risco (máximo 5% por aposta) sob perfis Conservador (Quarter Kelly 25%), Moderado (Half Kelly 50%) ou Agressivo.
   - **Simulação de Monte Carlo:** Simula 200 apostas em 100 universos estocásticos para projetar crescimento mediano, pior *drawdown* e risco de ruína.

5. **Dashboard Interativo Streamlit ([`src/dashboard/app.py`](src/dashboard/app.py)):**
   - **Simulador Hero:** Seleção dinâmica 5v5 com exclusão mútua e *presets* táticos instantâneos (*Late-Game vs Zoo Push*, *Teamfight vs Pickoff*, *Pro Balanced*).
   - **Barra Tug-of-War:** Placar visual animado de probabilidades.
   - **Diagnóstico nos 4 Pilares:** Explicabilidade transparente (Counters 1v1, Sinergias Aliadas, Relógio de Timing e Mecânicas de Teamfight).
   - **Calculadora de Odds & Gestão de Banca:** Embutida diretamente abaixo do draft.

---

## 📊 Benchmark Comparativo dos Modelos (Patch 60 — 2.211 Partidas)

Avaliação em conjunto de teste congelado (20% das partidas cronologicamente mais recentes):

| Modelo | Features | ROC-AUC | Acurácia | Log Loss (Calibrado) | Brier Score | ECE |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline (Picks/Bans puros)** | 509 esparsas | 0.5553 | 55.08% | 0.7178 | 0.2601 | 0.1240 |
| **NN V1 (Cooldowns & Atributos)** | 11 comp | 0.5595 | 55.20% | 0.6875 | 0.2472 | 0.0706 |
| **NN Etapa 1 (Matchups + Mecânicas)** | Attention + 15 comp | 0.7849 | 71.17% | 0.5608 | 0.1895 | 0.0537 |
| **NN Etapa 2 (Sinergias Intra-Equipe)** | Attention + 18 comp | 0.7819 | 70.87% | 0.5577 | 0.1888 | 0.0437 |
| **Regressão Logística (Etapa 3 Final)** | 509 + 21 comp | **0.8065** 🏆 | **72.69%** | **0.5375** 🏆 | **0.1803** 🏆 | — |
| **Rede Neural PyTorch (Etapa 3 Final)** | Attention + 21 comp | **0.7953** | **74.77%** 🏆 | **0.5510** | **0.1844** | **0.0504** 🏆 |

---

## 🏗️ Estrutura do Repositório

```text
dota2req1/
├── .streamlit/
│   └── config.toml             # Configurações do servidor Streamlit e tema Dark/Emerald
├── data/
│   ├── constants/              # Metadados e matrizes do Patch 60 (v7.41)
│   │   ├── heroes.json         # Atributos, papéis e estatísticas de combate
│   │   ├── hero_abilities.json # Mapeamento de magias por herói
│   │   ├── abilities.json      # BKB Pierce, Strong Dispel, Cooldowns, Hard Saves
│   │   ├── matchups.json       # Matriz 127x127 de vantagens 1v1 Bayesianas
│   │   ├── synergies.json      # Matriz 127x127 de sinergias de aliados empíricas
│   │   └── durations.json      # Curvas de win rate por duração de partida
│   └── opendota.sqlite         # Banco relacional local de partidas e eventos de draft
├── docs/                       # Documentação aprofundada de engenharia e modelagem
│   ├── ARQUITETURA_DO_SISTEMA.md       # Arquitetura completa de dados, redes e finanças
│   ├── GUIA_DE_USO_E_DEPLOY.md         # Manual de execução, testes e deploy em nuvem
│   ├── EVOLUCAO_TESTES_E_EXPERIMENTOS.md # Histórico detalhado de hipóteses e benchmarks
│   ├── CATALOGO_CONSTANTES_DOTA2.md    # Dicionário de mecânicas da Valve/OpenDota
│   └── arquitetura_modelo_visual.html  # Grafo visual da arquitetura neural
├── models/
│   ├── draft_win_lr.joblib     # Pesos treinados da Regressão Logística
│   ├── draft_win_scaler.joblib # Scaler anti-simétrico de features de composição
│   └── draft_win_nn/           # Artefatos da Rede Neural PyTorch
│       ├── weights.pt          # Pesos binários treinados da rede neural
│       ├── model_config.json   # Hiperparâmetros da arquitetura (comp_dim=21)
│       ├── calibration.json    # Parâmetro de temperatura T=1.0999
│       ├── encoder.json        # Vocabulários de IDs de heróis e tokens
│       └── metrics.json        # Métricas no conjunto de validação e teste
├── src/
│   ├── betting/                # Motor de Teoria dos Jogos e Gestão de Banca
│   │   ├── __init__.py
│   │   └── engine.py           # Vigorish, Expected Value (+EV), Kelly e Monte Carlo
│   ├── dashboard/              # Interface Web Interativa
│   │   └── app.py              # Aplicação Streamlit de simulação e explicabilidade
│   ├── db/                     # Camada de persistência relacional SQLite
│   │   ├── schema.py           # Esquema DDL com índices e chaves estrangeiras
│   │   └── repo.py             # Repositório transacional atômico e idempotente
│   ├── features/               # Extração e normalização de features
│   │   ├── composition_features.py # Extrator das 21 features (Matchups, Sinergias, Curvas)
│   │   ├── neural_features.py      # Codificação de tensores PyTorch e split temporal
│   │   └── build_features.py       # Pipeline tabular para modelos clássicos
│   ├── model/                  # Arquiteturas, treinamento e inferência
│   │   ├── neural_network.py   # Rede PyTorch com Attention e Deep Sets Pooling
│   │   ├── neural_dataset.py   # DataLoaders com augment de simetria
│   │   ├── train_nn.py         # Loop de treino, early stopping e calibração
│   │   ├── predict_nn.py       # Wrapper de inferência de alta performance
│   │   ├── train.py            # Treinamento da Regressão Logística
│   │   └── predict.py          # Inferência tabular
│   ├── opendota/               # Ingestão e integração com API OpenDota
│   │   ├── client.py           # Cliente HTTP com rate limiting e backoff
│   │   ├── fetch_constants.py  # Download e saneamento das constantes do jogo
│   │   └── ingest.py           # Ingestão profissional e comunitária com checkpoint
│   └── config.py               # Configurações globais e variáveis de ambiente
├── tests/                      # Suíte de 34 testes automatizados (pytest)
│   ├── test_betting.py         # 10 testes do motor quantitativo de apostas e Kelly
│   ├── test_composition_features.py # 7 testes de invariância, simetria e dimensões
│   ├── test_db_smoke.py        # 1 teste de integridade SQLite e idempotência
│   ├── test_filters.py         # 8 testes de filtragem estrita de partidas
│   ├── test_neural_features.py # 4 testes de encoding e split temporal anti-leakage
│   └── test_neural_model.py    # 4 testes da rede PyTorch e propagação de gradientes
├── Dockerfile                  # Container Docker para deploy em produção
├── Procfile                    # Configuração para PaaS (Railway, Heroku, Render)
├── requirements.txt            # Dependências Python com compatibilidade multi-SO
└── README.md
```

---

## 🚀 Instalação e Execução Local

### 1. Clonar o Repositório e Configurar o Ambiente Virtual
```bash
git clone https://github.com/ataidefilipe/dota2req1.git
cd dota2req1

# Criar ambiente virtual
python -m venv .venv

# Ativar no Windows (PowerShell)
.venv\Scripts\Activate.ps1
# No Linux / macOS
source .venv/bin/activate

# Instalar dependências
pip install -r requirements.txt
```

### 2. Configurar Variáveis de Ambiente (Opcional)
Copie o arquivo de exemplo caso deseje realizar novas ingestões de dados:
```bash
cp .env.example .env
```
*No uso padrão e no dashboard, o sistema funciona perfeitamente mesmo sem preencher a API key.*

### 3. Executar o Dashboard Streamlit
```bash
streamlit run src/dashboard/app.py
```
O navegador abrirá automaticamente em `http://localhost:8501`.

### 4. Executar a Suíte de Testes Automatizados
```bash
pytest
```
*Valida os 34 testes unitários de ponta a ponta em menos de 10 segundos.*

---

## 🌐 Deploy em Produção

### Opção A: Streamlit Community Cloud (Recomendado — 100% Gratuito)
1. Faça login em [share.streamlit.io](https://share.streamlit.io/) com sua conta do GitHub.
2. Clique em **"New app"**.
3. Selecione o repositório `ataidefilipe/dota2req1`, branch `master` e arquivo principal:
   `src/dashboard/app.py`
4. Clique em **"Deploy"**.

### Opção B: Docker Container
```bash
# Construir a imagem
docker build -t dota2-win-predictor .

# Rodar o container
docker run -p 8501:8501 dota2-win-predictor
```

### Opção C: Railway / Render (PaaS)
O repositório já inclui o arquivo [`Procfile`](Procfile), configurado para vincular a aplicação na porta dinâmica fornecida pela nuvem (`$PORT`):
```text
web: streamlit run src/dashboard/app.py --server.port $PORT --server.address 0.0.0.0
```

---

## 📚 Documentação Técnica Adicional

Para detalhes matemáticos e especificações de engenharia, consulte:
* 📐 **[docs/ARQUITETURA_DO_SISTEMA.md](docs/ARQUITETURA_DO_SISTEMA.md):** Especificação completa das redes neurais, self-attention, fórmulas bayesianas e critério de Kelly.
* 🛠️ **[docs/GUIA_DE_USO_E_DEPLOY.md](docs/GUIA_DE_USO_E_DEPLOY.md):** Manual operacional de ingestão, retreino de modelos e troubleshooting em nuvem.
* 📈 **[docs/EVOLUCAO_TESTES_E_EXPERIMENTOS.md](docs/EVOLUCAO_TESTES_E_EXPERIMENTOS.md):** Registro histórico empírico de todas as 8 hipóteses científicas testadas.
* 📖 **[docs/CATALOGO_CONSTANTES_DOTA2.md](docs/CATALOGO_CONSTANTES_DOTA2.md):** Catálogo de extração de mecânicas do cliente da Valve.

---

## ⚖️ Licença
Distribuído sob a licença **MIT**. Consulte `LICENSE` para obter mais informações.
