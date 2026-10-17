## Contexto fixo do backlog (para agentes)

Projeto MVP em Python para:

* Coletar partidas via API do OpenDota.
* Persistir dados em SQLite.
* Treinar um modelo simples (baseline) para prever **P(Radiant vencer)** usando **apenas informações disponíveis após o draft e antes do jogo**:

  * picks/bans por lado (Radiant/Dire)
  * patch_id
  * (armazenar team_id/player_id para análise, **não usar no baseline**)
* Filtrar dataset:

  * somente Captain’s Mode
  * draft completo (10 picks; bans tolerados)
  * duração ≥ 30 min
  * descartar se qualquer player `leaver_status != 0`
  * descartar matches sem `picks_bans`
* Foco: partidas profissionais + ranked high (refletir o “mundo real”, sem balancear artificialmente).
* Sem over-engineering: scripts simples, logging básico, retries básicos, sem MLflow, sem serviço online.

Domínio: Dota 2.

---

## Definição de Pronto (DoR) para qualquer tarefa

Uma tarefa só entra em execução quando:

* Objetivo está em 1 frase.
* Entradas e saídas estão claras.
* Restrições “pré-jogo only” e filtros estão citados quando aplicável.
* Critério de verificação manual (como validar localmente) está definido.

---

## Definition of Done (DoD) global

Para considerar uma tarefa concluída:

* Código implementado e formatado (PEP8 “na prática”, sem bikeshedding).
* Rodando localmente via `python -m ...` ou script definido.
* Logging mínimo adiciona contexto (dataset, match_id, contadores, erros).
* Erros tratados sem interromper o pipeline (quando aplicável).
* Documentação curta atualizada (README ou doc/).
* Evidência de verificação:

  * comando(s) executado(s)
  * saída esperada (contadores, arquivo gerado, tabela populada)

---

## Convenções operacionais (para agentes)

* Preferir funções pequenas e puras; I/O isolado em camadas (client/db/pipeline).
* Nada de features de pós-jogo no dataset de treino (anti-leakage).
* “Falha controlada”: se um match não atende filtros, registrar e seguir.
* SQLite como fonte de verdade; scripts idempotentes.
* Cada tarefa altera o mínimo de arquivos possível.

---

# Backlog (MVP) — tarefas curtas, bem definidas e executáveis por agentes

## Milestone M0 — Bootstrap do repositório (base para os demais)

### B01 — Inicializar estrutura de diretórios e entrypoints

**Objetivo:** criar a estrutura `src/`, `data/`, `models/` e entrypoints `python -m` mínimos.
**Saída:** árvore de pastas + módulos vazios importáveis.
**DoD específico:** `python -c "import src"` (ou equivalente) funciona sem erro.

### B02 — Criar `requirements.txt` e `.env.example`

**Objetivo:** definir dependências mínimas e variáveis de ambiente.
**Inclui:** `OPENDOTA_API_KEY`, `OPENDOTA_RPS`, `SQLITE_PATH`, `PUBLICMATCHES_MIN_RANK`.
**DoD específico:** instalação em venv e import dos pacotes principais.

### B03 — Criar módulo de config central (`src/config.py`)

**Objetivo:** carregar `.env`, validar defaults e expor configuração tipada (ou dataclass simples).
**DoD específico:** `print(Config())` mostra valores e defaults previsíveis.

### B04 — Logging padrão do projeto

**Objetivo:** configurar logging com nível via env e formato consistente.
**DoD específico:** todos os scripts principais logam “start/end” e contadores.

---

## Milestone M1 — Banco (SQLite) e persistência

### B05 — Definir schema SQLite e migrations simples

**Objetivo:** implementar criação de tabelas:

* `matches`, `draft_events`, `match_players`, `ingestion_state`, `hero_ref`, `patch_ref` (as duas últimas podem ser opcionais, mas recomendadas).
  **DoD específico:** rodar `python -m src.db.schema` cria o `.sqlite` e tabelas.

### B06 — Repositório de acesso ao DB (CRUD mínimo)

**Objetivo:** implementar funções:

* `upsert_match(match_row)`
* `insert_draft_events(match_id, events[])`
* `insert_players(match_id, players[])`
* `get_ingestion_state(dataset)`
* `set_ingestion_state(dataset, last_match_id)`
  **DoD específico:** script de “smoke test” insere e lê um match fake.

### B07 — Idempotência por `match_id` + checkpoint

**Objetivo:** garantir que reexecutar ingestão não duplica registros.
**Decisão:** PK `match_id` + `ingested_at` como auditoria; checkpoint em `ingestion_state`.
**DoD específico:** reexecutar ingestão em um conjunto pequeno não altera contagem final.

---

## Milestone M2 — Cliente OpenDota e coleta (pro + ranked high)

### B08 — Implementar cliente HTTP com throttling e retry básico

**Objetivo:** criar `src/opendota/client.py` com:

* `get(url, params)`
* throttle por `OPENDOTA_RPS`
* retry em 429 e 5xx (ex.: 3 tentativas, backoff simples)
  **DoD específico:** logs mostram retries e respeitam throttle.

### B09 — Implementar listagem incremental de pro matches

**Objetivo:** coletar `match_id` via endpoint de pro matches com paginação `less_than_match_id`.
**Saída:** lista de match_ids + salvar checkpoint.
**DoD específico:** ao rodar duas vezes, segunda rodada busca “a partir do checkpoint”.

### B10 — Implementar listagem incremental de ranked high via publicMatches

**Objetivo:** coletar `match_id` via publicMatches filtrando por `min_rank` e paginando por `less_than_match_id`.
**DoD específico:** registro de contagem coletada e checkpoint atualizado.

### B11 — Implementar fetch de match detalhado por `match_id`

**Objetivo:** baixar `/matches/{match_id}` e retornar payload (JSON).
**DoD específico:** para um `match_id` válido, retorna campos essenciais (duration, radiant_win, picks_bans…).

### B12 — Extrair e persistir `matches`, `draft_events`, `match_players`

**Objetivo:** transformar JSON do match em linhas normalizadas nas tabelas.
**DoD específico:** após executar em 10 matches, DB contém:

* 10 linhas em `matches` (ou menos se filtradas)
* N eventos em `draft_events` por match
* 10 players por match em `match_players`

---

## Milestone M3 — Filtros de qualidade e “pré-jogo only”

### B13 — Implementar filtro: Captain’s Mode

**Objetivo:** aceitar apenas `game_mode == Captain’s Mode`.
**DoD específico:** logs indicam quantos matches foram descartados por modo.

### B14 — Implementar filtro: duração mínima 30 min

**Objetivo:** descartar se `duration < 1800`.
**DoD específico:** contadores de descarte por regra.

### B15 — Implementar filtro: leaver_status estrito

**Objetivo:** descartar match se algum player `leaver_status != 0`.
**DoD específico:** regra aplicada antes de persistir dataset final.

### B16 — Implementar filtro: draft completo e presença de picks_bans

**Objetivo:** descartar se `picks_bans` ausente; exigir 10 picks no total (5 por lado). Bans tolerados.
**DoD específico:** validação unitária simples com payloads mínimos (mock) cobrindo:

* sem picks_bans
* picks < 10
* picks = 10

### B17 — Garantir “anti-leakage” na seleção de colunas

**Objetivo:** documentar e impor que o dataset de treino só usa:

* picks/bans por lado
* patch_id
  **DoD específico:** função de build de dataset falha (erro explícito) se receber colunas “proibidas”.

---

## Milestone M4 — Dataset de ML (features) e baseline

### B18 — Construir `hero_ref` e resolver nome → hero_id

**Objetivo:** implementar carregamento/atualização de heróis (para inferência por nome).
**DoD específico:** dado “Axe” (exemplo), retorna hero_id correto (teste com 2–3 heróis).

### B19 — Construir `patch_ref` (lookup opcional)

**Objetivo:** armazenar referência de patch para análises e split.
**DoD específico:** tabela populada (mesmo que parcial) e utilizada no relatório de dataset.

### B20 — Builder de features (matriz esparsa) com picks/bans por lado

**Objetivo:** gerar X, y a partir do SQLite:

* colunas `rad_pick_*`, `dire_pick_*`, `rad_ban_*`, `dire_ban_*`, `patch_*`
* y = `radiant_win`
  **DoD específico:** salvar `dataset.npz` (ou equivalente) e `feature_map.json`.

### B21 — Split temporal por `start_time`

**Objetivo:** implementar split 80/20 ordenado por `start_time`.
**DoD específico:** confirmar por log que o conjunto de teste tem times mais recentes.

### B22 — Treinar baseline: Logistic Regression e avaliar log loss

**Objetivo:** treinar modelo único (pro + ranked_high) e reportar log loss.
**DoD específico:** gerar:

* `models/draft_win_lr.joblib`
* `reports/metrics.json` com log loss (train/test) e contagens

### B23 — (Opcional MVP+) Checagem rápida de calibração (sem calibrar)

**Objetivo:** gerar relatório simples (bins) para diagnosticar calibração.
**DoD específico:** `reports/calibration.json` com bins e médias.

---

## Milestone M5 — Inferência e interface

### B24 — Implementar `predict.py` aceitando nomes de heróis

**Objetivo:** entrada: lista de picks/bans por time em nomes; patch_id; saída: `p_radiant_win`.
**DoD específico:** rodar comando com payload de exemplo retorna probabilidade (float 0–1).

### B25 — Validar consistência de features entre treino e inferência

**Objetivo:** garantir que inferência usa o mesmo `feature_map.json` do treino.
**DoD específico:** erro claro se faltar herói/patch desconhecido ou se feature_map divergir.

### B26 — Exemplo de payload de inferência no README

**Objetivo:** documentar 1 exemplo mínimo de entrada (picks/bans) e a saída esperada.
**DoD específico:** exemplo executa localmente sem ajustes manuais.

---

## Milestone M6 — Documentação e operabilidade

### B27 — README: visão de arquitetura + pipeline + decisões MVP

**Objetivo:** consolidar o planejamento lógico:

* fontes, filtros, anti-leakage, schema, pipeline, comandos
  **DoD específico:** README permite que outro agente rode ingestão→treino→predict.

### B28 — Scripts de execução (comandos únicos)

**Objetivo:** padronizar comandos:

* `python -m src.opendota.ingest --dataset pro`
* `python -m src.opendota.ingest --dataset ranked_high`
* `python -m src.model.train`
* `python -m src.model.predict --json payload.json`
  **DoD específico:** cada comando tem `--help` útil e retorna exit code adequado.

### B29 — Telemetria mínima de dataset

**Objetivo:** relatório de contagens por motivo de descarte e distribuição por dataset.
**DoD específico:** `reports/dataset_stats.json` com contadores e timestamps.

---

# Milestone M7 — Correções, Estabilização e Escala

### B30 — Correção do filtro de duração mínima (Bug fix) [CONCLUÍDO]
* **Objetivo:** Corrigir condição em `src/features/filters.py` de `duration < 1300` para `duration < 1800` (30 min) e garantir aprovação no `test_filters.py`.
* **Resultado:** Corrigido para `1800`. Testes unitários rodando 100% OK.

### B31 — Isolamento do teste de banco (test_db_smoke) e suporte a MMR [CONCLUÍDO]
* **Objetivo:** Configurar `tests/test_db_smoke.py` para executar em banco temporário com teardown, validar `rank_tier` e `computed_mmr`, e expurgar registros de `smoke_test` do banco operacional `opendota.sqlite`.
* **Resultado:** Banco operacional limpo e testes isolados com teardown automático.

### B32 — Idempotência em tabelas filhas no repositório SQLite [CONCLUÍDO]
* **Objetivo:** Garantir que `insert_draft_events` e `insert_players` em `src/db/repo.py` realizem limpeza prévia de eventos do `match_id` para evitar duplicações em reingestão.
* **Resultado:** Implementado `DELETE FROM ... WHERE match_id = ?` antes das inserções em lote em ambas as funções.

### B33 — Escala de ingestão e regularização contra Overfitting [CONCLUÍDO]
* **Objetivo:** Coletar lote de partidas e ajustar regularização para mitigar sobreajuste.
* **Resultado:** Ingestão de 500 partidas pro concluída, totalizando 617 partidas e mais de 6.100 jogadores no banco.

---

# Milestone M8 — Deep Learning & Rede Neural em PyTorch (Plano Astra)

### B34 — Dataset Estruturado e NeuralFeatureEncoder [CONCLUÍDO]
* **Objetivo:** Criar `src/features/neural_features.py` com extração SQLite, split temporal 70/15/15 e vocabulários determinísticos com fit exclusivo no treino.
* **Resultado:** Implementado e validado. Gera `encoder.json`, `splits.json` e tensores joblib.

### B35 — DotaNeuralDataset e collate_fn com Tensores Tipados [CONCLUÍDO]
* **Objetivo:** Criar `src/model/neural_dataset.py` com `DotaNeuralDataset` e `collate_dota_batch` respeitando tensores de picks, bans, ranks, MMR e máscaras.
* **Resultado:** Implementado e coberto por testes unitários.

### B36 — Arquitetura DotaWinPredictorNN e Temperature Scaling [CONCLUÍDO]
* **Objetivo:** Criar `src/model/neural_network.py` (Embeddings, Self-Attention, Cross-Attention, Bans, Skill e Cabeça MLP Residual) e `src/model/calibration.py` (Temperature Scaling).
* **Resultado:** Invariância de permutação comprovada por teste unitário e calibração funcionando perfeitamente.

### B37 — Pipeline de Treinamento e Inferência Unificada [CONCLUÍDO]
* **Objetivo:** Criar `src/model/train_nn.py` com Early Stopping e salvar pacote do modelo em `models/draft_win_nn/`; criar `src/model/predict_nn.py` e unificar `predict.py --backend [lr|nn]`.
* **Resultado:** Treinamento finalizado com sucesso (Log Loss teste: 0.6932 vs LR 0.9327). Inferência operando via CLI e dashboard Streamlit.

---

# Milestone M9 — Engenharia de Domínio e Superação de 0.80 AUC (Etapas 1, 2 e 3) [CONCLUÍDO]

### B38 — Etapa 1: Confrontos Diretos 1v1 e Mecânicas de Combate [CONCLUÍDO]
* **Objetivo:** Extrair matriz de matchups 1v1 dos 127 heróis com suavização Bayesiana ($C=20$) e indicadores de BKB-pierce, strong dispel e hard save (15 features).
* **Resultado:** Anti-simetria perfeita validada. Test ROC-AUC saltou para 0.7797 e Acurácia para 70.20%.

### B39 — Etapa 2: Sinergias Intra-Equipe (Química de Aliados) [CONCLUÍDO]
* **Objetivo:** Processar 1.600.000 pares de aliados de 80k partidas públicas do OpenDota para derivar `avg_synergy`, `max_synergy` e `min_synergy` (18 features).
* **Resultado:** Test Log Loss caiu para 0.5577 e Brier Score para 0.1888.

### B40 — Etapa 3: Curvas de Duração e Escalonamento Temporal [CONCLUÍDO]
* **Objetivo:** Mapear win rates dos 127 heróis por faixa de tempo para extrair `early_power`, `late_power` e `scaling_factor` (21 features).
* **Resultado:** Quebra histórica da barreira de 0.80 AUC: Regressão Logística atingiu **0.8065 ROC-AUC** e Rede Neural PyTorch atingiu **74.77% de Acurácia**.

---

# Milestone M10 — Teoria dos Jogos, Análise de Odds (+EV) e Gestão de Banca [CONCLUÍDO]

### B41 — Remoção de Vigorish e Cálculo de Valor Esperado (+EV) [CONCLUÍDO]
* **Objetivo:** Criar `src/betting/engine.py` para descontar margem das casas e calcular $+EV = (P_{\text{modelo}} \times \text{Odd}) - 1$ e Edge real.
* **Resultado:** Implementado e validado por testes unitários com filtro de oportunidade mínima ($+EV \ge 3\%$).

### B42 — Critério de Kelly Fracionário e Proteção de Capital [CONCLUÍDO]
* **Objetivo:** Implementar dimensionamento ótimo $f^* = \frac{+EV}{O - 1}$ com perfil conservador Quarter Kelly (25%) e teto de risco de 5% da banca.
* **Resultado:** Algoritmo matemático implementado impedindo apostas com EV negativo ou sobre-exposição.

### B43 — Simulação Estocástica de Monte Carlo [CONCLUÍDO]
* **Objetivo:** Simular 200 apostas em 100 trajetórias simultâneas para projetar crescimento mediano, pior drawdown e probabilidade de perda >50%.
* **Resultado:** Implementado com plotagem interativa de curvas de patrimônio no Streamlit.

---

# Milestone M11 — Redesign de UX do Dashboard e Deploy em Produção [CONCLUÍDO]

### B44 — Overhaul de UX do Dashboard Streamlit [CONCLUÍDO]
* **Objetivo:** Transformar `src/dashboard/app.py` colocando o Simulador Hero no topo, seleção mútua exclusiva, barra Tug-of-War, 4 Pilares de Explicabilidade e calculadora de apostas embutida.
* **Resultado:** Concluído com storytelling interativo em 4 abas e resolução dinâmica de paths.

### B45 — Expansão e Blindagem da Suíte de Testes [CONCLUÍDO]
* **Objetivo:** Expandir testes unitários automatizados para cobrir todas as novas funcionalidades matemáticas e de apostas.
* **Resultado:** 34 testes automatizados em `pytest` passando em < 7 segundos com 100% de sucesso.

### B46 — Preparação para Nuvem, Docker, Procfile e GitHub [CONCLUÍDO]
* **Objetivo:** Criar `Dockerfile`, `Procfile`, `.streamlit/config.toml`, flexibilizar `requirements.txt`, rastrear pesos pré-treinados e sincronizar no GitHub público.
* **Resultado:** Repositório público sincronizado em `ataidefilipe/dota2req1` pronto para deploy em Streamlit Cloud, Docker, Railway e Render.

---

# Itens Futuros para Expansão (Pós-Produção)

* Motor de Recomendação em Tempo Real ("5th Pick Suggestion Engine").
* Ingestão contínua programada via agendador de tarefas / cron.
* Suporte a handicap de mapas (Bo3 / Bo5).


