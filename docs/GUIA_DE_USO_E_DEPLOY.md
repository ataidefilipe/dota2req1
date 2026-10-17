# 🛠️ Guia de Uso, Operação e Deploy — Aegis Draft

Este guia fornece instruções operacionais passo a passo para desenvolvedores, cientistas de dados e operadores que desejam executar, testar, estender ou fazer o deploy em produção do **Aegis Draft**.

---

## 1. Requisitos de Sistema

* **Linguagem:** Python 3.10, 3.11, 3.12 ou 3.13.
* **Sistema Operacional:** Windows 10/11, Ubuntu 20.04+ (ou qualquer distribuição Linux moderna) e macOS.
* **Memória RAM:** Mínimo de 2 GB (recomendado 4 GB).
* **Armazenamento:** ~500 MB de espaço livre em disco.

---

## 2. Instalação e Configuração Local

### Passo 1: Obter o Código-Fonte
```bash
git clone https://github.com/ataidefilipe/dota2req1.git
cd dota2req1
```

### Passo 2: Criar e Ativar o Ambiente Virtual
* **No Windows (PowerShell):**
  ```powershell
  python -m venv .venv
  .venv\Scripts\Activate.ps1
  ```
* **No Linux ou macOS:**
  ```bash
  python3 -m venv .venv
  source .venv/bin/activate
  ```

### Passo 3: Instalar as Dependências
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Passo 4: Configurar Variáveis de Ambiente (Opcional)
Se você for apenas executar o dashboard e simular drafts, **nenhuma chave de API é necessária** (o repositório já contém os pesos treinados e as matrizes do Patch 60).

Se desejar realizar ingestões adicionais de partidas via OpenDota:
```bash
cp .env.example .env
```
Edite o arquivo `.env` para inserir sua chave:
```env
OPENDOTA_API_KEY=sua_chave_aqui
OPENDOTA_RPS=1.0
SQLITE_PATH=./data/opendota.sqlite
PUBLICMATCHES_MIN_RANK=70
TARGET_PATCH=60
```

---

## 3. Execução da Aplicação

### 3.1 Inicializar o Dashboard Interativo (Streamlit)
```bash
streamlit run src/dashboard/app.py
```
O painel estará acessível no navegador em `http://localhost:8501`.

#### O que você pode fazer no Dashboard:
1. **Experimentar Presets:** Clique nos botões táticos (*Late-Game vs Zoo Push*, *Teamfight vs Pickoff*, *Pro Balanced*) para carregar confrontos emblemáticos instantaneamente.
2. **Montar Drafts Personalizados:** Selecione 5 heróis para o Radiant e 5 para o Dire no catálogo suspenso (o sistema bloqueia automaticamente heróis já escolhidos).
3. **Calcular Probabilidades:** Observe a barra animada de puxar a corda (*tug-of-war*) com as chances reais de vitória.
4. **Analisar os 4 Pilares de Explicabilidade:**
   - Veja quais counters individuais estão decidindo o duelo.
   - Analise se a sua equipe possui química ou atrito de papéis.
   - Verifique quem possui a vantagem do relógio de jogo (Early vs Late game).
   - Inspecione as ferramentas de teamfight (*BKB Pierces*, *Strong Dispels*, *Hard Saves*).
5. **Simular Apostas (+EV):** Digite as cotações oferecidas pela casa de aposta (ex: Radiant 1.75 / Dire 2.10) e confira a recomendação em reais de aposta calculada pelo Critério de Kelly Fracionário.

---

### 3.2 Executar a Suíte de Testes Automatizados
O projeto conta com **34 testes unitários e de integração** garantindo conformidade matemática, integridade de dados e proteção contra quebras de código:

```bash
pytest
```
Para ver a saída detalhada com tempo de execução individual:
```bash
pytest -v
```

---

### 3.3 Inferência Rápida via Linha de Comando (CLI)
Você pode calcular a probabilidade de vitória de qualquer confronto diretamente pelo terminal sem abrir o navegador:

```bash
python -m src.model.predict_nn \
  --radiant "Anti-Mage" "Lion" "Tidehunter" "Invoker" "Mirana" \
  --dire "Slark" "Crystal Maiden" "Axe" "Shadow Fiend" "Pudge"
```

---

## 4. Pipeline de Dados & Retreinamento de Modelos

Caso você queira atualizar a base de dados com partidas adicionais ou retreinar a inteligência artificial do zero:

### 4.1 Ingestão de Novas Partidas
```bash
# Ingerir partidas profissionais do Patch 60 via OpenDota Explorer
python -m src.opendota.ingest --patch 60 --complete-patch --limit 200

# Ingerir partidas de torneios amadores/comunitários em Captain's Mode
python -m src.opendota.ingest --patch 60 --community-patch --limit 300
```

### 4.2 Download e Atualização das Constantes da Valve
```bash
python -m src.opendota.fetch_constants
```
*Atualiza atributos de heróis, habilidades, magias com BKB-pierce, matrizes de matchup e durações.*

### 4.3 Retreinamento dos Modelos

1. **Extração de Features Tabulares & Treino da Regressão Logística:**
   ```bash
   python -m src.features.build_features
   python -m src.model.train
   ```

2. **Codificação Neural & Treino da Rede PyTorch:**
   ```bash
   python -m src.features.neural_features
   python -m src.model.train_nn
   ```
*Os novos pesos, scalers e parâmetros de calibração serão salvos automaticamente em `models/`.*

---

## 5. Guia Completo de Deploy em Nuvem

### 5.1 Deploy no Streamlit Community Cloud (100% Gratuito)
Esta é a forma mais rápida de colocar a aplicação online com um link público compartilhável.

1. Acesse **[share.streamlit.io](https://share.streamlit.io/)** e autentique-se com sua conta do GitHub.
2. Clique no botão azul **"New app"**.
3. Preencha as configurações do repositório:
   - **Repository:** `ataidefilipe/dota2req1`
   - **Branch:** `master`
   - **Main file path:** `src/dashboard/app.py`
4. *(Opcional)* Em **Advanced settings**, você pode selecionar a versão do Python (3.11 recomendada).
5. Clique em **"Deploy!"**.
6. O Streamlit Cloud instalará as dependências do `requirements.txt` e iniciará o dashboard em menos de 2 minutos.

> [!TIP]
> Se você precisar forçar uma reinicialização limpa do app, clique em **"Manage app"** no canto inferior direito e selecione **"Reboot app"**.

---

### 5.2 Deploy com Docker Container
O projeto inclui um [`Dockerfile`](../Dockerfile) pronto para produção:

```bash
# 1. Construir a imagem Docker
docker build -t aegis-draft-dota2 .

# 2. Executar o container na porta 8501
docker run -d \
  --name aegis-draft \
  -p 8501:8501 \
  --restart unless-stopped \
  aegis-draft-dota2
```
Acesse `http://localhost:8501` ou o IP do seu servidor.

---

### 5.3 Deploy em Plataformas PaaS (Railway / Render / Heroku)
O repositório já contém o arquivo [`Procfile`](../Procfile), configurado com a diretiva:
```text
web: streamlit run src/dashboard/app.py --server.port $PORT --server.address 0.0.0.0
```

1. **No Railway:**
   - Crie um novo projeto e selecione **"Deploy from GitHub repo"**.
   - Aponte para `ataidefilipe/dota2req1`.
   - O Railway detectará o `Procfile` ou `Dockerfile` automaticamente e criará a URL pública.
2. **No Render:**
   - Crie um **"Web Service"**.
   - Conecte o repositório `ataidefilipe/dota2req1`.
   - Selecione o ambiente **Python 3**.
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `streamlit run src/dashboard/app.py --server.port $PORT --server.address 0.0.0.0`

---

## 6. Solução de Problemas Comuns (Troubleshooting)

| Sintoma / Erro | Causa Provável | Solução Recomendada |
| :--- | :--- | :--- |
| `ModuleNotFoundError: No module named 'src'` | O Streamlit foi iniciado a partir de uma subpasta e não encontrou a raiz do projeto. | O `src/dashboard/app.py` já injeta o `PROJECT_ROOT` dinamicamente no `sys.path`. Certifique-se de executar o comando da raiz: `streamlit run src/dashboard/app.py`. |
| `ModuleNotFoundError: No module named 'dotenv'` | Execução em ambiente de nuvem limpo sem o pacote instalado. | Atualizamos `src/config.py` para tornar o `dotenv` opcional. Em nuvem, utilize as variáveis de ambiente nativas da plataforma. |
| Gráficos do SQLite aparecem vazios | O repositório foi clonado sem o banco `opendota.sqlite`. | O simulador e os modelos funcionam 100% de forma desacoplada via arquivos JSON de constantes e pesos pré-treinados em `models/`. Para ver o histórico de picks/bans no banco, execute a ingestão (Seção 4.1). |
| `Rate limit exceeded (429)` na OpenDota | A chave da API gratuita atingiu o limite de 60 requisições/minuto. | O cliente `OpenDotaClient` já possui backoff exponencial embutido. Se persistir, aguarde 60 segundos ou configure uma chave paga em `OPENDOTA_API_KEY`. |
