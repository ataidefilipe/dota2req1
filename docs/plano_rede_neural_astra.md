## Recomendação executiva

Eu evoluiria o projeto por **adição de um backend neural**, mantendo a Regressão Logística como baseline e fallback:

```text
SQLite
  ├── Features esparsas atuais → Regressão Logística
  └── Dataset estruturado → PyTorch
                              ├── Embeddings
                              ├── Deep Sets
                              ├── Atenção intra/interequipes
                              ├── Contexto de habilidade e patch
                              └── Logit → Calibração → P(Radiant Win)
```

A arquitetura recomendada é uma **rede pequena, invariante à ordem dos picks dentro de cada equipe**, com atenção para modelar interações entre heróis. A sequência completa de draft ficaria como evolução posterior.

**Não substituiria o modelo atual em produção com o volume documentado.** O relatório registra aproximadamente 115 partidas e sobreajuste já na regressão logística. As 5.000 partidas sugeridas na análise são uma meta inicial de coleta, não uma garantia de que Deep Learning será superior.

As especificações abaixo são uma proposta de implementação fundamentada nos arquivos fornecidos, não uma descrição de componentes já existentes.

---

# 1. Condições de entrada: dados, contratos e anti-leakage

Antes da rede, há questões que afetam diretamente a validade do treinamento.

## 1.1. O que o código atual efetivamente permite

| Aspecto | Situação nos arquivos | Consequência |
|---|---|---|
| Ordem do draft | Persistida em `draft_events."order"`, ignorada em `build_features.py` | Podemos extrair sequência futuramente, mas o dataset atual não a representa |
| Habilidade | `rank_tier` e `computed_mmr` persistidos por jogador | Há armazenamento, mas não validação de disponibilidade pré-jogo |
| Herói por jogador | Não persistido em `match_players` | Não podemos associar corretamente MMR a um herói específico |
| Patch | Persistido em `matches.patch_id`; já usado no baseline | Pode entrar como categoria na rede |
| Reingestão | Eventos e jogadores são apagados antes da reinserção | Idempotência local corrigida |
| Atomicidade | Partida, eventos e jogadores usam transações separadas | Ainda pode existir uma partida parcialmente persistida |
| Atualização da partida | `upsert_match()` atualiza apenas `dataset` e `ingested_at` no conflito | Reingestão não corrige `patch_id`, `start_time` e demais metadados |

O código também não garante integralmente o contrato documentado de draft completo:

- rejeita menos de dez picks, mas aceita mais;
- não verifica exatamente cinco picks por lado;
- não exige exatamente dez jogadores;
- não valida unicidade de slots e heróis escolhidos.

Essas validações devem ser corrigidas antes de gerar o dataset neural.

## 1.2. `rank_tier` e `computed_mmr` precisam de aprovação de proveniência

O contrato documentado autoriza lado, picks, bans e patch. Acrescentar habilidade exige ampliar explicitamente esse contrato.

A simples presença dos campos na resposta de `/matches/{match_id}` **não comprova** que:

- representam o valor disponível no encerramento do draft;
- não foram atualizados posteriormente;
- `computed_mmr` não depende de informações posteriores à partida.

O código fornecido apenas copia esses valores; sua semântica temporal não está demonstrada.

**Política proposta:**

1. Auditar origem e temporalidade dos dois campos.
2. Autorizar cada campo separadamente.
3. Enquanto não houver comprovação, tratá-lo como indisponível para o modelo.
4. Nunca preencher partidas antigas com o rank atual de um perfil.
5. Registrar no manifesto de treinamento quais campos foram autorizados.

A arquitetura suportará habilidade, mas o primeiro modelo poderá operar com esse ramo mascarado.

## 1.3. Filtros pós-jogo não são features, mas mudam a população-alvo

O projeto exclui partidas curtas e com abandonos usando informações posteriores ao draft.

Isso não equivale a colocar duração ou abandono na entrada da rede, mas introduz **seleção retrospectiva**. O modelo será avaliado na população:

> Partidas de Captain’s Mode que posteriormente duraram pelo menos 30 minutos e não tiveram abandonos.

Essa população não é identificável integralmente no momento da previsão.

Manteria a regra existente nesta evolução, documentando a limitação. Uma eventual mudança para prever todas as partidas elegíveis no draft deve ser uma decisão de domínio separada, com nova versão do dataset.

---

# 2. Representação de heróis: conjuntos primeiro, sequência depois

## 2.1. Picks como conjuntos por equipe

Para o contrato atual, a entrada natural é:

```text
radiant_picks: 5 heróis
dire_picks:    5 heróis
```

A ordem em que o usuário informa esses heróis não deve alterar a previsão.

Não usaria:

```text
Embedding(hero_1) || ... || Embedding(hero_10)
```

seguido diretamente por uma MLP. Isso atribuiria significado a posições arbitrárias e exigiria que a rede aprendesse todas as permutações relevantes.

Usaria um **embedding compartilhado**:

```python
nn.Embedding(num_hero_tokens, 32, padding_idx=0)
```

Com vocabulário:

- `0`: `PAD`;
- `1`: `UNK`;
- demais índices: heróis observados no treino.

Os IDs externos não devem ser usados diretamente como índices contíguos.

### Heróis desconhecidos

Separar dois casos:

- **Nome/ID inválido:** erro de entrada; não ignorar silenciosamente.
- **Herói válido, mas não observado no treino:** mapear para `UNK` e emitir aviso.

Uma pequena substituição aleatória de tokens por `UNK`, somente durante treinamento, permite que esse embedding seja efetivamente treinado.

## 2.2. Deep Sets como primeira versão neural

Para uma equipe \(T\):

\[
s_T = \frac{1}{5}\sum_{h \in T}\phi(E_h)
\]

Onde:

- \(E_h \in \mathbb{R}^{32}\);
- \(\phi\) é uma MLP compartilhada;
- a média preserva invariância à permutação.

Configuração:

```text
Embedding 32
→ Linear(32, 32)
→ GELU
→ Dropout(0.10)
→ Linear(32, 32)
→ Mean pooling
```

Essa seria a primeira rede treinada. Ela testa a utilidade de representações densas sem introduzir imediatamente todos os parâmetros de atenção.

Uma Deep Sets com cabeça não linear pode aprender interações entre composições. A atenção acrescenta um mecanismo mais explícito de contextualização entre os heróis.

## 2.3. Sinergias: self-attention dentro da equipe

Na versão com atenção:

```text
5 embeddings Radiant → bloco compartilhado de self-attention
5 embeddings Dire    → mesmo bloco
```

Sem embeddings posicionais.

Cada herói passa a ter uma representação condicionada aos quatro aliados. O pooling posterior mantém a invariância à ordem.

Formalmente:

\[
H_R = \operatorname{TeamEncoder}(E_R)
\]

\[
H_D = \operatorname{TeamEncoder}(E_D)
\]

\[
s_R = \operatorname{mean}(H_R),\quad
s_D = \operatorname{mean}(H_D)
\]

Os pesos do encoder são compartilhados entre os lados para reduzir parâmetros e favorecer transferência de padrões.

## 2.4. Counters: cross-attention entre equipes

Acrescentaria atenção cruzada nas duas direções:

\[
C_R = \operatorname{CrossBlock}(H_R,H_D)
\]

\[
C_D = \operatorname{CrossBlock}(H_D,H_R)
\]

- queries: heróis da própria equipe;
- keys/values: heróis adversários;
- pesos compartilhados nas duas direções.

Depois:

\[
c_R = \operatorname{mean}(C_R),\quad
c_D = \operatorname{mean}(C_D)
\]

As duas operações devem usar os mesmos \(H_R,H_D\) originais, evitando dependência da ordem de execução.

Isso permite aprender interações direcionais entre composições, sem manter uma tabela explícita de todas as combinações de heróis.

**Atenção não é explicação causal:** um peso alto não prova, isoladamente, uma sinergia ou counter.

## 2.5. Bans

Não descartaria bans: eles já fazem parte do contrato e do baseline.

Para cada lado, representar os bans como conjunto de tamanho variável:

```text
Embeddings compartilhados dos heróis banidos
→ MLP compartilhada
→ Masked mean pooling
→ Projeção com informação de disponibilidade e quantidade
```

Preservar a semântica atual: `team` identifica o lado que realizou o ban. Não reinterpretar esse campo como “ban que beneficia Radiant” ou “herói adversário”.

É importante distinguir:

- bans fornecidos;
- lista explicitamente vazia;
- bans não informados na inferência.

O último caso deve carregar uma máscara de indisponibilidade, não ser confundido automaticamente com ausência real de bans.

## 2.6. Quando adotar sequência de draft

O banco já permite uma futura representação:

```text
evento = hero_id + team + is_pick + order
```

Uma arquitetura sequencial poderia usar:

```text
Embedding(hero)
+ Embedding(team)
+ Embedding(pick_or_ban)
+ Embedding(position)
→ Transformer Encoder pequeno
→ Masked pooling
```

Mas isso exige:

1. validar `order` por partida;
2. preservar a ordem na extração;
3. incluir eventos ordenados na API de inferência;
4. avaliar a disponibilidade desses dados por origem e patch.

**Nunca inferir a sequência a partir da ordem dos argumentos `--rad` e `--dire`.**

Como a previsão atual ocorre no encerramento do draft, um encoder sobre o draft completo não precisa de máscara causal. Previsão durante o draft seria outro contrato: apenas eventos já disponíveis poderiam entrar.

---

# 3. Integração de habilidade e patch

## 3.1. Habilidade dos dez jogadores como dois conjuntos independentes

Não existe `hero_id` em `match_players`. Portanto, nesta versão:

```text
Conjunto de heróis Radiant ──┐
Conjunto de jogadores Radiant ├─ Fusão da equipe Radiant
Conjunto de bans Radiant ────┘
```

O mesmo vale para Dire.

**Não associar o primeiro jogador ao primeiro pick.** Também não tratar `player_slot` como posição de jogo ou papel estratégico.

A separação por lado deve usar uma função explícita para interpretar `player_slot`, validada contra os payloads ingeridos. `team_id` não deve ser usado como substituto de Radiant/Dire: no código ele pode estar ausente.

Cada jogador será representado por:

```text
rank_tier categórico
computed_mmr normalizado
rank_observed
mmr_observed
```

Rank e MMR do mesmo jogador permanecem juntos na representação, mesmo que a ordem dos cinco jogadores seja irrelevante.

## 3.2. `rank_tier`

Trataria inicialmente o valor como **categoria**, não como uma escala numérica linear.

```python
nn.Embedding(num_rank_tokens, 8)
```

O contexto não documenta suficientemente a codificação para justificar operações como “diferença de 10 pontos de rank”.

Tokens específicos:

- `MISSING`;
- `UNK`;
- categorias observadas no treino.

A máscara de presença permite distinguir um rank desconhecido pelo vocabulário de um valor ausente.

## 3.3. `computed_mmr`

Após auditoria de proveniência:

1. aceitar somente valores finitos e semanticamente válidos;
2. calcular limites de clipping exclusivamente no treino;
3. calcular média e desvio exclusivamente no treino;
4. normalizar;
5. preencher ausentes com zero **depois** da normalização;
6. fornecer máscara de presença.

\[
x_{\text{mmr}} =
\frac{\operatorname{clip}(\text{mmr},q_{0.01},q_{0.99})-\mu_{\text{train}}}
{\max(\sigma_{\text{train}},\epsilon)}
\]

Os percentis são uma configuração inicial proposta, não uma propriedade dos dados atuais.

Para cada equipe, acrescentaria quatro resumos:

- média do MMR normalizado observado;
- desvio populacional do MMR normalizado observado;
- fração de jogadores com MMR;
- fração de jogadores com rank.

Sem observações, média e desvio recebem zero; as frações informam que não houve cobertura.

Isso oferece um caminho explícito para representar nível médio e dispersão sem depender exclusivamente do pooling aprendido.

## 3.4. Disponibilidade parcial e risco de atalhos

A arquitetura deve funcionar com:

- dez jogadores completos;
- valores parcialmente ausentes;
- habilidade totalmente ausente.

Aplicaria mascaramento artificial de habilidade no treinamento para aproximar o cenário atual da CLI, que não exige esses campos.

Ainda assim, ausência de rank/MMR pode funcionar como proxy da origem da partida. Por isso, avaliar separadamente:

- `pro`;
- `ranked_high`;
- partidas com e sem cobertura de habilidade.

A coluna `dataset` fica como metadado de avaliação, não como feature nesta proposta.

## 3.5. `patch_id`

Usaria:

```python
nn.Embedding(num_patch_tokens, 8)
```

Patch é categoria, não número contínuo: não há justificativa no código para assumir distância semântica entre IDs.

Tratamento:

- patch observado no treino: embedding próprio;
- ausente: `MISSING`;
- patch novo: `UNK`;
- mascaramento aleatório de alguns patches no treino para treinar o caminho `UNK`.

Isso não resolve sozinho a mudança de meta. Um patch inédito exige aviso, monitoramento e posterior retreinamento.

Não adicionaria atributos de patch derivados de `patch_ref`: embora a tabela exista, o código fornecido não demonstra seu preenchimento.

---

# 4. Arquitetura neural exata

## 4.1. Tensores de entrada

Considerando batch \(B\):

| Tensor | Shape | Tipo |
|---|---:|---|
| Picks | `[B, 2, 5]` | `torch.long` |
| Bans com padding | `[B, 2, L]` | `torch.long` |
| Máscara de bans válidos | `[B, 2, L]` | `torch.bool` |
| Disponibilidade de bans | `[B, 2, 1]` | `torch.float32` |
| Rank dos jogadores | `[B, 2, 5]` | `torch.long` |
| MMR normalizado | `[B, 2, 5, 1]` | `torch.float32` |
| Máscaras de habilidade | `[B, 2, 5, 2]` | `torch.float32` |
| Patch | `[B]` | `torch.long` |
| Target | `[B]` | `torch.float32` |

Os dois lados têm índice fixo: Radiant e Dire.

## 4.2. Encoder dos picks

Embedding compartilhado de dimensão 32.

Um único bloco de self-attention por equipe, com pesos compartilhados:

```text
X: [B, 5, 32]

A = X + Dropout(
    MultiheadAttention(
        LayerNorm(X),
        LayerNorm(X),
        LayerNorm(X)
    )
)

H = A + Dropout(
    Linear(64, 32)(
        Dropout(
            GELU(
                Linear(32, 64)(LayerNorm(A))
            )
        )
    )
)
```

Configuração:

- `embed_dim=32`;
- `num_heads=2`;
- `batch_first=True`;
- dropout `0.10`;
- sem positional encoding;
- mean pooling → `s_T ∈ R³²`.

## 4.3. Encoder de counters

Um bloco de cross-attention compartilhado:

```text
C_R = LayerNorm(
    H_R + Dropout(
        CrossAttention(LayerNorm(H_R), LayerNorm(H_D), LayerNorm(H_D))
    )
)
```

Dire usa a expressão equivalente com os lados trocados.

Configuração:

- dimensão 32;
- duas heads;
- dropout `0.10`;
- sem FFN adicional nesta primeira versão.

Mean pooling → `c_T ∈ R³²`.

## 4.4. Encoder de bans

```text
HeroEmbedding(32)
→ Linear(32, 32)
→ GELU
→ Masked mean pooling                  # 32
→ Concatenar log(1 + count), observed   # 34
→ Linear(34, 32)
→ GELU                                # b_T: 32
```

Para conjunto vazio ou indisponível, o pooling retorna zero de forma segura, sem divisão por zero.

As transformações são compartilhadas entre Radiant e Dire.

## 4.5. Encoder dos jogadores

Por jogador:

```text
RankEmbedding                         # 8
MMR normalizado                       # 1
Máscaras rank/MMR                     # 2
Concatenar                            # 11

→ Linear(11, 16)
→ GELU
→ Dropout(0.10)
→ Linear(16, 16)
→ GELU
```

Por equipe:

```text
Mean pooling                          # 16
Max pooling                           # 16
Resumos MMR/cobertura                  # 4
Concatenar                            # 36
→ Linear(36, 16)
→ GELU                                # k_T: 16
```

## 4.6. Fusão e classificação

Representação de cada equipe:

\[
z_T = [s_T;c_T;b_T;k_T] \in \mathbb{R}^{112}
\]

Entrada da cabeça:

\[
u = [z_R;z_D;z_R-z_D;z_R\odot z_D;E_{\text{patch}}]
\]

Dimensão:

\[
112 \times 4 + 8 = 456
\]

Cabeça:

```text
LayerNorm(456)
→ Linear(456, 64)
→ GELU
→ Dropout(0.20)
→ Linear(64, 16)
→ GELU
→ Dropout(0.10)
→ Linear(16, 1)
→ squeeze(-1)
```

Saída: um **logit**, não uma probabilidade.

A concatenação ordenada Radiant/Dire permite aprender vantagem de lado. Não imporia:

\[
P(R,D)=1-P(D,R)
\]

porque trocar as composições de lado não é apenas renomear os mesmos participantes: o lado faz parte do problema.

Pelo mesmo motivo, não usaria como augmentation a troca de equipes com inversão automática do rótulo.

## 4.7. Controle de complexidade

Essa configuração fica na ordem de dezenas de milhares de parâmetros, com o total exato dependente dos vocabulários.

Para limitar sobreajuste:

- embeddings pequenos;
- apenas um bloco de self-attention e um de cross-attention;
- pesos compartilhados entre equipes;
- pooling invariante;
- dropout;
- weight decay;
- early stopping;
- ausência de embeddings de `account_id` e `team_id`;
- sem expansão manual de todos os pares de heróis;
- sem estatísticas de vitória calculadas usando o dataset completo.

Preferiria LayerNorm a BatchNorm para evitar dependência das estatísticas de batch em um conjunto pequeno.

---

# 5. Pipeline de dados, treinamento e calibração

## 5.1. Dataset próprio, sem converter a CSR em entrada neural

Não transformaria `X.npz` em matriz densa para alimentar uma MLP.

O formato atual perde:

- associação dos campos por jogador;
- distinção entre ausência e valor;
- estrutura explícita dos conjuntos;
- ordem dos eventos.

Criaria um snapshot estruturado separado:

```text
data/processed_nn/
├── matches.npz
├── draft_events.npz
├── players.npz
└── manifest.json
```

Os arquivos conteriam arrays numéricos, sem objetos Python serializados. O manifesto registra versão, contagens, origem e hash do snapshot.

A extração deve:

1. ler partidas, eventos e jogadores sob uma visão consistente do banco;
2. agrupar por `match_id`;
3. validar o contrato;
4. preservar `"order"` para evolução futura;
5. ordenar por `(start_time, match_id)`;
6. manter target e metadados separados das entradas.

Evitar um join direto eventos × jogadores, que multiplicaria linhas por partida.

### Lista explícita de features

Somente os campos aprovados podem chegar ao `forward()`.

Ficam fora:

- `duration`;
- `leaver_status`;
- `radiant_win`, exceto como target;
- `account_id`;
- IDs de equipes e liga;
- `ingested_at`;
- `parse_version`;
- `start_time` e `match_id`, usados apenas em divisão e rastreabilidade.

## 5.2. Split temporal em quatro partes

Proposta inicial:

```text
60% mais antigos → treino
10% seguintes    → validação
10% seguintes    → calibração
20% mais recentes→ teste final
```

Isso preserva aproximadamente o horizonte final de 20% do baseline atual, mas permite separar seleção e calibração.

Responsabilidades:

| Partição | Uso |
|---|---|
| Treino | Gradientes, vocabulários, normalização |
| Validação | Early stopping e escolha de configuração |
| Calibração | Ajuste da transformação de probabilidades |
| Teste | Avaliação final, sem decisões de ajuste |

Regras:

- nenhum `match_id` em mais de uma partição;
- timestamps empatados permanecem no mesmo bloco;
- proporções podem variar ligeiramente para respeitar fronteiras;
- partidas sem timestamp confiável são rejeitadas;
- nenhum split aleatório estratificado;
- registrar os IDs e limites temporais efetivamente utilizados.

A comparação entre LR e rede deve usar **os mesmos exemplos e fronteiras**, não comparar uma rede nova diretamente com a métrica histórica `0.8032`.

Com volume suficiente, acrescentar validação walk-forward nos períodos anteriores ao teste final.

## 5.3. Preprocessamento ajustado somente no treino

Depois do split:

- construir vocabulários;
- ajustar clipping e normalização de MMR;
- congelar todos os parâmetros;
- transformar validação, calibração e teste.

O catálogo `heroes.json` pode resolver nomes, mas a mera existência de um herói nesse catálogo não significa que seu embedding foi treinado.

Essa separação também evita repetir a construção global de vocabulário anterior ao split usada atualmente pelo baseline.

## 5.4. `Dataset` e `DataLoader`

Contrato proposto:

```text
DraftDataset.__getitem__(index)
    → features tipadas
    → target
```

`match_id` pode acompanhar a amostra para relatórios, mas não entra no modelo.

Usar um `collate_fn` para:

- empilhar os arrays fixos;
- aplicar padding aos bans;
- gerar máscaras;
- preservar shapes consistentes.

Configuração inicial:

```text
batch_size = 64
shuffle = True apenas no treino
drop_last = False
num_workers = 0 inicialmente
pin_memory = True apenas com CUDA
```

Embaralhar dentro do bloco de treino é correto; o que não pode ser embaralhado é a definição das partições temporais.

Não abrir conexão SQLite em `__getitem__`. O treinamento deve consumir um snapshot reproduzível, não um banco mudando durante a ingestão.

## 5.5. Função de perda

```python
nn.BCEWithLogitsLoss()
```

\[
\mathcal{L} =
-\frac{1}{N}\sum_i
\left[
y_i\log\sigma(\ell_i)
+(1-y_i)\log(1-\sigma(\ell_i))
\right]
\]

Não colocar `Sigmoid` antes dessa loss.

Inicialmente:

- sem `pos_weight`;
- sem oversampling de classes;
- sem focal loss;
- sem label smoothing.

O objetivo é estimar probabilidades da população observada. Esses mecanismos alteram o objetivo probabilístico e não são justificados pelos dados apresentados.

## 5.6. Otimizador e scheduler

Configuração inicial proposta:

| Parâmetro | Valor |
|---|---:|
| Otimizador | `AdamW` |
| Learning rate | `1e-3` |
| Weight decay | `1e-3` |
| Máximo de épocas | 150 |
| Early stopping | 12 épocas |
| Gradient clipping | norma global 1.0 |
| Scheduler | `ReduceLROnPlateau` |
| Fator | 0.5 |
| Paciência do scheduler | 4 épocas |
| LR mínimo | `1e-5` |

Aplicar weight decay aos pesos, não a biases e parâmetros de LayerNorm.

Monitorar **log loss de validação**, salvar o melhor checkpoint e restaurá-lo antes da calibração. Esses valores são pontos de partida; não justificam uma busca extensa no dataset atual.

## 5.7. Calibração

Após congelar a rede, ajustar temperature scaling:

\[
p_{\text{cal}} = \sigma\left(\frac{\ell}{T}\right),\quad T>0
\]

- inicializar \(T=1\);
- parametrizar positividade, por exemplo \(T=\exp(t)\);
- ajustar apenas \(t\), na partição de calibração;
- minimizar BCE/log loss;
- salvar \(T\) junto ao modelo.

Temperature scaling reduz ou aumenta confiança, mas não corrige qualquer forma de viés. Uma calibração afim do logit pode ser avaliada depois, se houver dados suficientes.

Com calibração muito pequena ou degenerada, manter \(T=1\) e registrar explicitamente que o modelo não foi calibrado.

Não usaria isotonic regression como primeira opção no volume documentado.

## 5.8. Métricas e critério de promoção

Métricas principais:

- log loss;
- Brier score;
- curva de confiabilidade.

Secundárias:

- ROC-AUC, quando definida;
- acurácia com limiar 0,5;
- métricas por patch, origem e cobertura de habilidade.

Não repetir o comportamento atual de registrar log loss zero quando uma partição possui uma única classe. Log loss continua calculável com o conjunto de classes explicitado; AUC, por sua vez, deve ser marcada como indisponível.

Comparações obrigatórias:

1. probabilidade constante igual à frequência de vitória Radiant no treino;
2. LR regularizada;
3. Deep Sets;
4. Deep Sets com self-attention;
5. self-attention + cross-attention;
6. habilidade autorizada ligada/desligada.

Para separar ganho de arquitetura de ganho de informação, a LR também deve receber resumos de habilidade autorizados em uma comparação equivalente.

A rede só deve substituir o baseline se apresentar melhoria consistente em períodos futuros, não apenas um pico de acurácia em poucas dezenas de partidas.

---

# 6. Integração modular no `src/`

## 6.1. Estrutura proposta

Manteria o padrão atual de módulos executáveis, sem introduzir frameworks de treinamento ou infraestrutura adicional.

```text
src/
├── db/
│   ├── schema.py                 # Alterações mínimas, se necessárias
│   └── repo.py                   # Persistência atômica e extração consistente
├── features/
│   ├── filters.py                # Reforçar contrato de qualidade
│   ├── build_features.py         # Baseline preservado
│   ├── dataset_stats.py          # Cobertura dos novos campos
│   └── neural_features.py        # Snapshot, contratos e encoder fit/transform
└── model/
    ├── train.py                  # LR preservada
    ├── predict.py                # CLI existente + seleção de backend
    ├── neural_dataset.py         # Dataset e collate_fn
    ├── neural_network.py         # nn.Module e configuração tipada
    ├── train_nn.py               # Treino e avaliação
    ├── predict_nn.py             # Carregamento e inferência neural
    └── calibration.py            # Temperature scaling
```

Responsabilidades importantes:

- `neural_features.py` não conhece detalhes da rede;
- `neural_network.py` não conhece SQL, JSON ou nomes de heróis;
- `neural_dataset.py` não faz consultas ao banco;
- `train_nn.py` orquestra, sem duplicar encoding;
- `predict_nn.py` usa exatamente o encoder persistido no treinamento.

Utilizar `dataclass(frozen=True)` para configurações e registros de domínio, `TypedDict` quando apropriado para batches, tipos explícitos de retorno e validação de shape/dtype nas fronteiras.

## 6.2. Artefato autocontido

```text
models/draft_win_nn/
├── weights.pt
├── model_config.json
├── encoder.json
├── calibration.json
└── manifest.json
```

Conteúdo:

- `weights.pt`: `state_dict`;
- `model_config.json`: variante e dimensões;
- `encoder.json`: vocabulários, normalização e política de ausentes;
- `calibration.json`: temperatura e status;
- `manifest.json`: versão do schema, snapshot, split, seed, dependências e campos autorizados.

Salvar `state_dict`, não uma instância Python completa da rede.

A inferência neural não deve depender do `feature_map.json` do baseline nem de um diretório de dados processados que possa ser sobrescrito.

## 6.3. Compatibilidade de inferência

Preservar:

```python
predict(radiant_heroes, dire_heroes, radiant_bans=None, dire_bans=None, patch_id=None)
```

Adicionar habilidade como parâmetros opcionais, preferencialmente keyword-only, com uma coleção de cinco registros por lado.

Exemplo de item:

```json
{
  "rank_tier": 80,
  "computed_mmr": 6500
}
```

`null` representa campo ausente. A ordem dos cinco jogadores não significa associação aos heróis.

Na CLI:

```bash
python -m src.model.predict --backend nn --json draft.json
```

O backend padrão permanece `lr` até aprovação da rede.

Há uma correção necessária no parser atual: `--rad` e `--dire` são obrigatórios mesmo quando `--json` é fornecido. A validação deve ocorrer após o parsing:

- JSON válido; **ou**
- os dois conjuntos de picks.

Inferência neural:

```text
Validar entrada
→ Resolver nomes
→ Encoder congelado
→ model.eval()
→ torch.inference_mode()
→ Logit
→ Temperatura
→ Sigmoid
→ float
```

Avisos de patch/herói desconhecido e habilidade indisponível devem ser observáveis, sem transformar entradas malformadas em previsão silenciosa.

---

# 7. Plano de implementação por etapas

## Etapa 1 — Consolidar contrato e qualidade

1. Atualizar documentação anti-leakage.
2. Auditar os campos de habilidade.
3. Exigir dez picks, cinco por lado e heróis escolhidos únicos.
4. Exigir dez jogadores, slots únicos e cinco por lado.
5. Validar target, timestamp e IDs.
6. Persistir partida, eventos e jogadores em uma transação.
7. Corrigir os campos atualizados por `upsert_match()`.

**Entrega:** banco utilizável por ambos os modelos, sem alterar ainda a inferência.

## Etapa 2 — Medir e ampliar a base

Expandir `dataset_stats.py` com:

- cobertura de rank/MMR;
- cobertura por origem e patch;
- quantidade de partidas válidas;
- valores inválidos;
- integridade dos drafts;
- distribuição temporal.

A ingestão atual lê uma página por execução. Portanto, `--limit 5000` não implementa backfill de 5.000 partidas.

Adicionar paginação/backfill explícitos, separados do checkpoint de atualização incremental. Medir a taxa de aceitação de `ranked_high` sob o filtro de Captain’s Mode; não pressupor que essa origem fornecerá volume suficiente.

**Entrega:** evidência de que há dados suficientes e compatíveis com o domínio.

## Etapa 3 — Dataset neural e baseline comparável

1. Implementar snapshot.
2. Implementar split temporal persistido.
3. Implementar encoder com `fit` somente no treino.
4. Treinar novamente a LR com regularização selecionada na validação.
5. Usar os mesmos IDs nas comparações.

**Entrega:** base experimental reproduzível.

## Etapa 4 — Deep Sets mínima

Treinar embeddings, pooling, bans, patch e habilidade autorizada, sem atenção.

**Objetivo:** verificar se o ganho já aparece com a solução mais simples.

## Etapa 5 — Atenção incremental

1. Adicionar self-attention.
2. Medir ganho.
3. Adicionar cross-attention.
4. Medir ganho adicional.
5. Remover componentes sem benefício consistente.

A arquitetura completa é uma candidata, não uma obrigação de implantação.

## Etapa 6 — Calibração e artefato

1. Restaurar melhor checkpoint.
2. Ajustar temperatura.
3. Avaliar o pipeline congelado no teste.
4. Gerar artefato autocontido.
5. Registrar métricas e limitações.

## Etapa 7 — Integração e promoção

1. Adicionar backend neural opcional.
2. Preservar comandos antigos.
3. Executar previsões paralelas com LR e rede.
4. Promover somente após melhoria consistente.
5. Manter rollback explícito para o baseline.

---

# 8. Testes indispensáveis

| Área | Teste |
|---|---|
| Persistência | Reingestão mantém contagens; falha intermediária desfaz a transação |
| Qualidade | Rejeitar picks extras, lados desequilibrados e jogadores duplicados |
| Split | IDs disjuntos e fronteiras temporais corretas |
| Anti-leakage | Alterar duração/KDA/metadados proibidos não altera os tensores de entrada |
| Normalização | Estatísticas não mudam ao modificar apenas validação/teste |
| Permutação | Reordenar picks, bans ou jogadores dentro de um lado mantém o logit em `eval()` |
| Habilidade | Permutar registros completos de jogadores preserva pares rank/MMR |
| Ausência | Todos os ranks/MMRs ausentes não produzem `NaN` |
| Bans | Lista vazia e indisponível são tratadas sem erro |
| Unknown | Herói válido inédito e patch novo têm comportamento explícito |
| Serialização | Modelo recarregado reproduz probabilidades |
| Inferência | Encoding de treino e produção é idêntico |
| Calibração | Ajustar temperatura não altera pesos da rede |
| Compatibilidade | CLI antiga funciona; JSON não exige picks duplicados em flags |

## Decisão final

A evolução mais adequada para o `dota2req1` é:

> **Embeddings compartilhados + Deep Sets como ponto de partida; self-attention para sinergias e cross-attention para counters como extensões validadas; habilidade em conjuntos separados de jogadores; patch categórico; split temporal com calibração independente.**

O maior risco atual não é escolher uma rede pouco sofisticada. É treinar uma rede sofisticada sobre poucas partidas, campos de habilidade sem proveniência temporal comprovada e uma avaliação que confunda ganho real com sobreajuste.