# Catálogo e Estrutura de Constantes do Dota 2 — Patch 59 (v7.40)

Este documento descreve detalhadamente as constantes estáticas de domínio de Dota 2 baixadas da OpenDota e persistidas em `data/constants/`. Ele serve como especificação formal dos metadados para orientar o desenho de **features de composição e draft** para o **Dota 2 Win Predictor**.

---

## 1. Visão Geral dos Três Arquivos Baixados

| Arquivo | Origem | Registros | Tamanho | Descrição Principal |
| :--- | :--- | :--- | :--- | :--- |
| **`data/constants/heroes.json`** | `/heroStats` | 127 heróis | ~255 KB | Atributos base, ganhos por nível, HP, Mana, Armadura, Alcance, Funções e Velocidade. |
| **`data/constants/hero_abilities.json`** | `/constants/hero_abilities` | 127 heróis | ~259 KB | Mapeamento herói $\to$ magias ativas, passivas, facetas e árvores de talentos. |
| **`data/constants/abilities.json`** | `/constants/abilities` | 3.084 magias | ~2.04 MB | Catálogo detalhado de magias: dano, custo de mana, cooldown, pierce BKB, tipo de dano e stuns. |

---

## 2. Estrutura e Schemas Detalhados

### 2.1. `heroes.json` (Atributos e Estatísticas Base)

Cada registro do array de 127 heróis contém os seguintes campos numéricos e categóricos:

```json
{
  "id": 1,
  "name": "npc_dota_hero_antimage",
  "localized_name": "Anti-Mage",
  "primary_attr": "agi",
  "attack_type": "Melee",
  "roles": ["Carry", "Escape", "Nuker"],
  "base_str": 21,
  "str_gain": 1.6,
  "base_agi": 24,
  "agi_gain": 2.8,
  "base_int": 12,
  "int_gain": 1.8,
  "base_health": 120,
  "base_health_regen": 0.25,
  "base_mana": 75,
  "base_mana_regen": 0.0,
  "base_armor": 1.0,
  "base_mr": 25,
  "base_attack_min": 29,
  "base_attack_max": 33,
  "attack_range": 150,
  "attack_rate": 1.4,
  "move_speed": 310,
  "turn_rate": 0.6,
  "cm_enabled": true,
  "legs": 2
}
```

#### Estatísticas Agregadas do Pool (127 Heróis):
* **Atributo Primário (`primary_attr`):**
  * `str` (Força): 36 heróis (28.3%)
  * `agi` (Agilidade): 35 heróis (27.6%)
  * `int` (Inteligência): 34 heróis (26.8%)
  * `all` (Universal): 22 heróis (17.3%)
* **Tipo de Ataque (`attack_type`):**
  * `Ranged` (Distância): 65 heróis (51.2%)
  * `Melee` (Corpo a corpo): 62 heróis (48.8%)
* **Funções Oficiais da Valve (`roles`):**
  * `Disabler`: 95 heróis
  * `Nuker`: 89 heróis
  * `Carry`: 72 heróis
  * `Initiator`: 55 heróis
  * `Escape`: 53 heróis
  * `Durable`: 50 heróis
  * `Support`: 48 heróis
  * `Pusher`: 29 heróis

---

### 2.2. `hero_abilities.json` (Mapeamento de Habilidades e Facetas)

Mapeia a chave de recurso do herói para suas magias, árvores de talentos e facetas:

```json
{
  "npc_dota_hero_antimage": {
    "abilities": [
      "antimage_mana_break",
      "antimage_blink",
      "antimage_counterspell",
      "generic_hidden",
      "antimage_persectur",
      "antimage_mana_void"
    ],
    "talents": [
      {"name": "special_bonus_hp_regen_3", "level": 1},
      {"name": "special_bonus_unique_antimage_manavoid_aoe", "level": 1},
      {"name": "special_bonus_unique_antimage_5", "level": 2},
      {"name": "special_bonus_unique_antimage_6", "level": 2},
      {"name": "special_bonus_unique_antimage_3", "level": 3},
      {"name": "special_bonus_unique_antimage_8", "level": 3},
      {"name": "special_bonus_unique_antimage", "level": 4},
      {"name": "special_bonus_unique_antimage_2", "level": 4}
    ],
    "facets": [
      {
        "id": 0,
        "name": "antimage_magebanes_mirror",
        "title": "Magebane's Mirror"
      },
      {
        "id": 1,
        "name": "antimage_mana_thirst",
        "title": "Mana Thirst"
      }
    ]
  }
}
```

---

### 2.3. `abilities.json` (Catálogo Detalhado de Magias e Efeitos)

Dicionário indexado pelo nome da habilidade. Contém informações mecânicas de cada magia:

```json
{
  "enigma_black_hole": {
    "dname": "Black Hole",
    "behavior": ["Point Target", "Channeled", "AOE"],
    "dmg_type": "Pure",
    "bkbpierce": "Yes",
    "desc": "SUMMONS A VORTEX THAT PULLS ENEMY UNITS IN, DAMAGING AND DISABLING THEM.",
    "attrib": [
      {"key": "duration", "header": "DURATION:", "value": "4.0"},
      {"key": "radius", "header": "RADIUS:", "value": "420"},
      {"key": "damage", "header": "DAMAGE:", "value": ["100", "150", "200"]}
    ],
    "mc": ["300", "400", "500"],
    "cd": ["200", "180", "160"]
  },
  "sven_storm_bolt": {
    "dname": "Storm Hammer",
    "behavior": ["Unit Target", "AOE"],
    "dmg_type": "Magical",
    "bkbpierce": "No",
    "attrib": [
      {"key": "bolt_stun_duration", "header": "STUN DURATION:", "value": ["1.4", "1.6", "1.8", "2.0"]},
      {"key": "bolt_aoe", "header": "RADIUS:", "value": "255"},
      {"key": "bolt_damage", "header": "DAMAGE:", "value": ["95", "170", "245", "320"]}
    ],
    "mc": ["110", "120", "130", "140"],
    "cd": ["16", "14", "12", "10"]
  }
}
```

#### Estatísticas do Catálogo de Magias (3.084 Habilidades):
* **Tipo de Dano (`dmg_type`):**
  * `Magical`: 422 habilidades com dano mágico
  * `Physical`: 99 habilidades com dano físico
  * `Pure`: 44 habilidades com dano puro (ignora armadura e resistência mágica)
* **Atravessa Imunidade a Magia / BKB (`bkbpierce`):**
  * `No`: 455 habilidades (bloqueadas por BKB)
  * `Yes`: 193 habilidades (**atravessam BKB** — habilidades críticas de counter em lutas de equipe)
* **Comportamentos Relevantes (`behavior`):**
  * `Passive`: 432 passivas
  * `AOE` (Área de Efeito): 153 habilidades
  * `Channeled` (Canalizadas): 46 habilidades
  * `Instant Cast` (Conjuração Imediata): 156 habilidades

---

## 3. Espaço de Hipóteses de Features para Previsão de Vitória

A partir desses dados estáticos, podemos derivar vetores de composição $V \in \mathbb{R}^k$ para cada equipe (Radiant e Dire) no draft e calcular a diferença diferencial $\Delta = V_{\text{Rad}} - V_{\text{Dire}}$:

### Categoria A: Robustez e Escalação de Atributos (Base & Lategame)
1. **Ganho Total de Atributos:** $\sum_{h \in T} (\text{str\_gain} + \text{agi\_gain} + \text{int\_gain})$. Mede o potencial intrínseco de escalação para o final de jogo.
2. **HP Efetivo Base do Time:** $\sum_{h \in T} (\text{base\_health} \times (1 + 0.06 \times \text{base\_armor}))$. Mede a resistência a trocas de dano na fase de rotas (*laning phase*).
3. **Regeneração e Sustentação:** $\sum_{h \in T} \text{base\_health\_regen}$ e $\sum \text{base\_mana\_regen}$.

### Categoria B: Equilíbrio Tático e Geometria de Draft
4. **Balanço Melee vs Ranged:** $\text{count}(\text{Ranged}) - \text{count}(\text{Melee})$. Times com 5 Melees sofrem desvantagem severa de posicionamento e assédio de rotas.
5. **Alcance Médio de Ataque:** $\text{mean}(\text{attack\_range})$.
6. **Velocidade de Movimento Média:** $\text{mean}(\text{move\_speed})$. Permite rotações mais rápidas de suporte e controle de mapa.

### Categoria C: Perfil Funcional (Roles)
7. **Disabler Score:** $\text{count}(h \text{ com 'Disabler'} \in \text{roles})$.
8. **Initiator Score:** $\text{count}(h \text{ com 'Initiator'} \in \text{roles})$.
9. **Carry / Greed Index:** $\text{count}(h \text{ com 'Carry'} \in \text{roles})$. Composições com 4 carries disputam recursos limitados de mapa.
10. **Durable Score:** $\text{count}(h \text{ com 'Durable'} \in \text{roles})$.

### Categoria D: Mecânicas de Magias (Counters e Spells)
11. **BKB Pierce Count:** Contagem de magias no draft que atravessam BKB (`bkbpierce == 'Yes'`). Times sem nenhum disable que fura BKB perdem controle no lategame contra carries de BKB.
12. **Pure Damage Count:** Contagem de magias de dano puro (counters naturais contra times de alta armadura e heróis de força/agilidade).
13. **Cooldowns de Teamfight / Ultimates Longos:** Contagem de magias com cooldown $> 100$s (ex: Black Hole, Ravage, Chronosphere). Indica dependência de janelas de luta.
