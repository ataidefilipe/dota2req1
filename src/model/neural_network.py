import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple

class PickSelfAttentionBlock(nn.Module):
    """
    Bloco de Self-Attention intra-equipe (Sinergias) com pré-norm e conexões residuais.
    Invariante à ordem ao final através de mean pooling.
    """
    def __init__(self, embed_dim: int = 32, num_heads: int = 2, dropout: float = 0.10):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.mha = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)

        self.ln2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 5, embed_dim]
        norm_x = self.ln1(x)
        attn_out, _ = self.mha(norm_x, norm_x, norm_x)
        x = x + self.dropout1(attn_out)

        norm_x = self.ln2(x)
        x = x + self.ffn(norm_x)
        return x # [B, 5, embed_dim]


class CounterCrossAttentionBlock(nn.Module):
    """
    Bloco de Cross-Attention inter-equipes (Counters).
    Queries: heróis da própria equipe; Keys/Values: heróis da equipe adversária.
    """
    def __init__(self, embed_dim: int = 32, num_heads: int = 2, dropout: float = 0.10):
        super().__init__()
        self.ln_q = nn.LayerNorm(embed_dim)
        self.ln_kv = nn.LayerNorm(embed_dim)
        self.cross_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.ln_out = nn.LayerNorm(embed_dim)

    def forward(self, h_team: torch.Tensor, h_opp: torch.Tensor) -> torch.Tensor:
        # h_team: [B, 5, embed_dim]
        # h_opp:  [B, 5, embed_dim]
        q = self.ln_q(h_team)
        kv = self.ln_kv(h_opp)
        attn_out, _ = self.cross_attn(q, kv, kv)
        c = self.ln_out(h_team + self.dropout(attn_out))
        return c # [B, 5, embed_dim]


class BansEncoder(nn.Module):
    """
    Codificador de bans de tamanho variável com masked mean pooling.
    """
    def __init__(self, hero_embed: nn.Embedding, embed_dim: int = 32):
        super().__init__()
        self.hero_embed = hero_embed
        self.mlp_pre = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU()
        )
        # 32 (pooling) + 1 (log(1 + count)) + 1 (observed) = 34
        self.mlp_post = nn.Sequential(
            nn.Linear(embed_dim + 2, embed_dim),
            nn.GELU()
        )

    def forward(self, bans: torch.Tensor, bans_mask: torch.Tensor, bans_available: torch.Tensor) -> torch.Tensor:
        # bans: [B, L]
        # bans_mask: [B, L] (bool)
        # bans_available: [B, 1] (float)
        emb = self.hero_embed(bans) # [B, L, 32]
        h = self.mlp_pre(emb)       # [B, L, 32]

        # Masked mean pooling
        mask_expanded = bans_mask.unsqueeze(-1).float() # [B, L, 1]
        sum_h = (h * mask_expanded).sum(dim=1)          # [B, 32]
        counts = mask_expanded.sum(dim=1).clamp(min=1.0) # [B, 1]
        pooled = sum_h / counts                         # [B, 32]

        # Se nenhum ban foi informado, zera com segurança
        has_any_bans = (mask_expanded.sum(dim=1) > 0).float()
        pooled = pooled * has_any_bans

        log_count = torch.log1p(mask_expanded.sum(dim=1)) # [B, 1]
        features = torch.cat([pooled, log_count, bans_available], dim=-1) # [B, 34]
        out = self.mlp_post(features) # [B, 32]
        return out


class PlayerSkillEncoder(nn.Module):
    """
    Codificador de habilidade/MMR dos jogadores por equipe.
    Combina Rank categórico, MMR contínuo normalizado e resumos estatísticos do time.
    """
    def __init__(self, num_ranks: int, rank_dim: int = 8, hidden_dim: int = 16, dropout: float = 0.10):
        super().__init__()
        self.rank_embed = nn.Embedding(num_ranks, rank_dim, padding_idx=0)

        # Entrada por player: 8 (rank) + 1 (mmr) + 2 (masks) = 11
        self.player_mlp = nn.Sequential(
            nn.Linear(rank_dim + 3, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU()
        )

        # Entrada por equipe: 16 (mean) + 16 (max) + 4 (resumos MMR/cobertura) = 36
        self.team_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 4, hidden_dim),
            nn.GELU()
        )

    def forward(self, ranks: torch.Tensor, mmr: torch.Tensor, skill_masks: torch.Tensor) -> torch.Tensor:
        # ranks: [B, 5]
        # mmr:   [B, 5, 1]
        # skill_masks: [B, 5, 2] (col 0: rank_obs, col 1: mmr_obs)
        B = ranks.shape[0]

        r_emb = self.rank_embed(ranks) # [B, 5, 8]
        p_in = torch.cat([r_emb, mmr, skill_masks], dim=-1) # [B, 5, 11]
        p_feat = self.player_mlp(p_in) # [B, 5, 16]

        # Pooling sobre os 5 jogadores
        mean_p = p_feat.mean(dim=1)    # [B, 16]
        max_p, _ = p_feat.max(dim=1)   # [B, 16]

        # Resumos estatísticos
        mmr_vals = mmr.squeeze(-1) # [B, 5]
        mmr_obs = skill_masks[..., 1] # [B, 5]
        rank_obs = skill_masks[..., 0] # [B, 5]

        # Média e desvio de MMR observado
        obs_count = mmr_obs.sum(dim=1, keepdim=True).clamp(min=1.0) # [B, 1]
        mean_mmr = (mmr_vals * mmr_obs).sum(dim=1, keepdim=True) / obs_count
        var_mmr = (((mmr_vals - mean_mmr) ** 2) * mmr_obs).sum(dim=1, keepdim=True) / obs_count
        std_mmr = torch.sqrt(var_mmr.clamp(min=1e-6))

        # Se nenhum jogador tiver MMR, zera média e desvio
        has_mmr = (mmr_obs.sum(dim=1, keepdim=True) > 0).float()
        mean_mmr = mean_mmr * has_mmr
        std_mmr = std_mmr * has_mmr

        frac_mmr = mmr_obs.mean(dim=1, keepdim=True)  # [B, 1]
        frac_rank = rank_obs.mean(dim=1, keepdim=True) # [B, 1]

        summaries = torch.cat([mean_mmr, std_mmr, frac_mmr, frac_rank], dim=-1) # [B, 4]

        team_in = torch.cat([mean_p, max_p, summaries], dim=-1) # [B, 36]
        k_t = self.team_mlp(team_in) # [B, 16]
        return k_t


class DotaWinPredictorNN(nn.Module):
    """
    Arquitetura de Rede Neural Completa para Dota 2 Win Predictor:
    - Embeddings Compartilhados (Heróis, Patches, Ranks)
    - Self-Attention intra-equipe (Sinergias)
    - Cross-Attention inter-equipes (Counters)
    - Codificador de Bans com Masked Pooling
    - Ramo de Habilidade (MMR normalizado + Ranks + Resumos)
    - Fusão com operações simétricas (Diferença e Hadamard)
    - Cabeça MLP Residual com LayerNorm e Dropout
    """
    def __init__(
        self,
        num_heroes: int,
        num_patches: int,
        num_ranks: int,
        hero_dim: int = 32,
        patch_dim: int = 8,
        rank_dim: int = 8,
        comp_dim: int = 21,
        dropout: float = 0.10
    ):
        super().__init__()
        self.num_heroes = num_heroes
        self.num_patches = num_patches
        self.num_ranks = num_ranks
        self.hero_dim = hero_dim
        self.comp_dim = comp_dim

        # 1. Embeddings
        self.hero_embed = nn.Embedding(num_heroes, hero_dim, padding_idx=0)
        self.patch_embed = nn.Embedding(num_patches, patch_dim, padding_idx=0)

        # 2. Encoders compartilhados
        self.self_attn = PickSelfAttentionBlock(embed_dim=hero_dim, num_heads=2, dropout=dropout)
        self.cross_attn = CounterCrossAttentionBlock(embed_dim=hero_dim, num_heads=2, dropout=dropout)
        self.bans_encoder = BansEncoder(hero_embed=self.hero_embed, embed_dim=hero_dim)
        self.skill_encoder = PlayerSkillEncoder(num_ranks=num_ranks, rank_dim=rank_dim, hidden_dim=16, dropout=dropout)

        # Dimensão por equipe z_T: 32 (self) + 32 (cross) + 32 (bans) + 16 (skill) = 112
        team_dim = hero_dim * 3 + 16
        # Fusão: [z_R; z_D; z_R - z_D; z_R * z_D; E_patch; comp_delta] = 112 * 4 + 8 + comp_dim
        fusion_dim = team_dim * 4 + patch_dim + comp_dim

        # 3. Cabeça de Classificação
        self.head = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Linear(fusion_dim, 64),
            nn.GELU(),
            nn.Dropout(0.20),
            nn.Linear(64, 16),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(16, 1)
        )

    def encode_team(
        self,
        picks: torch.Tensor,
        opp_picks: torch.Tensor,
        bans: torch.Tensor,
        bans_mask: torch.Tensor,
        bans_avail: torch.Tensor,
        ranks: torch.Tensor,
        mmr: torch.Tensor,
        skill_masks: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Embeddings de picks
        e_picks = self.hero_embed(picks)       # [B, 5, 32]
        e_opp_picks = self.hero_embed(opp_picks) # [B, 5, 32]

        # 1. Self-Attention (Sinergias)
        h_team = self.self_attn(e_picks)     # [B, 5, 32]
        h_opp = self.self_attn(e_opp_picks)  # [B, 5, 32]

        # Invariância à permutação via mean pooling
        s_team = h_team.mean(dim=1)          # [B, 32]

        # 2. Cross-Attention (Counters)
        c_team = self.cross_attn(h_team, h_opp).mean(dim=1) # [B, 32]

        # 3. Bans
        b_team = self.bans_encoder(bans, bans_mask, bans_avail) # [B, 32]

        # 4. Habilidade
        k_team = self.skill_encoder(ranks, mmr, skill_masks) # [B, 16]

        z_team = torch.cat([s_team, c_team, b_team, k_team], dim=-1) # [B, 112]
        return z_team, h_team

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Executa a predição para um lote padronizado. Retorna logit escalar [B].
        """
        picks = batch['picks'] # [B, 2, 5]
        rad_picks = picks[:, 0]
        dire_picks = picks[:, 1]

        bans = batch['bans'] # [B, 2, L]
        bans_mask = batch['bans_mask']
        bans_avail = batch['bans_available']
        rad_bans, dire_bans = bans[:, 0], bans[:, 1]
        rad_bmask, dire_bmask = bans_mask[:, 0], bans_mask[:, 1]
        rad_bavail, dire_bavail = bans_avail[:, 0], bans_avail[:, 1]

        ranks = batch['ranks'] # [B, 2, 5]
        mmr = batch['mmr']     # [B, 2, 5, 1]
        skill_masks = batch['skill_masks'] # [B, 2, 5, 2]

        rad_ranks, dire_ranks = ranks[:, 0], ranks[:, 1]
        rad_mmr, dire_mmr = mmr[:, 0], mmr[:, 1]
        rad_smask, dire_smask = skill_masks[:, 0], skill_masks[:, 1]

        # Obter representação dos heróis pós self-attention em paralelo
        e_rad = self.hero_embed(rad_picks)
        e_dire = self.hero_embed(dire_picks)

        h_rad = self.self_attn(e_rad)
        h_dire = self.self_attn(e_dire)

        s_rad = h_rad.mean(dim=1)
        s_dire = h_dire.mean(dim=1)

        # Cross-Attention usando os tensores pré-computados
        c_rad = self.cross_attn(h_rad, h_dire).mean(dim=1)
        c_dire = self.cross_attn(h_dire, h_rad).mean(dim=1)

        # Bans
        b_rad = self.bans_encoder(rad_bans, rad_bmask, rad_bavail)
        b_dire = self.bans_encoder(dire_bans, dire_bmask, dire_bavail)

        # Habilidade
        k_rad = self.skill_encoder(rad_ranks, rad_mmr, rad_smask)
        k_dire = self.skill_encoder(dire_ranks, dire_mmr, dire_smask)

        # Vetores de equipe [B, 112]
        z_rad = torch.cat([s_rad, c_rad, b_rad, k_rad], dim=-1)
        z_dire = torch.cat([s_dire, c_dire, b_dire, k_dire], dim=-1)

        # Patch Embedding
        patch_emb = self.patch_embed(batch['patch']) # [B, 8]

        # Fusão de características: [B, 456 + comp_dim]
        fusion_parts = [
            z_rad,
            z_dire,
            z_rad - z_dire,
            z_rad * z_dire,
            patch_emb
        ]
        if self.comp_dim > 0:
            if 'composition_delta' in batch and batch['composition_delta'] is not None:
                fusion_parts.append(batch['composition_delta'])
            else:
                B = z_rad.shape[0]
                fusion_parts.append(torch.zeros((B, self.comp_dim), device=z_rad.device, dtype=z_rad.dtype))

        u = torch.cat(fusion_parts, dim=-1)

        # Logit final [B]
        logits = self.head(u).squeeze(-1)
        return logits
