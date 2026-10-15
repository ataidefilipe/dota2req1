"""Módulo matemático para precificação de apostas, análise de valor esperado (+EV)
e gestão quantitativa de banca com Critério de Kelly Fracionário.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import numpy as np


@dataclass
class BetOpportunity:
    """Representação estruturada de uma oportunidade de aposta."""
    side: Optional[str]  # 'radiant', 'dire' ou None
    side_name: Optional[str]
    odd: float
    model_prob: float
    market_prob_fair: float
    ev: float  # Expected Value relativo (ex: 0.08 = +8% EV)
    edge: float  # p_model - p_market_fair (ex: 0.05 = +5% de vantagem líquida)
    full_kelly: float  # Fração ótima teórica pura (0 a 1)
    fractional_kelly: float  # Fração ajustada pelo perfil de risco
    recommended_stake: float  # Valor em moeda (R$)
    stake_percentage: float  # Porcentagem da banca (0 a 100%)
    is_actionable: bool  # True se cumprir todos os critérios de risco (+EV >= min_ev)
    reason: str  # Explicação humana do veredito


@dataclass
class MonteCarloSimulation:
    """Resultado estatístico de simulações de trajetória de banca."""
    initial_bankroll: float
    median_final_bankroll: float
    percentile_10: float
    percentile_90: float
    max_drawdown_median: float
    risk_of_ruin: float  # Probabilidade da banca cair mais de 50%
    trajectories_sample: list  # Amostra de 5 caminhos para plotagem


def remove_vigorish(odd_rad: float, odd_dire: float) -> Tuple[float, float]:
    """Remove a margem da casa (vigorish / overround) para obter as probabilidades
    justas implícitas do mercado.

    Args:
        odd_rad: Odd decimal do Radiant (ex: 1.85)
        odd_dire: Odd decimal do Dire (ex: 2.05)

    Returns:
        Tuple[float, float]: (p_fair_radiant, p_fair_dire)
    """
    if odd_rad <= 1.0 or odd_dire <= 1.0:
        raise ValueError("As odds devem ser estritamente maiores que 1.00.")

    imp_rad = 1.0 / odd_rad
    imp_dire = 1.0 / odd_dire
    overround_sum = imp_rad + imp_dire

    p_fair_rad = imp_rad / overround_sum
    p_fair_dire = imp_dire / overround_sum
    return p_fair_rad, p_fair_dire


def calculate_ev(p_model: float, odd: float) -> float:
    """Calcula o Valor Esperado (Expected Value) por unidade monetária apostada.

    Fórmula: EV = (P_model * O) - 1

    Args:
        p_model: Probabilidade estimada pelo modelo calibrado (0.0 a 1.0)
        odd: Odd decimal oferecida pela casa (> 1.0)

    Returns:
        float: Retorno esperado relativo (-1.0 a +inf). Ex: 0.05 significa +5% de EV.
    """
    if odd <= 1.0:
        return -1.0
    p_clamped = max(0.0, min(1.0, p_model))
    return (p_clamped * odd) - 1.0


def calculate_kelly_fraction(p_model: float, odd: float) -> float:
    """Calcula a fração ótima teórica da banca segundo o Critério de Kelly puro.

    Fórmula: f* = (P * O - 1) / (O - 1) = EV / (O - 1)

    Args:
        p_model: Probabilidade de vitória do modelo (0.0 a 1.0)
        odd: Odd decimal (> 1.0)

    Returns:
        float: Fração recomendada da banca (0.0 se EV <= 0).
    """
    if odd <= 1.0:
        return 0.0
    ev = calculate_ev(p_model, odd)
    if ev <= 0.0:
        return 0.0
    b = odd - 1.0
    f_star = ev / b
    return max(0.0, min(1.0, f_star))


def evaluate_betting_opportunity(
    p_model_radiant: float,
    odd_radiant: float,
    odd_dire: float,
    bankroll: float = 1000.0,
    kelly_multiplier: float = 0.25,
    max_stake_pct: float = 0.05,
    min_ev: float = 0.03,
) -> BetOpportunity:
    """Avalia o confronto entre Radiant e Dire comparando o modelo calibrado com as odds.

    Aplica remoção de vig, cálculo de +EV, critério de Kelly fracionário e travas de segurança.

    Args:
        p_model_radiant: Probabilidade estimada de vitória do Radiant (0 a 1)
        odd_radiant: Odd da casa para vitória do Radiant
        odd_dire: Odd da casa para vitória do Dire
        bankroll: Valor total da banca disponível (em R$)
        kelly_multiplier: Fração de Kelly (0.25 = Quarter Kelly - padrão do mercado)
        max_stake_pct: Teto máximo por aposta em relação à banca (0.05 = max 5%)
        min_ev: Limiar mínimo de +EV para acionar recomendação (0.03 = +3% EV)

    Returns:
        BetOpportunity com o veredito completo.
    """
    if odd_radiant <= 1.0 or odd_dire <= 1.0:
        return BetOpportunity(
            side=None,
            side_name=None,
            odd=0.0,
            model_prob=0.0,
            market_prob_fair=0.0,
            ev=0.0,
            edge=0.0,
            full_kelly=0.0,
            fractional_kelly=0.0,
            recommended_stake=0.0,
            stake_percentage=0.0,
            is_actionable=False,
            reason="Odds inválidas (devem ser maiores que 1.00).",
        )

    p_fair_rad, p_fair_dire = remove_vigorish(odd_radiant, odd_dire)
    p_model_dire = 1.0 - p_model_radiant

    ev_rad = calculate_ev(p_model_radiant, odd_radiant)
    ev_dire = calculate_ev(p_model_dire, odd_dire)

    edge_rad = p_model_radiant - p_fair_rad
    edge_dire = p_model_dire - p_fair_dire

    # Escolher o lado com maior Valor Esperado
    if ev_rad >= ev_dire:
        best_side = "radiant"
        best_name = "Radiant"
        best_ev = ev_rad
        best_odd = odd_radiant
        best_p_model = p_model_radiant
        best_p_fair = p_fair_rad
        best_edge = edge_rad
    else:
        best_side = "dire"
        best_name = "Dire"
        best_ev = ev_dire
        best_odd = odd_dire
        best_p_model = p_model_dire
        best_p_fair = p_fair_dire
        best_edge = edge_dire

    full_k = calculate_kelly_fraction(best_p_model, best_odd)

    if best_ev < min_ev or full_k <= 0.0:
        overround = (1.0 / odd_radiant + 1.0 / odd_dire - 1.0) * 100.0
        reason = (
            f"Sem valor matemático suficiente. O melhor lado ({best_name}) possui EV de {best_ev*100:.1f}%, "
            f"abaixo da margem mínima de segurança (+{min_ev*100:.1f}%). "
            f"A margem da casa (vig) é de {overround:.1f}%."
        )
        return BetOpportunity(
            side=None,
            side_name=None,
            odd=best_odd,
            model_prob=best_p_model,
            market_prob_fair=best_p_fair,
            ev=best_ev,
            edge=best_edge,
            full_kelly=full_k,
            fractional_kelly=0.0,
            recommended_stake=0.0,
            stake_percentage=0.0,
            is_actionable=False,
            reason=reason,
        )

    # Aplicação do Kelly Fracionário com teto máximo (Cap)
    f_k = full_k * kelly_multiplier
    f_capped = min(f_k, max_stake_pct)
    stake_value = round(bankroll * f_capped, 2)
    stake_pct = round(f_capped * 100.0, 2)

    reason = (
        f"Oportunidade +EV identificada no {best_name}! "
        f"O modelo estima {best_p_model*100:.1f}% de vitória contra {best_p_fair*100:.1f}% implícito na casa "
        f"(Edge: +{best_edge*100:.1f}%, EV: +{best_ev*100:.1f}%). "
        f"Gestão Kelly Fracionário ({kelly_multiplier*100:.0f}%) recomenda alocar {stake_pct:.2f}% da banca."
    )

    return BetOpportunity(
        side=best_side,
        side_name=best_name,
        odd=best_odd,
        model_prob=best_p_model,
        market_prob_fair=best_p_fair,
        ev=best_ev,
        edge=best_edge,
        full_kelly=full_k,
        fractional_kelly=f_capped,
        recommended_stake=stake_value,
        stake_percentage=stake_pct,
        is_actionable=True,
        reason=reason,
    )


def simulate_bankroll_growth(
    p_win: float,
    odd: float,
    initial_bankroll: float = 1000.0,
    kelly_multiplier: float = 0.25,
    max_stake_pct: float = 0.05,
    n_bets: int = 250,
    n_simulations: int = 200,
    seed: int = 42,
) -> MonteCarloSimulation:
    """Simula a evolução da banca através do método de Monte Carlo ao longo de N apostas,
    permitindo avaliar a variância, pior drawdown e risco de quebra sob o Critério de Kelly.

    Args:
        p_win: Probabilidade real de acerto (ex: 0.65)
        odd: Odd média recebida (ex: 1.80)
        initial_bankroll: Valor inicial da banca
        kelly_multiplier: Fração de Kelly aplicada
        max_stake_pct: Teto máximo por aposta
        n_bets: Quantidade de apostas sequenciais simuladas
        n_simulations: Quantidade de universos paralelos simulados
        seed: Semente pseudo-aleatória

    Returns:
        MonteCarloSimulation com percentis, drawdowns e amostra de trajetórias.
    """
    rng = np.random.default_rng(seed)
    f_star = calculate_kelly_fraction(p_win, odd)
    fraction = min(f_star * kelly_multiplier, max_stake_pct)

    if fraction <= 0.0:
        flat = [initial_bankroll] * (n_bets + 1)
        return MonteCarloSimulation(
            initial_bankroll=initial_bankroll,
            median_final_bankroll=initial_bankroll,
            percentile_10=initial_bankroll,
            percentile_90=initial_bankroll,
            max_drawdown_median=0.0,
            risk_of_ruin=0.0,
            trajectories_sample=[flat for _ in range(5)],
        )

    b = odd - 1.0
    all_trajectories = np.zeros((n_simulations, n_bets + 1))
    all_trajectories[:, 0] = initial_bankroll
    max_drawdowns = np.zeros(n_simulations)

    outcomes = rng.random((n_simulations, n_bets)) < p_win

    for i in range(n_simulations):
        current_b = initial_bankroll
        peak = initial_bankroll
        max_dd = 0.0

        for t in range(n_bets):
            stake = current_b * fraction
            if outcomes[i, t]:
                current_b += stake * b
            else:
                current_b -= stake
                if current_b < 0.01:
                    current_b = 0.0

            if current_b > peak:
                peak = current_b
            dd = (peak - current_b) / peak if peak > 0 else 0.0
            if dd > max_dd:
                max_dd = dd

            all_trajectories[i, t + 1] = current_b

        max_drawdowns[i] = max_dd

    finals = all_trajectories[:, -1]
    median_final = float(np.median(finals))
    p10 = float(np.percentile(finals, 10))
    p90 = float(np.percentile(finals, 90))
    median_dd = float(np.median(max_drawdowns))
    # Risco de ruína definido como probabilidade de sofrer drawdown >= 50%
    ruin_count = np.sum(max_drawdowns >= 0.50)
    risk_of_ruin = float(ruin_count / n_simulations)

    sample_indices = np.linspace(0, n_simulations - 1, 5, dtype=int)
    samples = [all_trajectories[idx].tolist() for idx in sample_indices]

    return MonteCarloSimulation(
        initial_bankroll=initial_bankroll,
        median_final_bankroll=round(median_final, 2),
        percentile_10=round(p10, 2),
        percentile_90=round(p90, 2),
        max_drawdown_median=round(median_dd * 100.0, 1),
        risk_of_ruin=round(risk_of_ruin * 100.0, 1),
        trajectories_sample=samples,
    )
