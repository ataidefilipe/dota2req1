"""Testes unitários para o módulo matemático de apostas, +EV e Critério de Kelly."""

import pytest
from src.betting.engine import (
    remove_vigorish,
    calculate_ev,
    calculate_kelly_fraction,
    evaluate_betting_opportunity,
    simulate_bankroll_growth,
)


def test_remove_vigorish_symmetric():
    """Odds simétricas com vig devem retornar probabilidades justas de 50%/50%."""
    p_rad, p_dire = remove_vigorish(1.85, 1.85)
    assert pytest.approx(p_rad, 0.001) == 0.5
    assert pytest.approx(p_dire, 0.001) == 0.5
    assert pytest.approx(p_rad + p_dire, 0.0001) == 1.0


def test_remove_vigorish_skewed():
    """Odds assimétricas devem somar 1.0 preservando a proporção de mercado."""
    p_rad, p_dire = remove_vigorish(1.25, 4.00)
    assert p_rad > p_dire
    assert pytest.approx(p_rad + p_dire, 0.0001) == 1.0
    # 1/1.25 = 0.8, 1/4.0 = 0.25 -> sum = 1.05. p_rad = 0.8 / 1.05 = 0.7619
    assert pytest.approx(p_rad, 0.001) == 0.7619


def test_remove_vigorish_invalid():
    """Odds <= 1.00 devem levantar ValueError."""
    with pytest.raises(ValueError):
        remove_vigorish(1.00, 2.00)
    with pytest.raises(ValueError):
        remove_vigorish(1.90, 0.95)


def test_calculate_ev():
    """Valida o cálculo do retorno esperado relativo."""
    # Moeda justa a odd 2.00 -> EV neutro (0%)
    assert pytest.approx(calculate_ev(0.50, 2.00), 0.001) == 0.0

    # 60% de chance a odd 2.00 -> EV positivo (+20%)
    assert pytest.approx(calculate_ev(0.60, 2.00), 0.001) == 0.20

    # 40% de chance a odd 2.00 -> EV negativo (-20%)
    assert pytest.approx(calculate_ev(0.40, 2.00), 0.001) == -0.20

    # Odd inválida
    assert calculate_ev(0.50, 0.90) == -1.0


def test_calculate_kelly_fraction():
    """Valida o cálculo da fração teórica do Critério de Kelly."""
    # Se EV <= 0, Kelly deve recomendar 0%
    assert calculate_kelly_fraction(0.40, 2.00) == 0.0
    assert calculate_kelly_fraction(0.50, 2.00) == 0.0

    # Se P=0.60 e Odd=2.00 -> b=1, EV=0.20 -> f* = 0.20 (20% da banca)
    assert pytest.approx(calculate_kelly_fraction(0.60, 2.00), 0.001) == 0.20

    # Certeza absoluta (P=1.0) -> f* = 1.0 (100% da banca)
    assert pytest.approx(calculate_kelly_fraction(1.0, 2.00), 0.001) == 1.0


def test_evaluate_betting_opportunity_radiant_value():
    """Cenário com clara assimetria a favor do Radiant."""
    # Modelo diz 65% Radiant. Casa paga 1.90 no Radiant e 1.90 no Dire
    opp = evaluate_betting_opportunity(
        p_model_radiant=0.65,
        odd_radiant=1.90,
        odd_dire=1.90,
        bankroll=1000.0,
        kelly_multiplier=0.25,
        max_stake_pct=0.05,
        min_ev=0.03,
    )
    assert opp.is_actionable is True
    assert opp.side == "radiant"
    assert opp.ev > 0.15  # EV = 0.65 * 1.90 - 1 = +23.5%
    assert opp.edge > 0.10  # Modelo 65% vs Casa 50%
    assert opp.recommended_stake > 0.0
    assert opp.recommended_stake <= 50.0  # Respeita o teto de 5% de R$ 1000


def test_evaluate_betting_opportunity_dire_value():
    """Cenário onde o Dire é o lado com valor."""
    # Modelo diz 30% Radiant (logo 70% Dire). Casa paga 2.00 no Dire
    opp = evaluate_betting_opportunity(
        p_model_radiant=0.30,
        odd_radiant=1.80,
        odd_dire=2.00,
        bankroll=2000.0,
        kelly_multiplier=0.25,
        max_stake_pct=0.05,
    )
    assert opp.is_actionable is True
    assert opp.side == "dire"
    assert opp.ev > 0.20  # EV = 0.70 * 2.00 - 1 = +40%
    assert opp.recommended_stake > 0.0


def test_evaluate_betting_opportunity_no_value():
    """Cenário onde a margem da casa elimina o valor de ambos os lados."""
    # Modelo 50% vs Casa 1.85 / 1.85 -> EV negativo em ambos
    opp = evaluate_betting_opportunity(
        p_model_radiant=0.50,
        odd_radiant=1.85,
        odd_dire=1.85,
        bankroll=1000.0,
    )
    assert opp.is_actionable is False
    assert opp.side is None
    assert opp.recommended_stake == 0.0
    assert "Sem valor matemático" in opp.reason


def test_simulate_bankroll_growth_positive_edge():
    """Simulação de Monte Carlo com edge positivo deve gerar crescimento de banca."""
    sim = simulate_bankroll_growth(
        p_win=0.65,
        odd=1.85,
        initial_bankroll=1000.0,
        kelly_multiplier=0.25,
        n_bets=100,
        n_simulations=50,
        seed=42,
    )
    assert sim.median_final_bankroll > sim.initial_bankroll
    assert len(sim.trajectories_sample) == 5
    assert len(sim.trajectories_sample[0]) == 101
    assert 0.0 <= sim.risk_of_ruin <= 100.0


def test_simulate_bankroll_growth_zero_edge():
    """Sem edge, a banca deve permanecer estável/inalterada por não apostar."""
    sim = simulate_bankroll_growth(
        p_win=0.50,
        odd=1.85,  # EV negativo
        initial_bankroll=1000.0,
        n_bets=50,
    )
    assert sim.median_final_bankroll == 1000.0
    assert sim.risk_of_ruin == 0.0
