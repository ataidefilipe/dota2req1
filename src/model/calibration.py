import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Tuple, Dict, Any

class TemperatureScaler:
    """
    Calibração pós-processamento de probabilidades via Temperature Scaling (Platt Scaling no espaço de logits).
    Ajusta um único parâmetro T > 0 minimizando a perda NLL (Log Loss) na partição de validação.
    """
    def __init__(self):
        self.temperature: float = 1.0
        self.fitted: bool = False

    def fit(self, logits: torch.Tensor, targets: torch.Tensor) -> "TemperatureScaler":
        """
        Encontra o T ótimo sobre os logits e alvos da validação.
        """
        # Desconecta do grafo
        logits = logits.detach().cpu()
        targets = targets.detach().cpu().float()

        # Parâmetro T como parâmetro treinável (log_t para garantir T > 0)
        log_temp = nn.Parameter(torch.zeros(1))
        optimizer = optim.LBFGS([log_temp], lr=0.05, max_iter=50)

        criterion = nn.BCEWithLogitsLoss()

        def _eval():
            optimizer.zero_grad()
            temp = torch.exp(log_temp).clamp(min=1e-3, max=100.0)
            scaled_logits = logits / temp
            loss = criterion(scaled_logits, targets)
            loss.backward()
            return loss

        optimizer.step(_eval)

        self.temperature = float(torch.exp(log_temp).clamp(min=0.01, max=50.0).item())
        self.fitted = True
        return self

    def predict_proba(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Converte logits em probabilidades calibradas P(Radiant Win).
        """
        scaled = logits / max(self.temperature, 1e-4)
        return torch.sigmoid(scaled)

    def evaluate_calibration(self, probs: torch.Tensor, targets: torch.Tensor, num_bins: int = 10) -> Dict[str, Any]:
        """
        Calcula Brier Score e Expected Calibration Error (ECE).
        """
        probs_np = probs.detach().cpu().numpy()
        targets_np = targets.detach().cpu().numpy()

        brier_score = float(np.mean((probs_np - targets_np) ** 2))

        # ECE (Expected Calibration Error)
        bin_limits = np.linspace(0.0, 1.0, num_bins + 1)
        ece = 0.0
        bin_stats = []

        for i in range(num_bins):
            bin_lower = bin_limits[i]
            bin_upper = bin_limits[i + 1]
            in_bin = (probs_np >= bin_lower) & (probs_np < bin_upper if i < num_bins - 1 else probs_np <= bin_upper)
            prop_in_bin = np.mean(in_bin)

            if prop_in_bin > 0:
                accuracy_in_bin = np.mean(targets_np[in_bin])
                avg_confidence_in_bin = np.mean(probs_np[in_bin])
                bin_error = abs(avg_confidence_in_bin - accuracy_in_bin)
                ece += bin_error * prop_in_bin
                bin_stats.append({
                    'bin': i,
                    'range': [round(bin_lower, 2), round(bin_upper, 2)],
                    'count': int(np.sum(in_bin)),
                    'acc': round(float(accuracy_in_bin), 4),
                    'conf': round(float(avg_confidence_in_bin), 4)
                })

        return {
            'temperature': round(self.temperature, 4),
            'brier_score': round(brier_score, 4),
            'ece': round(float(ece), 4),
            'bin_stats': bin_stats
        }
