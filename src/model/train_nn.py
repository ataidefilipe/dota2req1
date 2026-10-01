import os
import json
import logging
import copy
import shutil
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import log_loss, accuracy_score, roc_auc_score

from .neural_dataset import create_data_loaders
from .neural_network import DotaWinPredictorNN
from .calibration import TemperatureScaler
from ..config import Config

logger = logging.getLogger(__name__)

def evaluate_model(model: nn.Module, data_loader, criterion, device) -> dict:
    model.eval()
    total_loss = 0.0
    all_logits = []
    all_targets = []
    total_samples = 0

    with torch.no_grad():
        for batch in data_loader:
            # Enviar tensores para device
            batch_dev = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            logits = model(batch_dev)
            targets = batch_dev['target']

            loss = criterion(logits, targets)
            total_loss += loss.item() * len(targets)
            total_samples += len(targets)

            all_logits.append(logits.cpu())
            all_targets.append(targets.cpu())

    all_logits = torch.cat(all_logits, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    all_probs = torch.sigmoid(all_logits).numpy()
    targets_np = all_targets.numpy()

    # Métricas
    avg_loss = total_loss / max(total_samples, 1)
    acc = accuracy_score(targets_np, (all_probs >= 0.5).astype(int))

    try:
        auc = float(roc_auc_score(targets_np, all_probs))
    except Exception:
        auc = 0.5

    brier = float(np.mean((all_probs - targets_np) ** 2))

    return {
        'loss': float(avg_loss),
        'accuracy': float(acc),
        'roc_auc': float(auc),
        'brier_score': float(brier),
        'logits': all_logits,
        'targets': all_targets,
        'probs': all_probs
    }


def train_neural_network(
    data_dir: str = 'data/processed_neural',
    output_dir: str = 'models/draft_win_nn',
    reports_dir: str = 'reports',
    epochs: int = 100,
    batch_size: int = 32,
    lr: float = 3e-4,
    weight_decay: float = 0.02,
    dropout: float = 0.20,
    label_smoothing: float = 0.06,
    augment_symmetry: bool = True,
    patience: int = 20,
    seed: int = 42
):
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Dispositivo de treinamento selecionado: {device}")

    # 1. Carregar DataLoaders e Encoder
    train_loader, val_loader, test_loader, encoder = create_data_loaders(
        data_dir=data_dir,
        batch_size=batch_size
    )

    logger.info(
        f"Dados carregados: Treino={len(train_loader.dataset)}, "
        f"Validação={len(val_loader.dataset)}, Teste={len(test_loader.dataset)} "
        f"(Augment Simetria: {augment_symmetry} | Label Smoothing: {label_smoothing})"
    )

    # 2. Inicializar Modelo
    model_config = {
        'num_heroes': len(encoder.hero_to_idx),
        'num_patches': len(encoder.patch_to_idx),
        'num_ranks': len(encoder.rank_to_idx),
        'hero_dim': 32,
        'patch_dim': 8,
        'rank_dim': 8,
        'dropout': dropout
    }

    model = DotaWinPredictorNN(**model_config).to(device)
    logger.info(f"Modelo instanciado com {sum(p.numel() for p in model.parameters() if p.requires_grad)} parâmetros treináveis.")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    # 3. Loop de Treinamento com Early Stopping
    best_val_loss = float('inf')
    best_val_score = -float('inf')
    best_weights = None
    epochs_no_improve = 0
    history = []

    logger.info("Iniciando treinamento da Rede Neural...")

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_accum = 0.0
        train_samples = 0

        for batch in train_loader:
            batch_dev = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            # Aumento de dados por simetria (inverte lados Radiant e Dire e o target)
            if augment_symmetry:
                swapped = {
                    'picks': torch.stack([batch_dev['picks'][:, 1], batch_dev['picks'][:, 0]], dim=1),
                    'bans': torch.stack([batch_dev['bans'][:, 1], batch_dev['bans'][:, 0]], dim=1),
                    'bans_mask': torch.stack([batch_dev['bans_mask'][:, 1], batch_dev['bans_mask'][:, 0]], dim=1),
                    'bans_available': torch.stack([batch_dev['bans_available'][:, 1], batch_dev['bans_available'][:, 0]], dim=1),
                    'ranks': torch.stack([batch_dev['ranks'][:, 1], batch_dev['ranks'][:, 0]], dim=1),
                    'mmr': torch.stack([batch_dev['mmr'][:, 1], batch_dev['mmr'][:, 0]], dim=1),
                    'skill_masks': torch.stack([batch_dev['skill_masks'][:, 1], batch_dev['skill_masks'][:, 0]], dim=1),
                    'patch': batch_dev['patch'],
                    'target': 1.0 - batch_dev['target'],
                    'match_ids': batch_dev['match_ids']
                }
                train_batch = {
                    k: torch.cat([batch_dev[k], swapped[k]], dim=0) if isinstance(batch_dev[k], torch.Tensor) else batch_dev[k]
                    for k in batch_dev
                }
            else:
                train_batch = batch_dev

            optimizer.zero_grad()
            logits = model(train_batch)
            targets = train_batch['target']

            # Label smoothing para evitar super-confiança em esports
            if label_smoothing > 0.0:
                targets_smooth = targets * (1.0 - 2.0 * label_smoothing) + label_smoothing
            else:
                targets_smooth = targets

            loss = criterion(logits, targets_smooth)
            loss.backward()

            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss_accum += loss.item() * len(targets)
            train_samples += len(targets)

        scheduler.step()

        train_loss = train_loss_accum / max(train_samples, 1)

        # Avaliar na Validação (com targets reais e sem alteração)
        val_eval = evaluate_model(model, val_loader, criterion, device)
        val_loss = val_eval['loss']
        val_acc = val_eval['accuracy']
        val_auc = val_eval['roc_auc']

        history.append({
            'epoch': epoch,
            'train_loss': round(train_loss, 4),
            'val_loss': round(val_loss, 4),
            'val_acc': round(val_acc, 4),
            'val_auc': round(val_auc, 4),
            'lr': float(scheduler.get_last_lr()[0])
        })

        is_best = val_auc > best_val_score
        if is_best:
            best_val_score = val_auc
            best_val_loss = val_loss
            best_weights = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0

        if epoch % 5 == 0 or is_best:
            logger.info(
                f"Época [{epoch:03d}/{epochs:03d}] | "
                f"Treino Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val Acc: {val_acc * 100:.1f}% | "
                f"Val AUC: {val_auc:.4f}" + (" [*]" if is_best else "")
            )

        if not is_best:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                logger.info(f"Early Stopping acionado após {epoch} épocas sem melhora no Val AUC.")
                break

    # Restaurar melhores pesos
    if best_weights is not None:
        model.load_state_dict(best_weights)
        logger.info(f"Melhor checkpoint restaurado com Val AUC: {best_val_score:.4f} (Val Loss: {best_val_loss:.4f})")

    # 4. Calibração de Probabilidades (Temperature Scaling)
    val_final_eval = evaluate_model(model, val_loader, criterion, device)
    scaler = TemperatureScaler()
    scaler.fit(val_final_eval['logits'], val_final_eval['targets'])

    calib_stats = scaler.evaluate_calibration(
        scaler.predict_proba(val_final_eval['logits']),
        val_final_eval['targets']
    )
    logger.info(f"Temperatura ótima calibrada: T = {scaler.temperature:.4f} (ECE Val: {calib_stats['ece']:.4f})")

    # 5. Avaliação Final no Teste (Sem vazamento)
    test_eval = evaluate_model(model, test_loader, criterion, device)
    test_calib_probs = scaler.predict_proba(test_eval['logits']).numpy()
    test_targets_np = test_eval['targets'].numpy()

    test_calib_loss = float(log_loss(test_targets_np, test_calib_probs))
    test_calib_acc = float(accuracy_score(test_targets_np, (test_calib_probs >= 0.5).astype(int)))
    test_calib_auc = float(roc_auc_score(test_targets_np, test_calib_probs))
    test_calib_brier = float(np.mean((test_calib_probs - test_targets_np) ** 2))

    logger.info("=" * 60)
    logger.info("RESULTADOS NO CONJUNTO DE TESTE (CONGELADO):")
    logger.info(f"Log Loss (Sem Calibrar): {test_eval['loss']:.4f}")
    logger.info(f"Log Loss (Calibrado):    {test_calib_loss:.4f}")
    logger.info(f"Acurácia:                {test_calib_acc * 100:.2f}%")
    logger.info(f"ROC-AUC:                 {test_calib_auc:.4f}")
    logger.info(f"Brier Score:             {test_calib_brier:.4f}")
    logger.info("=" * 60)

    # 6. Salvar Artefato Autocontido em models/draft_win_nn/
    os.makedirs(output_dir, exist_ok=True)
    weights_path = os.path.join(output_dir, 'weights.pt')
    torch.save(model.state_dict(), weights_path)

    with open(os.path.join(output_dir, 'model_config.json'), 'w', encoding='utf-8') as f:
        json.dump(model_config, f, indent=2)

    with open(os.path.join(output_dir, 'calibration.json'), 'w', encoding='utf-8') as f:
        json.dump(calib_stats, f, indent=2)

    # Copiar encoder.json para dentro da pasta do modelo para torná-lo autocontido
    src_encoder_path = os.path.join(data_dir, 'encoder.json')
    if os.path.exists(src_encoder_path):
        shutil.copy2(src_encoder_path, os.path.join(output_dir, 'encoder.json'))

    # Salvar métricas completas
    metrics_report = {
        'n_train': len(train_loader.dataset),
        'n_val': len(val_loader.dataset),
        'n_test': len(test_loader.dataset),
        'best_val_loss': round(best_val_loss, 4),
        'test_uncalibrated_log_loss': round(test_eval['loss'], 4),
        'test_log_loss': round(test_calib_loss, 4),
        'test_accuracy': round(test_calib_acc, 4),
        'test_roc_auc': round(test_calib_auc, 4),
        'test_brier_score': round(test_calib_brier, 4),
        'optimal_temperature': round(scaler.temperature, 4),
        'history': history[-20:] # últimas 20 épocas
    }

    with open(os.path.join(output_dir, 'metrics.json'), 'w', encoding='utf-8') as f:
        json.dump(metrics_report, f, indent=2)

    os.makedirs(reports_dir, exist_ok=True)
    with open(os.path.join(reports_dir, 'nn_metrics.json'), 'w', encoding='utf-8') as f:
        json.dump(metrics_report, f, indent=2)

    logger.info(f"Modelo e artefatos salvos com sucesso em '{output_dir}'.")
    return metrics_report

if __name__ == "__main__":
    from ..logger import setup_logging
    setup_logging()

    parser = argparse.ArgumentParser(description="Treinamento da Rede Neural Dota 2 Win Predictor")
    parser.add_argument("--epochs", type=int, default=100, help="Máximo de épocas de treino")
    parser.add_argument("--batch-size", type=int, default=32, help="Tamanho do lote")
    parser.add_argument("--lr", type=float, default=3e-4, help="Taxa de aprendizado inicial")
    parser.add_argument("--weight-decay", type=float, default=0.02, help="Regularização L2")
    parser.add_argument("--dropout", type=float, default=0.20, help="Taxa de dropout")
    parser.add_argument("--label-smoothing", type=float, default=0.06, help="Taxa de label smoothing")
    parser.add_argument("--no-augment", action="store_true", help="Desativar aumento de dados por simetria")
    parser.add_argument("--patience", type=int, default=20, help="Paciência do early stopping")
    args = parser.parse_args()

    train_neural_network(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        dropout=args.dropout,
        label_smoothing=args.label_smoothing,
        augment_symmetry=not args.no_augment,
        patience=args.patience
    )
