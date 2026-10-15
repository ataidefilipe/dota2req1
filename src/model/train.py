import logging
import os
import joblib
import json
import numpy as np
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, accuracy_score, roc_auc_score, brier_score_loss
from ..config import Config
from ..features.composition_features import CompositionScaler, FEATURE_NAMES

logger = logging.getLogger(__name__)

def train_model(data_dir='data/processed', model_dir='models', reports_dir='reports'):
    # Paths
    x_path = os.path.join(data_dir, 'X.npz')
    y_path = os.path.join(data_dir, 'y.npy')
    meta_path = os.path.join(data_dir, 'meta.joblib')
    comp_path = os.path.join(data_dir, 'composition_delta.npy')

    if not os.path.exists(x_path):
        logger.error(f"Dataset not found at {x_path}. Run build_features first.")
        return

    # Load
    logger.info("Loading dataset...")
    X = sparse.load_npz(x_path)
    y = np.load(y_path)
    meta = joblib.load(meta_path) # list of dicts with start_time
    comp = np.load(comp_path) if os.path.exists(comp_path) else None

    # Check consistency
    if X.shape[0] != len(y) or len(y) != len(meta):
        logger.error("Mismatch in dataset dimensions.")
        return

    # Temporal Split (80/20)
    n_total = len(y)
    n_train = int(n_total * 0.8)

    X_train_base = X[:n_train]
    y_train = y[:n_train]
    X_test_base = X[n_train:]
    y_test = y[n_train:]

    logger.info(f"Split Temporal: Treino={n_train}, Teste={n_total - n_train}")

    comp_scaler = None
    if comp is not None:
        logger.info(f"Carregando features de composição: shape={comp.shape}")
        comp_scaler = CompositionScaler()
        comp_train = comp_scaler.fit_transform(comp[:n_train])
        comp_test = comp_scaler.transform(comp[n_train:])

        X_train_full = sparse.hstack([X_train_base, sparse.csr_matrix(comp_train)], format='csr')
        X_test_full = sparse.hstack([X_test_base, sparse.csr_matrix(comp_test)], format='csr')
    else:
        X_train_full = X_train_base
        X_test_full = X_test_base

    # Benchmark: Baseline vs Composição
    logger.info("Executando benchmark comparativo...")
    models_to_test = {
        'lr_c0.1_base': (LogisticRegression(C=0.1, max_iter=1000, solver='lbfgs', random_state=42), X_train_base, X_test_base),
        'lr_c0.1_comp': (LogisticRegression(C=0.1, max_iter=1000, solver='lbfgs', random_state=42), X_train_full, X_test_full),
        'lr_c1.0_base': (LogisticRegression(C=1.0, max_iter=1000, solver='lbfgs', random_state=42), X_train_base, X_test_base),
        'lr_c1.0_comp': (LogisticRegression(C=1.0, max_iter=1000, solver='lbfgs', random_state=42), X_train_full, X_test_full),
    }

    benchmark_results = {}
    for name, (clf, xtr, xte) in models_to_test.items():
        clf.fit(xtr, y_train)
        probs_tr = clf.predict_proba(xtr)[:, 1]
        probs_te = clf.predict_proba(xte)[:, 1]

        tr_loss = log_loss(y_train, probs_tr)
        te_loss = log_loss(y_test, probs_te)
        te_auc = roc_auc_score(y_test, probs_te)
        te_brier = brier_score_loss(y_test, probs_te)
        te_acc = accuracy_score(y_test, (probs_te >= 0.5).astype(int))

        benchmark_results[name] = {
            'train_log_loss': round(float(tr_loss), 4),
            'test_log_loss': round(float(te_loss), 4),
            'test_roc_auc': round(float(te_auc), 4),
            'test_brier_score': round(float(te_brier), 4),
            'test_accuracy': round(float(te_acc), 4)
        }
        logger.info(f"{name:15} | LogLoss: {te_loss:.4f} | AUC: {te_auc:.4f} | Brier: {te_brier:.4f} | Acc: {te_acc*100:.2f}%")

    # Escolha do modelo de produção: LR C=0.1 com features de composição
    best_clf, X_tr_prod, X_te_prod = models_to_test['lr_c0.1_comp']
    prod_metrics = benchmark_results['lr_c0.1_comp']

    # Log dos coeficientes de composição
    if comp is not None:
        comp_coefs = best_clf.coef_[0][-len(FEATURE_NAMES):]
        logger.info("Pesos das Features de Composição no Modelo:")
        for fname, val in zip(FEATURE_NAMES, comp_coefs):
            logger.info(f"  - {fname:<22}: {val:+.4f}")

    # Salvar modelos
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, 'draft_win_lr.joblib')
    joblib.dump(best_clf, model_path)
    logger.info(f"Modelo salvo em {model_path}")

    if comp_scaler is not None:
        scaler_path = os.path.join(model_dir, 'draft_win_scaler.joblib')
        joblib.dump(comp_scaler, scaler_path)
        logger.info(f"Scaler de composição salvo em {scaler_path}")

    # Salvar relatórios
    os.makedirs(reports_dir, exist_ok=True)
    with open(os.path.join(reports_dir, 'tabular_benchmark.json'), 'w') as f:
        json.dump(benchmark_results, f, indent=2)

    metrics = {
        'n_samples': int(n_total),
        'n_train': int(n_train),
        'n_test': int(n_total - n_train),
        'train_log_loss': prod_metrics['train_log_loss'],
        'test_log_loss': prod_metrics['test_log_loss'],
        'test_roc_auc': prod_metrics['test_roc_auc'],
        'test_brier_score': prod_metrics['test_brier_score'],
        'test_accuracy': prod_metrics['test_accuracy']
    }
    with open(os.path.join(reports_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Métricas salvas em {os.path.join(reports_dir, 'metrics.json')}")

if __name__ == "__main__":
    from ..logger import setup_logging
    setup_logging()
    train_model()

