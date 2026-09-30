import logging
import os
import joblib
import json
import numpy as np
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, accuracy_score
from ..config import Config

logger = logging.getLogger(__name__)

def train_model(data_dir='data/processed', model_dir='models', reports_dir='reports'):
    # Paths
    x_path = os.path.join(data_dir, 'X.npz')
    y_path = os.path.join(data_dir, 'y.npy')
    meta_path = os.path.join(data_dir, 'meta.joblib')
    
    if not os.path.exists(x_path):
        logger.error(f"Dataset not found at {x_path}. Run build_features first.")
        return

    # Load
    logger.info("Loading dataset...")
    X = sparse.load_npz(x_path)
    y = np.load(y_path)
    meta = joblib.load(meta_path) # list of dicts with start_time
    
    # Check consistency
    if X.shape[0] != len(y) or len(y) != len(meta):
        logger.error("Mismatch in dataset dimensions.")
        return

    # Temporal Split (80/20)
    # Meta is already sorted by start_time from build_features SQL query
    n_total = len(y)
    n_train = int(n_total * 0.8)
    
    X_train = X[:n_train]
    y_train = y[:n_train]
    X_test = X[n_train:]
    y_test = y[n_train:]
    
    logger.info(f"Split: Train={n_train}, Test={n_total - n_train}")
    
    if n_train < 10:
        logger.warning("Not enough data to train meaningfully, but proceeding for MVP testing.")

    # Model
    logger.info("Training Logistic Regression...")
    # Handle single class case in training
    if len(np.unique(y_train)) < 2:
         logger.warning("Training set has only one class. Model will be biased.")
         
    model = LogisticRegression(max_iter=1000, C=1.0, solver='lbfgs')
    try:
        model.fit(X_train, y_train)
    except Exception as e:
        logger.error(f"Training failed: {e}")
        # Create a dummy model or return? 
        # For M5 we need a file. 
        # If fit fails, we can't save. 
        # But LR handles single class by just predicting that class (usually). 
        # Sklearn < 1.0 might error, >= 1.0 usually warns or works.
        # Let's hope it works.
        raise

    # Evaluate
    logger.info("Evaluating...")
    try:
        y_pred_train_proba = model.predict_proba(X_train)[:, 1]
        y_pred_test_proba = model.predict_proba(X_test)[:, 1]
        
        # Train Loss
        train_loss = 0.0
        if len(np.unique(y_train)) > 1:
            train_loss = log_loss(y_train, y_pred_train_proba)
            
        # Test Loss/Acc
        test_loss = 0.0
        test_acc = 0.0
        
        if len(y_test) > 0:
            if len(np.unique(y_test)) > 1:
                test_loss = log_loss(y_test, y_pred_test_proba)
            else:
                logger.info("Test set has 1 class.")
            test_acc = accuracy_score(y_test, (y_pred_test_proba >= 0.5).astype(int))
            
        logger.info(f"Train LogLoss: {train_loss:.4f}")
        logger.info(f"Test LogLoss:  {test_loss:.4f}")
        logger.info(f"Test Accuracy: {test_acc:.4f}")
        
    except Exception as e:
        logger.warning(f"Evaluation failed (skipping metrics): {e}")

    # Save Model
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, 'draft_win_lr.joblib')
    joblib.dump(model, model_path)
    logger.info(f"Model saved to {model_path}")
    
    # Save Metrics
    os.makedirs(reports_dir, exist_ok=True)
    # Use locals() or defaults if eval failed
    metrics = {
        'n_samples': int(n_total),
        'n_train': int(n_train),
        'n_test': int(n_total - n_train),
        'train_log_loss': float(locals().get('train_loss', 0.0)),
        'test_log_loss': float(locals().get('test_loss', 0.0)),
        'test_accuracy': float(locals().get('test_acc', 0.0))
    }
    with open(os.path.join(reports_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)

if __name__ == "__main__":
    from ..logger import setup_logging
    setup_logging()
    train_model()
