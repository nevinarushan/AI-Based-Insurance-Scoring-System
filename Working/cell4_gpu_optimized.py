# ====================
# 🎯 CELL 4: KAGGLE-OPTIMIZED GPU-ACCELERATED TRAINING
# Microsoft Malware Prediction - Complete Integration with Pipeline
# ====================

import os
# 🔧 CRITICAL: Enable CUDA launch blocking for better GPU debugging
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score, log_loss, accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import gc
import warnings
warnings.filterwarnings('ignore')

print("🎯 GPU-ACCELERATED MULTI-MODEL TRAINING PIPELINE")
print("="*60)

# ====================
# 🎯 COMPATIBILITY CHECK WITH CELL 3 OUTPUT
# ====================

print("🔍 Checking compatibility with Cell 3 output...")

# Check for required variables from Cell 3
required_from_cell3 = ['X_ready_for_models', 'y_work', 'cv_folds_enhanced', 'feature_metadata']
missing_vars = [var for var in required_from_cell3 if var not in globals()]

if missing_vars:
    print(f"❌ Missing variables from Cell 3: {missing_vars}")
    print("   Please run Cell 3 first!")

    # Check for alternative variable names
    available_vars = [var for var in globals().keys() if any(keyword in var.lower() for keyword in ['x_', 'y_', 'cv', 'feature', 'model'])]
    print(f"   Available data variables: {available_vars}")
    raise RuntimeError(f"Required variables from Cell 3 not found: {missing_vars}")

# Compatibility layer - ensure all expected variables exist
if 'X_ready_for_models' in globals():
    X_train = X_ready_for_models  # Primary dataset for training
    print(f"✅ Training data ready: {X_train.shape}")

if 'y_work' in globals():
    y_train = y_work  # Target variable
    print(f"✅ Target variable ready: {len(y_train)} samples")

if 'cv_folds_enhanced' in globals():
    cv_folds = cv_folds_enhanced  # Use enhanced CV folds from Cell 3
    print(f"✅ CV folds ready: {len(cv_folds)} folds")
elif 'cv_folds' in globals():
    print(f"✅ Using original CV folds: {len(cv_folds)} folds")

# Check for scaled versions
scaled_available = []
if 'X_scaled_standard' in globals():
    scaled_available.append('StandardScaler')
if 'X_scaled_robust' in globals():
    scaled_available.append('RobustScaler')

print(f"✅ Scaled datasets available: {scaled_available}")

# Fix feature metadata access to use correct keys from Cell 3
if 'feature_metadata' in globals() and feature_metadata:
    feature_count = feature_metadata.get('final_feature_count', len(X_train.columns) if 'X_train' in locals() else 0)
    original_count = feature_metadata.get('original_feature_count', feature_count)
    print(f"✅ Feature metadata: {feature_count} features (reduced from {original_count})")

    # Show feature selection summary if available
    if 'removed_features' in feature_metadata:
        removed_stats = feature_metadata['removed_features']
        total_removed = sum(len(removed_stats.get(method, [])) for method in removed_stats.keys())
        print(f"   📊 Feature selection: {total_removed} features removed via various methods")
else:
    feature_count = len(X_train.columns) if 'X_train' in locals() else 0
    print(f"✅ Feature count: {feature_count} features")

print("✅ All compatibility checks passed - ready for model training!")

# 🔧 CRITICAL: Validate label format for PyTorch/CUDA compatibility
print("\n🔧 VALIDATING LABEL FORMAT FOR PYTORCH/CUDA:")
if 'y_train' in locals():
    print(f"   Original y_train labels: {y_train.unique()}")
    print(f"   Label dtype: {y_train.dtype}")

    # Ensure labels are integer and 0-based for PyTorch/CUDA
    if y_train.dtype != 'int' or y_train.min() != 0:
        print("   🔧 Fixing label format for PyTorch/CUDA compatibility...")
        y_train = y_train.astype(int) - y_train.min()
        print(f"   ✅ Fixed y_train labels: {y_train.unique()}")
    else:
        print("   ✅ Labels already properly formatted for PyTorch/CUDA")

    print(f"   🎯 Labels verified compatible with PyTorch/CUDA (integer, 0-based)")
else:
    print("   ⚠️ No target variable found - using dummy validation")

print("✅ All compatibility checks passed - ready for model training!")

# ====================
# 🎯 GPU Detection and Configuration
# ====================
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    GPU_AVAILABLE = torch.cuda.is_available()
    if GPU_AVAILABLE:
        DEVICE = torch.device('cuda')
        GPU_MEMORY = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"🚀 GPU DETECTED: {torch.cuda.get_device_name(0)}")
        print(f"⚡ GPU Memory: {GPU_MEMORY:.1f}GB - READY TO USE!")
        torch.cuda.empty_cache()  # Clear GPU memory
    else:
        DEVICE = torch.device('cpu')
        print("❌ No GPU detected - using CPU")
except ImportError:
    GPU_AVAILABLE = False
    DEVICE = torch.device('cpu')
    print("❌ PyTorch not installed - install with: pip install torch")

# ====================
# 🎯 GPU-ACCELERATED NEURAL NETWORK
# ====================

class GPUMalwareNet(nn.Module):
    """Advanced Neural Network for Malware Detection - GPU Optimized"""

    def __init__(self, input_size, hidden_sizes=[512, 256, 128, 64], dropout_rate=0.3):
        super(GPUMalwareNet, self).__init__()

        layers = []
        prev_size = input_size

        # Build hidden layers with dropout and batch normalization
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_size = hidden_size

        # Output layer
        layers.extend([
            nn.Linear(prev_size, 1),
            nn.Sigmoid()
        ])

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

def train_gpu_neural_network(X_train, y_train, X_val, y_val, input_size):
    """Train neural network on GPU with proper regularization"""

    if not GPU_AVAILABLE:
        print("⚠️ GPU not available, skipping GPU neural network")
        return None, None

    print("🚀 Training GPU Neural Network...")

    try:
        # Convert to PyTorch tensors and move to GPU
        X_train_tensor = torch.FloatTensor(X_train.values).to(DEVICE)
        y_train_tensor = torch.FloatTensor(y_train.values).unsqueeze(1).to(DEVICE)
        X_val_tensor = torch.FloatTensor(X_val.values).to(DEVICE)
        y_val_tensor = torch.FloatTensor(y_val.values).unsqueeze(1).to(DEVICE)

        # Create data loaders
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

        # Initialize model
        model = GPUMalwareNet(input_size, hidden_sizes=[512, 256, 128, 64], dropout_rate=DROPOUT_RATE).to(DEVICE)

        # Loss and optimizer with L2 regularization
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=L2_REGULARIZATION)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

        best_val_auc = 0
        patience_counter = 0

        # Training loop
        for epoch in range(EPOCHS):
            model.train()
            total_loss = 0

            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)

                # Add L1 regularization manually
                l1_penalty = sum(p.abs().sum() for p in model.parameters())
                loss += L1_REGULARIZATION * l1_penalty

                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            # Validation
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val_tensor)
                val_probs = val_outputs.cpu().numpy().flatten()
                val_auc = roc_auc_score(y_val.values, val_probs)

            scheduler.step(val_auc)

            if val_auc > best_val_auc:
                best_val_auc = val_auc
                patience_counter = 0
                best_model_state = model.state_dict().copy()
            else:
                patience_counter += 1

            if epoch % 10 == 0:
                print(f"   Epoch {epoch}: Loss={total_loss/len(train_loader):.4f}, Val AUC={val_auc:.4f}")

            if patience_counter >= EARLY_STOPPING_ROUNDS:
                print(f"   Early stopping at epoch {epoch}")
                break

        # Load best model
        model.load_state_dict(best_model_state)
        return model, best_val_auc

    except Exception as e:
        print(f"   ❌ GPU training failed: {e}")
        return None, 0.5

# ====================
# 🎯 PERFORMANCE OPTIMIZATION CONFIGURATION
# ====================

# Get configuration from pipeline_config (from previous cells)
GLOBAL_SEED = pipeline_config.get('GLOBAL_SEED', 42)
DEVELOPMENT_MODE = False  # Switch to production mode for better performance
DEV_SAMPLE_SIZE = 0.5  # Use more data even in dev mode
N_FOLDS = 5
EARLY_STOPPING_ROUNDS = 100
LEARNING_RATE = 0.05
L1_REGULARIZATION = 0.1
L2_REGULARIZATION = 0.1
DROPOUT_RATE = 0.3
BATCH_SIZE = 512
EPOCHS = 100

print(f"🔧 PERFORMANCE SETTINGS:")
print(f"   Development Mode: {DEVELOPMENT_MODE}")
print(f"   Sample Size: {'100%' if not DEVELOPMENT_MODE else f'{DEV_SAMPLE_SIZE:.0%}'}")
print(f"   Features Available: {feature_metadata.get('final_feature_count', feature_count) if 'feature_metadata' in globals() else (feature_count if 'feature_count' in locals() else 'Unknown')}")
print(f"   Early Stopping: {EARLY_STOPPING_ROUNDS} rounds")

# ====================
# 🎯 ENHANCED MODEL CONFIGURATIONS
# ====================

def get_gpu_optimized_configs():
    """Get GPU-optimized model configurations with better performance"""
    configs = {}

    # Enhanced LightGBM Configuration
    try:
        import lightgbm as lgb
        LIGHTGBM_AVAILABLE = True

        lgb_params = {
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'num_leaves': 100,  # Increased for better performance
            'learning_rate': 0.05,
            'feature_fraction': 0.9,  # Use more features
            'bagging_fraction': 0.9,
            'bagging_freq': 5,
            'lambda_l1': 0.01,
            'lambda_l2': 0.01,
            'min_child_samples': 20,
            'random_state': GLOBAL_SEED,
            'n_estimators': 1000,
            'verbose': -1,
            'force_col_wise': True,  # Better for wide datasets
            'max_bin': 255
        }

        # GPU optimization if available
        if GPU_AVAILABLE:
            try:
                lgb_params.update({
                    'device': 'gpu',
                    'gpu_use_dp': True,
                    'num_leaves': 150  # Can use more leaves with GPU
                })
            except:
                pass  # Fall back to CPU if GPU setup fails

        configs['lightgbm'] = {
            'model': lgb.LGBMClassifier,
            'params': lgb_params,
            'use_scaled': False,
            'gpu_accelerated': GPU_AVAILABLE
        }
    except ImportError:
        LIGHTGBM_AVAILABLE = False

    # Enhanced CatBoost Configuration
    try:
        import catboost as cb
        CATBOOST_AVAILABLE = True

        cb_params = {
            'objective': 'Logloss',
            'eval_metric': 'AUC',
            'learning_rate': 0.05,
            'depth': 8,
            'l2_leaf_reg': 10,
            'random_seed': GLOBAL_SEED,
            'iterations': 1000,
            'verbose': False,
            'thread_count': -1,
            'border_count': 254,
            'bootstrap_type': 'Bayesian',
            'bagging_temperature': 0.2,
            'od_type': 'Iter',
            'od_wait': 50
        }

        # GPU optimization if available
        if GPU_AVAILABLE:
            try:
                cb_params.update({
                    'task_type': 'GPU',
                    'devices': '0',
                    'gpu_ram_part': 0.7
                })
            except:
                pass

        configs['catboost'] = {
            'model': cb.CatBoostClassifier,
            'params': cb_params,
            'use_scaled': False,
            'gpu_accelerated': GPU_AVAILABLE
        }
    except ImportError:
        CATBOOST_AVAILABLE = False

    # Enhanced XGBoost Configuration
    try:
        import xgboost as xgb
        XGBOOST_AVAILABLE = True

        xgb_params = {
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'learning_rate': 0.05,
            'max_depth': 8,
            'min_child_weight': 1,
            'subsample': 0.9,
            'colsample_bytree': 0.9,
            'reg_alpha': 0.01,
            'reg_lambda': 0.01,
            'random_state': GLOBAL_SEED,
            'n_estimators': 1000,
            'verbosity': 0,
            'scale_pos_weight': 1  # Will adjust based on class imbalance
        }

        # GPU optimization if available
        if GPU_AVAILABLE:
            try:
                xgb_params.update({
                    'tree_method': 'gpu_hist',
                    'gpu_id': 0,
                    'predictor': 'gpu_predictor'
                })
            except:
                pass

        configs['xgboost'] = {
            'model': xgb.XGBClassifier,
            'params': xgb_params,
            'use_scaled': False,
            'gpu_accelerated': GPU_AVAILABLE
        }
    except ImportError:
        XGBOOST_AVAILABLE = False

    return configs

def prepare_training_data(X, y, development_mode=True, sample_size=0.2):
    """Prepare training data with optional sampling for development mode"""
    if not development_mode:
        print("🎯 PRODUCTION MODE: Using full dataset")
        return X, y

    print(f"🔧 DEVELOPMENT MODE: Sampling {sample_size:.1%} of data for fast iteration...")

    # Stratified sampling to maintain class balance
    X_sample, _, y_sample, _ = train_test_split(
        X, y,
        train_size=sample_size,
        stratify=y,
        random_state=GLOBAL_SEED
    )

    print(f"📊 Sampled Dataset: {X_sample.shape[0]:,} rows × {X_sample.shape[1]} features")
    print(f"🎯 Sample Target Distribution: {y_sample.value_counts().to_dict()}")

    return X_sample, y_sample

def train_model_cv(model_name, config, X_train, y_train, cv_folds):
    """Train a single model using cross-validation with regularization"""
    print(f"\n🚀 Training {model_name}...")

    # Choose appropriate dataset (scaled or unscaled)
    if config['use_scaled']:
        if 'X_scaled_standard' in globals():
            X_data = X_scaled_standard
        else:
            print(f"   Creating scaled data for {model_name}...")
            scaler = StandardScaler()
            X_data = pd.DataFrame(
                scaler.fit_transform(X_train),
                columns=X_train.columns,
                index=X_train.index
            )
    else:
        X_data = X_train

    # Initialize results storage
    oof_predictions = np.zeros(len(X_data))
    cv_scores = []
    models = []

    # Cross-validation training
    for fold_idx, (train_idx, val_idx) in enumerate(cv_folds):
        print(f"   Fold {fold_idx + 1}/{len(cv_folds)}", end=" ")

        # Split data
        X_fold_train = X_data.iloc[train_idx]
        X_fold_val = X_data.iloc[val_idx]
        y_fold_train = y_train.iloc[train_idx]
        y_fold_val = y_train.iloc[val_idx]

        # Initialize and train model
        model = config['model'](**config['params'])

        # Special handling for gradient boosting models with early stopping
        if model_name in ['lightgbm', 'xgboost']:
            try:
                model.fit(
                    X_fold_train, y_fold_train,
                    eval_set=[(X_fold_val, y_fold_val)],
                    verbose=False
                )
            except:
                # Fallback without eval_set
                model.fit(X_fold_train, y_fold_train)
        elif model_name == 'catboost':
            try:
                model.fit(
                    X_fold_train, y_fold_train,
                    eval_set=(X_fold_val, y_fold_val),
                    verbose=False
                )
            except:
                # Fallback without eval_set
                model.fit(X_fold_train, y_fold_train)
        else:
            # Standard training for other models
            model.fit(X_fold_train, y_fold_train)

        # Predict on validation set
        if hasattr(model, 'predict_proba'):
            fold_predictions = model.predict_proba(X_fold_val)[:, 1]
        else:
            fold_predictions = model.decision_function(X_fold_val)

        # Store predictions and calculate score
        oof_predictions[val_idx] = fold_predictions
        fold_score = roc_auc_score(y_fold_val, fold_predictions)
        cv_scores.append(fold_score)
        models.append(model)

        print(f"AUC: {fold_score:.4f}")

    # Calculate overall CV score
    overall_score = roc_auc_score(y_train, oof_predictions)
    print(f"   📊 {model_name} CV Score: {overall_score:.4f} (±{np.std(cv_scores):.4f})")

    return {
        'model_name': model_name,
        'models': models,
        'oof_predictions': oof_predictions,
        'cv_scores': cv_scores,
        'mean_cv_score': np.mean(cv_scores),
        'std_cv_score': np.std(cv_scores),
        'overall_score': overall_score
    }

def train_model_cv_enhanced(model_name, config, X_train, y_train, cv_folds):
    """Enhanced model training with better error handling and performance"""
    print(f"\n🚀 Training {model_name}...")

    # Choose appropriate dataset
    X_data = X_train  # Use unscaled data for tree models

    # Calculate class weights for imbalanced data
    pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    # Update XGBoost scale_pos_weight
    if model_name == 'xgboost' and 'scale_pos_weight' in config['params']:
        config['params']['scale_pos_weight'] = pos_weight
        print(f"   Adjusted scale_pos_weight: {pos_weight:.2f}")

    # Initialize results storage
    oof_predictions = np.zeros(len(X_data))
    cv_scores = []
    models = []

    # Cross-validation training with better error handling
    for fold_idx, (train_idx, val_idx) in enumerate(cv_folds):
        print(f"   Fold {fold_idx + 1}/{len(cv_folds)}", end=" ")

        try:
            # Split data
            X_fold_train = X_data.iloc[train_idx]
            X_fold_val = X_data.iloc[val_idx]
            y_fold_train = y_train.iloc[train_idx]
            y_fold_val = y_train.iloc[val_idx]

            # Initialize model with current parameters
            model_params = config['params'].copy()
            model = config['model'](**model_params)

            # Enhanced training for each model type
            if model_name == 'lightgbm':
                # Fixed LightGBM training
                model.fit(
                    X_fold_train, y_fold_train,
                    eval_set=[(X_fold_val, y_fold_val)],
                    callbacks=[
                        lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False),
                        lgb.log_evaluation(0)
                    ]
                )

            elif model_name == 'catboost':
                # Enhanced CatBoost training
                model.fit(
                    X_fold_train, y_fold_train,
                    eval_set=(X_fold_val, y_fold_val),
                    early_stopping_rounds=EARLY_STOPPING_ROUNDS,
                    verbose=False,
                    plot=False
                )

            elif model_name == 'xgboost':
                # Enhanced XGBoost training
                model.fit(
                    X_fold_train, y_fold_train,
                    eval_set=[(X_fold_val, y_fold_val)],
                    early_stopping_rounds=EARLY_STOPPING_ROUNDS,
                    verbose=False
                )
            else:
                # Standard training for other models
                model.fit(X_fold_train, y_fold_train)

            # Get predictions
            if hasattr(model, 'predict_proba'):
                fold_predictions = model.predict_proba(X_fold_val)[:, 1]
            else:
                fold_predictions = model.decision_function(X_fold_val)
                # Convert to probabilities if needed
                from scipy.special import expit
                fold_predictions = expit(fold_predictions)

            # Store predictions and calculate score
            oof_predictions[val_idx] = fold_predictions
            fold_score = roc_auc_score(y_fold_val, fold_predictions)
            cv_scores.append(fold_score)
            models.append(model)

            print(f"AUC: {fold_score:.4f}")

        except Exception as e:
            print(f"❌ Error in fold {fold_idx + 1}: {e}")
            # Use dummy predictions for failed folds
            fold_predictions = np.full(len(val_idx), 0.5)
            oof_predictions[val_idx] = fold_predictions
            cv_scores.append(0.5)
            models.append(None)

    # Calculate overall CV score
    valid_scores = [s for s in cv_scores if s != 0.5]
    if valid_scores:
        overall_score = roc_auc_score(y_train, oof_predictions)
        mean_score = np.mean(valid_scores)
        std_score = np.std(valid_scores)
    else:
        overall_score = 0.5
        mean_score = 0.5
        std_score = 0.0

    print(f"   📊 {model_name} CV Score: {overall_score:.4f} (±{std_score:.4f})")

    return {
        'model_name': model_name,
        'models': models,
        'oof_predictions': oof_predictions,
        'cv_scores': cv_scores,
        'mean_cv_score': mean_score,
        'std_cv_score': std_score,
        'overall_score': overall_score,
        'config': config
    }

# ====================
# 🎯 MAIN TRAINING PIPELINE EXECUTION
# ====================

print("\n" + "="*60)
print("🚀 STARTING GPU-ACCELERATED MULTI-MODEL TRAINING PIPELINE")
print("="*60)

# Prepare training data (with optional sampling for development)
X_train_final, y_train_final = prepare_training_data(
    X_train, y_train,
    development_mode=DEVELOPMENT_MODE,
    sample_size=DEV_SAMPLE_SIZE
)

# Get model configurations
model_configs = get_gpu_optimized_configs()

print(f"📋 Available models: {list(model_configs.keys())}")
print(f"🎯 Training on dataset: {X_train_final.shape}")

# Initialize results storage
trained_models = {}
model_results = {}
all_oof_predictions = {}

# Train each available model
for model_name, config in model_configs.items():
    try:
        # Train model with cross-validation
        result = train_model_cv_enhanced(
            model_name, config, X_train_final, y_train_final, cv_folds
        )

        # Store results
        trained_models[model_name] = result['models']
        model_results[model_name] = {
            'cv_mean': result['mean_cv_score'],
            'cv_std': result['std_cv_score'],
            'overall_score': result['overall_score'],
            'cv_scores': result['cv_scores'],
            'config': result['config']
        }
        all_oof_predictions[model_name] = result['oof_predictions']

        # Memory cleanup for Kaggle
        if pipeline_config.get('IN_KAGGLE', False):
            gc.collect()

    except Exception as e:
        print(f"❌ Failed to train {model_name}: {e}")
        continue

# Train GPU Neural Network if available
if GPU_AVAILABLE and len(cv_folds) > 0:
    print(f"\n🚀 Training GPU Neural Network...")
    nn_results = []
    nn_oof_predictions = np.zeros(len(X_train_final))

    for fold_idx, (train_idx, val_idx) in enumerate(cv_folds):
        print(f"   Neural Network Fold {fold_idx + 1}/{len(cv_folds)}")

        X_fold_train = X_train_final.iloc[train_idx]
        X_fold_val = X_train_final.iloc[val_idx]
        y_fold_train = y_train_final.iloc[train_idx]
        y_fold_val = y_train_final.iloc[val_idx]

        # Train GPU neural network
        nn_model, nn_score = train_gpu_neural_network(
            X_fold_train, y_fold_train, X_fold_val, y_fold_val,
            input_size=X_train_final.shape[1]
        )

        if nn_model is not None:
            # Get predictions
            with torch.no_grad():
                X_val_tensor = torch.FloatTensor(X_fold_val.values).to(DEVICE)
                val_preds = nn_model(X_val_tensor).cpu().numpy().flatten()
                nn_oof_predictions[val_idx] = val_preds

            nn_results.append(nn_score)
        else:
            nn_oof_predictions[val_idx] = 0.5
            nn_results.append(0.5)

    # Store neural network results
    if nn_results and max(nn_results) > 0.5:
        trained_models['neural_network'] = None  # Models are too large to store
        model_results['neural_network'] = {
            'cv_mean': np.mean(nn_results),
            'cv_std': np.std(nn_results),
            'overall_score': roc_auc_score(y_train_final, nn_oof_predictions),
            'cv_scores': nn_results,
            'config': {'gpu_accelerated': True}
        }
        all_oof_predictions['neural_network'] = nn_oof_predictions

# ====================
# 🎯 TRAINING RESULTS SUMMARY
# ====================

print("\n" + "="*60)
print("📊 TRAINING RESULTS SUMMARY")
print("="*60)

# Sort models by performance
performance_ranking = sorted(
    model_results.items(),
    key=lambda x: x[1]['cv_mean'],
    reverse=True
)

print("🏆 MODEL PERFORMANCE RANKING:")
for i, (model_name, results) in enumerate(performance_ranking, 1):
    gpu_indicator = "🚀" if results['config'].get('gpu_accelerated', False) else ""
    print(f"   {i}. {model_name}: {results['cv_mean']:.4f} (±{results['cv_std']:.4f}) {gpu_indicator}")

# Best model identification
best_model_name = performance_ranking[0][0] if performance_ranking else None
best_model_score = performance_ranking[0][1]['cv_mean'] if performance_ranking else 0.0

print(f"\n🥇 BEST MODEL: {best_model_name} with CV AUC: {best_model_score:.4f}")

# ====================
# 🎯 PIPELINE OUTPUT - CREATE VARIABLES FOR CELL 5 COMPATIBILITY
# ====================

print("\n" + "="*60)
print("🏆 CELL 4 COMPLETED - CREATING VARIABLES FOR CELL 5")
print("="*60)

# Create baseline configuration for Cell 5 hyperparameter tuning
baseline_config = {}
optimized_config = {}

for model_name, results in model_results.items():
    # Extract best parameters as baseline
    baseline_config[f"{model_name}_params"] = results['config']['params'].copy()

    # Create optimized config with enhanced parameters for Cell 5 to tune
    if model_name == 'lightgbm':
        optimized_config['lgb_params'] = {
            'objective': 'binary',
            'metric': 'auc',
            'num_leaves': results['config']['params'].get('num_leaves', 100),
            'learning_rate': results['config']['params'].get('learning_rate', 0.05),
            'feature_fraction': results['config']['params'].get('feature_fraction', 0.9),
            'bagging_fraction': results['config']['params'].get('bagging_fraction', 0.9),
            'lambda_l1': results['config']['params'].get('lambda_l1', 0.01),
            'lambda_l2': results['config']['params'].get('lambda_l2', 0.01),
            'random_state': GLOBAL_SEED,
            'verbose': -1
        }
    elif model_name == 'xgboost':
        optimized_config['xgb_params'] = {
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'max_depth': results['config']['params'].get('max_depth', 8),
            'learning_rate': results['config']['params'].get('learning_rate', 0.05),
            'subsample': results['config']['params'].get('subsample', 0.9),
            'colsample_bytree': results['config']['params'].get('colsample_bytree', 0.9),
            'random_state': GLOBAL_SEED,
            'verbosity': 0
        }

# Training metadata for Cell 5
training_metadata = {
    'total_models_trained': len(model_results),
    'best_model': best_model_name,
    'best_cv_score': best_model_score,
    'training_shape': X_train_final.shape,
    'development_mode': DEVELOPMENT_MODE,
    'gpu_available': GPU_AVAILABLE,
    'cv_folds_used': len(cv_folds),
    'feature_count': X_train_final.shape[1]
}

# Variables that Cell 5 expects
oof_predictions_dict = all_oof_predictions  # Out-of-fold predictions for all models
model_performance_summary = model_results   # Performance summary for all models

print(f"✅ Variables created for Cell 5:")
print(f"   📊 trained_models: {len(trained_models)} models")
print(f"   📊 model_results: {len(model_results)} result sets")
print(f"   📊 baseline_config: {len(baseline_config)} configurations")
print(f"   📊 optimized_config: {len(optimized_config)} optimized configurations")
print(f"   📊 oof_predictions_dict: {len(all_oof_predictions)} prediction sets")
print(f"   📊 training_metadata: Complete training summary")

# Additional variables for Cell 5 advanced CV and tuning
X_for_advanced_cv = X_train_final.copy()  # Dataset for Cell 5 CV strategies
y_for_advanced_cv = y_train_final.copy()  # Target for Cell 5 CV strategies
cv_folds_baseline = cv_folds.copy()       # Baseline CV folds for Cell 5

print(f"\n🔗 CELL 5 INTEGRATION READY:")
print(f"   ✅ X_for_advanced_cv: {X_for_advanced_cv.shape}")
print(f"   ✅ y_for_advanced_cv: {len(y_for_advanced_cv)} samples")
print(f"   ✅ cv_folds_baseline: {len(cv_folds_baseline)} folds")
print(f"   ✅ Best baseline AUC: {best_model_score:.4f}")

print(f"\n🚀 READY FOR CELL 5 - ADVANCED CV STRATEGY & HYPERPARAMETER TUNING")
print("="*60)

# Memory cleanup for Kaggle
if pipeline_config.get('IN_KAGGLE', False):
    # Clean up large intermediate variables
    if 'X_train' in locals() and 'X_for_advanced_cv' in locals():
        del X_train
    if 'y_train' in locals() and 'y_for_advanced_cv' in locals():
        del y_train
    gc.collect()

    print(f"🧹 Memory cleanup completed for Kaggle environment")
