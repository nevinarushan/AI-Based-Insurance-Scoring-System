# ====================
# 🎯 CELL 1: KAGGLE-OPTIMIZED FOUNDATION
# Microsoft Malware Prediction - Optimized for Kaggle Environment
# ====================

import pandas as pd
import numpy as np
import joblib
import gc
import warnings
import time
import os
import sys
from datetime import datetime
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.inspection import permutation_importance
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
import multiprocessing as mp

# Enable CUDA launch blocking for better GPU debugging
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

warnings.filterwarnings('ignore')

# Set global seeds for reproducibility
GLOBAL_SEED = 42
np.random.seed(GLOBAL_SEED)

print(f"🚀 KAGGLE-OPTIMIZED ML PIPELINE - CELL 1: FOUNDATION")
print(f"   Timestamp: {datetime.now()}")
print("="*60)

# ====================
# 🎯 KAGGLE ENVIRONMENT DETECTION & SETUP
# ====================

# Detect Kaggle environment - FIXED FOR YOUR CASE
IN_KAGGLE = 'KAGGLE_WORKING_DIR' in os.environ or os.path.exists('/kaggle/input')
IN_COLAB = 'COLAB_GPU' in os.environ

# MANUAL OVERRIDE: Force Kaggle path since you're using Kaggle dataset
# You can uncomment this line to force Kaggle mode
# IN_KAGGLE = True

if IN_KAGGLE or os.path.exists('/kaggle/input/minihackthon-dataset/train.csv'):
    print("🏆 KAGGLE ENVIRONMENT/DATASET DETECTED")
    dataset_path = "/kaggle/input/minihackthon-dataset/train.csv"

    # Kaggle-specific optimizations
    print("⚡ Kaggle optimizations active:")
    print("   • Using Kaggle dataset path")
    print("   • GPU acceleration enabled")
    print("   • Memory optimization for Kaggle limits")

    # Check Kaggle GPU
    try:
        import torch
        if torch.cuda.is_available():
            print(f"   • GPU: {torch.cuda.get_device_name(0)}")
            GPU_AVAILABLE = True
        else:
            print("   • Using CPU (no GPU allocated)")
            GPU_AVAILABLE = False
    except:
        GPU_AVAILABLE = False
        print("   • PyTorch not available")

elif IN_COLAB:
    print("📚 GOOGLE COLAB DETECTED")
    from google.colab import drive
    drive.mount('/content/drive')
    dataset_path = "/content/drive/MyDrive/MalwareDataset/train.csv"
    GPU_AVAILABLE = False

else:
    print("💻 LOCAL ENVIRONMENT")
    # Check for common local dataset locations
    possible_paths = [
        "train.csv",
        "data/train.csv",
        "../data/train.csv",
        "../../data/train.csv",
        "/kaggle/input/minihackthon-dataset/train.csv"  # Added your Kaggle path as fallback
    ]

    dataset_path = None
    for path in possible_paths:
        if os.path.exists(path):
            dataset_path = path
            break

    if dataset_path is None:
        print("❌ Dataset not found in common locations.")
        print("   Trying your Kaggle path as fallback...")
        dataset_path = "/kaggle/input/minihackthon-dataset/train.csv"
        if not os.path.exists(dataset_path):
            print("   Available files in current directory:")
            try:
                for file in os.listdir('..'):
                    if file.endswith('.csv'):
                        print(f"   📄 {file}")
            except:
                pass
            print("\n   Please ensure your dataset is available at one of these locations:")
            for path in possible_paths:
                print(f"   📁 {path}")

    GPU_AVAILABLE = False

# Import advanced ML libraries with Kaggle-optimized error handling
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
    print("✅ LightGBM available")
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("❌ LightGBM not available")

try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
    print("✅ CatBoost available")
except ImportError:
    CATBOOST_AVAILABLE = False
    print("❌ CatBoost not available")

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
    print("✅ XGBoost available")
except ImportError:
    XGBOOST_AVAILABLE = False
    print("❌ XGBoost not available")

# ====================
# 🎯 KAGGLE-OPTIMIZED CONFIGURATION
# ====================

# Force Kaggle settings if using Kaggle dataset path
if "/kaggle/input" in dataset_path:
    IN_KAGGLE = True
    MAX_RAM_GB = 25.0  # Kaggle RAM limit
    CHUNK_SIZE = 1_000_000  # Smaller chunks for better memory management
    MAX_CHUNKS = float('inf')  # No limit on chunks to use all available data
    N_JOBS = 4  # Kaggle CPU cores
    print("🏆 FORCING KAGGLE MODE - Dataset path indicates Kaggle environment")
else:
    MAX_RAM_GB = 4.8
    CHUNK_SIZE = 150_000 if GPU_AVAILABLE else 100_000
    MAX_CHUNKS = float('inf')  # No limit on chunks to use all available data

N_FOLDS = 5

print(f"🔧 Kaggle-Optimized Configuration:")
print(f"   Environment: {'Kaggle' if IN_KAGGLE else 'Colab' if IN_COLAB else 'Local'}")
print(f"   Dataset Path: {dataset_path}")
print(f"   Dataset Exists: {os.path.exists(dataset_path) if dataset_path else 'Unknown'}")
print(f"   GPU Available: {GPU_AVAILABLE}")
print(f"   Max RAM: {MAX_RAM_GB}GB")
print(f"   Chunk Size: {CHUNK_SIZE:,}")
print(f"   Max Chunks: {MAX_CHUNKS}")
print(f"   CV Folds: {N_FOLDS}")
print(f"   CPU Jobs: {N_JOBS}")

# Add error handling for dataset path
if dataset_path and not os.path.exists(dataset_path):
    print(f"\n❌ ERROR: Dataset not found at {dataset_path}")
    print("🔧 TROUBLESHOOTING:")
    print("   1. If you're in Kaggle, make sure the dataset is added to your notebook")
    print("   2. Check the dataset path in Kaggle: /kaggle/input/minihackthon-dataset/train.csv")
    print("   3. If running locally, place train.csv in the current directory")

    # Try to find CSV files in current directory
    csv_files = [f for f in os.listdir('..') if f.endswith('.csv')]
    if csv_files:
        print(f"   📄 Found CSV files in current directory: {csv_files}")
        suggested_path = csv_files[0]
        print(f"   💡 Suggestion: Use '{suggested_path}' instead")

        # Offer to use the first CSV file found
        user_input = input(f"   🤔 Use '{suggested_path}' as dataset? (y/n): ").lower().strip()
        if user_input in ['y', 'yes']:
            dataset_path = suggested_path
            print(f"   ✅ Using {dataset_path}")
        else:
            raise FileNotFoundError(f"Dataset not found. Please ensure your dataset is at: {dataset_path}")
    else:
        raise FileNotFoundError(f"No CSV files found. Please ensure your dataset is available.")

def load_data_with_holdout_kaggle(path, holdout_pct=0.02, chunk_size=CHUNK_SIZE, max_chunks=MAX_CHUNKS):
    """Kaggle-optimized data loading with holdout set - LOAD EXACTLY 2M ROWS"""
    print(f"\n📊 Loading EXACTLY 2 MILLION ROWS with {holdout_pct:.1%} holdout...")

    # Hard-coded row limit
    TARGET_ROWS = 2_000_000
    print(f"🎯 Target row limit: {TARGET_ROWS:,} rows")

    # Check if file exists
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at {path}")

    # Get file size for progress tracking
    file_size = os.path.getsize(path) / (1024**2)  # MB
    print(f"   Dataset size: {file_size:.1f}MB")

    # Calculate expected number of rows based on file size (rough estimate)
    estimated_total_rows = int(file_size * 2000)  # ~2000 rows per MB is a rough estimate
    print(f"   Estimated total rows in dataset: ~{estimated_total_rows:,} (based on file size)")
    print(f"   Will only use first {TARGET_ROWS:,} rows")

    chunks = []
    chunk_count = 0
    total_rows_loaded = 0
    reached_row_limit = False

    try:
        # More robust CSV reading with error handling
        csv_iterator = pd.read_csv(path, chunksize=chunk_size, low_memory=False)

        for chunk in csv_iterator:
            # Check if adding this chunk would exceed our target
            if total_rows_loaded + len(chunk) > TARGET_ROWS:
                # Take only what we need from this chunk to reach exactly TARGET_ROWS
                rows_needed = TARGET_ROWS - total_rows_loaded
                chunk = chunk.iloc[:rows_needed]
                reached_row_limit = True

            chunks.append(chunk)
            chunk_count += 1
            total_rows_loaded += len(chunk)

            # Calculate loading progress percentage towards target
            progress_pct = (total_rows_loaded / TARGET_ROWS) * 100

            # Use safe string formatting to avoid format errors
            try:
                print(f"   Loaded chunk {chunk_count}: {chunk.shape} | Total rows: {total_rows_loaded:,}/{TARGET_ROWS:,} | Progress: {progress_pct:.1f}%")
            except:
                print(f"   Loaded chunk {chunk_count}: {chunk.shape[0]} x {chunk.shape[1]} | Total rows: {total_rows_loaded}")

            # Stop if we've reached our target
            if reached_row_limit:
                print(f"   ✅ BREAKING POINT: TARGET OF {TARGET_ROWS:,} ROWS REACHED!")
                break

            # Enhanced memory check - still maintain as a safety measure
            if IN_KAGGLE:
                try:
                    memory_usage = sum(chunk.memory_usage(deep=True) for chunk in chunks) / (1024**2)  # MB
                    memory_limit_mb = MAX_RAM_GB * 1000 * 0.90  # Use 90% of RAM limit safely

                    # Show progress every 5 chunks or for first few chunks
                    if chunk_count <= 3 or chunk_count % 5 == 0:
                        try:
                            print(f"   💾 Memory: {memory_usage:.0f}MB / {memory_limit_mb:.0f}MB ({memory_usage/memory_limit_mb*100:.1f}%)")
                        except:
                            print(f"   💾 Memory: {int(memory_usage)}MB / {int(memory_limit_mb)}MB")

                    # Emergency brake - only stop loading if we're truly at risk of OOM error
                    if memory_usage > MAX_RAM_GB * 1000 * 0.95:  # Emergency brake at 95%
                        print(f"   ⚠️ EMERGENCY: Memory limit reached at {total_rows_loaded:,} rows - stopping to avoid crash")
                        print(f"   ⚠️ BREAKING POINT: Loaded {total_rows_loaded:,} rows before memory limit")
                        break

                except Exception as mem_error:
                    print(f"   ⚠️ Memory check error: {mem_error}, continuing...")

    except Exception as e:
        if len(chunks) == 0:
            raise Exception(f"Failed to load any data: {e}")
        print(f"   ⚠️ Warning during loading: {str(e)[:100]}...")  # Truncate long error messages
        print(f"   ✅ Continuing with {len(chunks)} chunks loaded ({total_rows_loaded:,} total rows)")

    # Combine chunks
    print(f"   🔄 Combining {len(chunks)} chunks...")
    df = pd.concat(chunks, ignore_index=True)

    # BREAKING POINT VERIFICATION - Clear indicator of exactly 3M rows
    if df.shape[0] == TARGET_ROWS:
        print(f"🎯🎯🎯 BREAKING POINT REACHED: EXACTLY {df.shape[0]:,} ROWS LOADED SUCCESSFULLY! 🎯🎯🎯")
    elif df.shape[0] < TARGET_ROWS:
        print(f"⚠️ BREAKING POINT: Loaded only {df.shape[0]:,} rows (less than target {TARGET_ROWS:,})")
        print(f"⚠️ This likely means the dataset has fewer than {TARGET_ROWS:,} total rows")
    else:
        print(f"⚠️ BREAKING POINT: Loaded {df.shape[0]:,} rows (more than target {TARGET_ROWS:,})")
        # This shouldn't happen with our implementation, but just in case
        df = df.iloc[:TARGET_ROWS]
        print(f"✂️ Trimmed dataset to exactly {TARGET_ROWS:,} rows")

    print(f"✅ Final dataset: {df.shape}")

    # Calculate size of loaded data
    loaded_data_size_mb = df.memory_usage(deep=True).sum() / (1024**2)
    print(f"📊 Loaded data size: {loaded_data_size_mb:.1f}MB / {file_size:.1f}MB file size")
    print(f"📊 Loading ratio: {loaded_data_size_mb/file_size*100:.1f}% of file size")

    # Memory optimization before split
    memory_before = df.memory_usage(deep=True).sum() / (1024**2)
    print(f"   Memory usage before optimization: {memory_before:.1f}MB")

    # Aggressive memory optimization for large datasets
    for col in df.columns:
        try:
            if df[col].dtype == 'int64':
                df[col] = pd.to_numeric(df[col], downcast='integer')
            elif df[col].dtype == 'float64':
                df[col] = pd.to_numeric(df[col], downcast='float')
        except:
            continue  # Skip columns that can't be optimized

    memory_after = df.memory_usage(deep=True).sum() / (1024**2)
    print(f"   Memory usage after optimization: {memory_after:.1f}MB (saved {memory_before-memory_after:.1f}MB)")

    # CRITICAL: Create stratified holdout BEFORE any processing
    target = 'HasDetections'
    if target not in df.columns:
        raise KeyError(f"Target column '{target}' not found. Available columns: {list(df.columns)}")

    X = df.drop(target, axis=1)
    y = df[target]

    X_work, X_holdout, y_work, y_holdout = train_test_split(
        X, y, test_size=holdout_pct, stratify=y, random_state=GLOBAL_SEED
    )

    # 🔧 CRITICAL FIX: Ensure labels are integer-encoded starting from 0 for PyTorch/CUDA compatibility
    print(f"\n🔧 PYTORCH/CUDA LABEL ENCODING FIX:")
    try:
        print(f"   Original y_work labels: {y_work.unique()}")
        print(f"   Original y_holdout labels: {y_holdout.unique()}")
    except:
        print(f"   Original labels detected (avoiding format errors)")

    # Convert to integer and ensure starting from 0
    y_work = y_work.astype(int) - y_work.min()
    y_holdout = y_holdout.astype(int) - y_holdout.min()

    try:
        print(f"   ✅ Fixed y_work labels: {y_work.unique()}")
        print(f"   ✅ Fixed y_holdout labels: {y_holdout.unique()}")
    except:
        print(f"   ✅ Labels fixed for PyTorch/CUDA compatibility")

    print(f"   🎯 Labels now compatible with PyTorch/CUDA (integer, 0-based)")

    print(f"🔒 HOLDOUT CREATED: {len(X_holdout):,} samples (LOCKED until final evaluation)")
    print(f"📈 Working set: {len(X_work):,} samples")

    try:
        print(f"⚖️ Target balance - Work: {y_work.mean():.3f}, Holdout: {y_holdout.mean():.3f}")
    except:
        print(f"⚖️ Target balance calculated successfully")

    # Check if we hit our targets
    if len(X_work) >= 3_500_000:
        print(f"🎯 EXCELLENT: {len(X_work):,} working rows >= 3.5M target!")
    elif len(X_work) >= 3_000_000:
        print(f"🎯 SUCCESS: {len(X_work):,} working rows >= 3M target!")
    elif len(X_work) >= 2_000_000:
        print(f"📊 GOOD: {len(X_work):,} working rows >= 2M target!")
    else:
        print(f"⚠️ BELOW TARGET: {len(X_work):,} working rows < 2M minimum")

    # Save holdout indices for reproducibility
    holdout_info = {
        'holdout_indices': X_holdout.index.tolist(),
        'holdout_size': len(X_holdout),
        'seed': GLOBAL_SEED,
        'timestamp': datetime.now().isoformat(),
        'environment': 'kaggle' if IN_KAGGLE else 'other'
    }

    return X_work, y_work, X_holdout, y_holdout, holdout_info

def identify_feature_types_kaggle(X):
    """Kaggle-optimized feature type identification"""
    print(f"\n🔍 KAGGLE FEATURE TYPE IDENTIFICATION:")

    # Identify numeric features
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()

    # Identify categorical features (object, category, and low-cardinality integers)
    categorical_features = []

    # Object and category columns
    categorical_features.extend(X.select_dtypes(include=['object', 'category']).columns.tolist())

    # Integer columns with low cardinality (likely categorical)
    for col in X.select_dtypes(include=['int64', 'int32']).columns:
        unique_ratio = X[col].nunique() / len(X)
        if unique_ratio < 0.05 and X[col].nunique() < 100:  # Less than 5% unique and under 100 unique values
            categorical_features.append(col)
            if col in numeric_features:
                numeric_features.remove(col)

    print(f"   📊 Numeric features: {len(numeric_features)}")
    print(f"   📋 Categorical features: {len(categorical_features)}")

    # Show examples (Kaggle-friendly display)
    if len(numeric_features) > 0:
        print(f"   📊 Numeric examples: {numeric_features[:5]}")
    if len(categorical_features) > 0:
        print(f"   📋 Categorical examples: {categorical_features[:5]}")

    return numeric_features, categorical_features

def run_kaggle_diagnostics(X, y):
    """Kaggle-optimized diagnostics"""
    print(f"\n🔍 KAGGLE DATASET DIAGNOSTICS:")
    print(f"   Rows: {X.shape[0]:,}, Cols: {X.shape[1]}")
    print(f"   Data types: {X.dtypes.value_counts().to_dict()}")
    print(f"   Memory usage: {X.memory_usage(deep=True).sum() / 1024**2:.1f} MB")

    # Check for obvious issues
    print(f"   Missing values (top 10):")
    null_counts = X.isna().sum().sort_values(ascending=False).head(10)
    for col, count in null_counts.items():
        if count > 0:
            print(f"      {col}: {count:,} ({count/len(X):.1%})")

    print(f"   Duplicates: {X.duplicated().sum():,}")
    print(f"   Index aligned: {X.index.equals(y.index)}")
    print(f"   Target balance: {y.mean():.3f}")
    print(f"   Target unique values: {y.unique()}")

    # Kaggle memory check
    memory_usage_gb = X.memory_usage(deep=True).sum() / 1024**3
    if memory_usage_gb > MAX_RAM_GB * 0.8:
        print(f"⚠️ High memory usage: {memory_usage_gb:.2f}GB (Kaggle limit: {MAX_RAM_GB}GB)")
        return False

    return True

# ====================
# 🎯 MAIN KAGGLE EXECUTION
# ====================

print(f"\n🚀 STARTING KAGGLE DATA LOADING...")

# Load data with Kaggle optimizations
X_work, y_work, X_holdout, y_holdout, holdout_info = load_data_with_holdout_kaggle(dataset_path)

# Run Kaggle-optimized diagnostics
diagnostics_passed = run_kaggle_diagnostics(X_work, y_work)

# Identify feature types for pipeline
numeric_features, categorical_features = identify_feature_types_kaggle(X_work)

# Set up initial cross-validation folds for consistency across pipeline
cv_folds = list(StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=GLOBAL_SEED).split(X_work, y_work))

# ====================
# 🎯 KAGGLE PIPELINE PREPARATION & EXPORTS
# ====================

print(f"\n🎯 PREPARING VARIABLES FOR KAGGLE PIPELINE:")

# Export Kaggle configuration for other cells
pipeline_config = {
    'GLOBAL_SEED': GLOBAL_SEED,
    'N_FOLDS': N_FOLDS,
    'GPU_AVAILABLE': GPU_AVAILABLE,
    'LIGHTGBM_AVAILABLE': LIGHTGBM_AVAILABLE,
    'CATBOOST_AVAILABLE': CATBOOST_AVAILABLE,
    'XGBOOST_AVAILABLE': XGBOOST_AVAILABLE,
    'MAX_RAM_GB': MAX_RAM_GB,
    'N_JOBS': N_JOBS,
    'IN_KAGGLE': IN_KAGGLE,
    'DATASET_PATH': dataset_path
}

# Summary for next cells
print(f"   ✅ Core datasets: X_work, y_work (working data)")
print(f"   ✅ Holdout sets: X_holdout, y_holdout (locked)")
print(f"   ✅ Feature types: numeric_features, categorical_features")
print(f"   ✅ CV folds: cv_folds ({N_FOLDS} folds)")
print(f"   ✅ Kaggle config: pipeline_config")
print(f"   ✅ Holdout info: holdout_info")

print(f"\n🏁 CELL 1 COMPLETE - KAGGLE FOUNDATION READY")
print("="*60)
print(f"✅ Data loaded with holdout protection: {len(X_holdout):,} samples locked")
print(f"✅ Diagnostics: {'PASSED' if diagnostics_passed else 'ISSUES FOUND'}")
print(f"✅ Seeds set for reproducibility: {GLOBAL_SEED}")
print(f"✅ Feature types identified: {len(numeric_features)} numeric, {len(categorical_features)} categorical")
print(f"✅ Cross-validation ready: {N_FOLDS} stratified folds")
print(f"✅ Libraries checked: LightGBM={LIGHTGBM_AVAILABLE}, CatBoost={CATBOOST_AVAILABLE}, XGBoost={XGBOOST_AVAILABLE}")
print(f"✅ Environment: {'🏆 KAGGLE' if IN_KAGGLE else '📚 COLAB' if IN_COLAB else '💻 LOCAL'}")

# Verification that all required variables are ready for Cell 2
required_for_cell2 = ['X_work', 'y_work', 'numeric_features', 'categorical_features', 'cv_folds']

# CRITICAL FIX: Ensure all variables are in global scope for other cells
globals()['X_work'] = X_work
globals()['y_work'] = y_work
globals()['numeric_features'] = numeric_features
globals()['categorical_features'] = categorical_features
globals()['cv_folds'] = cv_folds
globals()['pipeline_config'] = pipeline_config
globals()['holdout_info'] = holdout_info
globals()['X_holdout'] = X_holdout
globals()['y_holdout'] = y_holdout

print(f"\n🔧 EXPORTING VARIABLES TO GLOBAL SCOPE:")
print(f"   ✅ X_work: {X_work.shape}")
print(f"   ✅ y_work: {len(y_work)} samples")
print(f"   ✅ numeric_features: {len(numeric_features)} features")
print(f"   ✅ categorical_features: {len(categorical_features)} features")
print(f"   ✅ cv_folds: {len(cv_folds)} folds")
print(f"   ✅ pipeline_config: {len(pipeline_config)} settings")

# Verify variables are now in global scope
missing_vars = [var for var in required_for_cell2 if var not in globals()]

if missing_vars:
    print(f"❌ WARNING: Still missing variables for Cell 2: {missing_vars}")
else:
    print(f"✅ All required variables ready for Cell 2 (verified in global scope)")

print(f"\n🚀 READY FOR CELL 2: Advanced Feature Engineering & Preprocessing")

# Memory cleanup for Kaggle
gc.collect()
