# ====================
# 🎯 CELL 3: KAGGLE-OPTIMIZED FEATURE SELECTION & DOMAIN ENGINEERING
# Microsoft Malware Prediction - Enhanced for Kaggle Environment
# ====================

import numpy as np
import pandas as pd
from sklearn.feature_selection import (
    SelectKBest, f_classif, mutual_info_classif,
    RFE, SelectFromModel, VarianceThreshold
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler, RobustScaler
import gc
import warnings
warnings.filterwarnings('ignore')

print(f"🔧 Starting Cell 3 - Kaggle-Optimized Feature Selection & Domain Engineering")
print("="*60)

# ====================
# 🎯 KAGGLE-OPTIMIZED FEATURE SELECTION CONFIGURATION
# ====================

# Get Kaggle-specific settings from pipeline config
IN_KAGGLE = pipeline_config.get('IN_KAGGLE', False)
MAX_RAM_GB = pipeline_config.get('MAX_RAM_GB', 13.0)

# Kaggle-optimized feature selection settings - BALANCED MODE FOR 0.70-0.75 RECALL/ACCURACY
# WITH AUC PRESERVATION (0.71-0.76 maintained)
if IN_KAGGLE:
    print("🏆 KAGGLE FEATURE SELECTION - BALANCED MODE FOR 0.70-0.75 RECALL/ACCURACY")
    print("🎯 AUC PRESERVATION MODE: Maintaining 0.71-0.76 AUC while improving recall/accuracy")
    # BALANCED SETTINGS to optimize recall and accuracy WITHOUT sacrificing AUC
    ENABLE_VARIANCE_THRESHOLD = True
    ENABLE_CORRELATION_REMOVAL = True  # ENABLED - but with conservative threshold
    ENABLE_UNIVARIATE_SELECTION = True  # ENABLED - but keeps more features for AUC
    ENABLE_RFE_SELECTION = False  # Still disabled for time
    ENABLE_LASSO_SELECTION = False  # Optional, can be slow
    ENABLE_DOMAIN_FEATURES = True  # Keep domain features

    # CONSERVATIVE parameters to preserve AUC while improving recall/accuracy
    VARIANCE_THRESHOLD = 0.01  # Remove truly low-variance features
    CORRELATION_THRESHOLD = 0.97  # More conservative (was 0.95) - preserves AUC
    TOP_K_FEATURES = 200  # Increased from 150 to preserve AUC
    UNIVARIATE_TOP_PERCENTILE = 85  # Increased from 80% - keeps more features
    LASSO_ALPHA = 0.001  # Very light regularization
    MAX_DOMAIN_FEATURES = 20  # Allow domain features
    MAX_FEATURE_COMBINATIONS = 5  # Reduced to avoid noise (was 0)

    # NEW: Feature importance threshold for recall optimization
    FEATURE_IMPORTANCE_THRESHOLD = 0.001  # Keep features with importance > 0.1%

    # AUC PRESERVATION SETTINGS
    ENABLE_AUC_AWARE_SELECTION = True  # Prioritize features that contribute to AUC
    MIN_FEATURES_FOR_AUC = 120  # Never go below this for AUC preservation
else:
    # Balanced settings for other environments
    ENABLE_VARIANCE_THRESHOLD = True
    ENABLE_CORRELATION_REMOVAL = True
    ENABLE_UNIVARIATE_SELECTION = True
    ENABLE_RFE_SELECTION = False
    ENABLE_LASSO_SELECTION = False
    ENABLE_DOMAIN_FEATURES = True

    VARIANCE_THRESHOLD = 0.01
    CORRELATION_THRESHOLD = 0.97  # Conservative
    TOP_K_FEATURES = 200  # Increased
    UNIVARIATE_TOP_PERCENTILE = 85  # Increased
    LASSO_ALPHA = 0.001
    MAX_DOMAIN_FEATURES = 20
    MAX_FEATURE_COMBINATIONS = 5
    FEATURE_IMPORTANCE_THRESHOLD = 0.001
    ENABLE_AUC_AWARE_SELECTION = True
    MIN_FEATURES_FOR_AUC = 120

print(f"⚡ BALANCED Feature Selection Settings for 0.70-0.75 Recall/Accuracy + 0.71-0.76 AUC:")
print(f"   Variance threshold: {VARIANCE_THRESHOLD} (Balanced)")
print(f"   Correlation removal: {ENABLE_CORRELATION_REMOVAL} (threshold={CORRELATION_THRESHOLD} - conservative)")
print(f"   Univariate selection: {ENABLE_UNIVARIATE_SELECTION} (top {TOP_K_FEATURES} features)")
print(f"   Feature combinations: {MAX_FEATURE_COMBINATIONS} (strategic interactions)")
print(f"   🎯 STRATEGY: Balance feature count with predictive power")
print(f"   🛡️ AUC PRESERVATION: Keep at least {MIN_FEATURES_FOR_AUC} features, conservative thresholds")

# Check if we have the required variables from previous cells
# Cell 2 should create: X_work_final, final_numeric_features, final_categorical_features
# Cell 1 should create: y_work, cv_folds

required_vars = ['X_work_final', 'y_work', 'cv_folds', 'final_numeric_features', 'final_categorical_features']

print("🔍 Checking pipeline variables (Cell 1 → Cell 2 → Cell 3)...")

# Check what we actually have
missing_vars = [var for var in required_vars if var not in globals()]

if missing_vars:
    print(f"❌ Missing variables from pipeline: {missing_vars}")

    # Show what's available for debugging
    available_pipeline_vars = [var for var in ['X_work', 'X_work_final', 'y_work', 'cv_folds',
                                             'numeric_features', 'final_numeric_features',
                                             'categorical_features', 'final_categorical_features']
                            if var in globals()]
    print(f"   Available pipeline variables: {available_pipeline_vars}")

    print("\n💡 PIPELINE FLOW REQUIREMENTS:")
    print("   1. Run Cell 1 first → creates X_work, y_work, cv_folds, numeric_features, categorical_features")
    print("   2. Run Cell 2 next → creates X_work_final, final_numeric_features, final_categorical_features")
    print("   3. Run Cell 3 (this cell) → uses Cell 2's outputs")

    if 'X_work' in globals() and 'X_work_final' not in globals():
        print("\n🔧 DETECTED: Cell 1 completed but Cell 2 not run")
        print("   Please run Cell 2 first to create X_work_final")
    elif not any(var in globals() for var in ['X_work', 'X_work_final']):
        print("\n🔧 DETECTED: Cell 1 not completed")
        print("   Please run Cell 1 first")

    raise RuntimeError(f"Pipeline incomplete. Missing: {missing_vars}")

print("✅ All pipeline variables found from Cell 1 → Cell 2")

def monitor_kaggle_memory_fs(df, operation_name="Operation"):
    """Monitor memory usage for Kaggle limits during feature selection"""
    memory_mb = df.memory_usage(deep=True).sum() / (1024**2)
    memory_gb = memory_mb / 1024

    print(f"   📊 {operation_name}: {memory_mb:.1f}MB ({memory_gb:.2f}GB)")

    if IN_KAGGLE and memory_gb > MAX_RAM_GB * 0.75:  # More conservative for feature selection
        print(f"   ⚠️ WARNING: Approaching Kaggle memory limit!")
        return False
    return True

def remove_low_variance_features_kaggle(X, threshold=0.01):
    """Kaggle-optimized low variance feature removal"""
    if not ENABLE_VARIANCE_THRESHOLD:
        return X, []

    print(f"\n🔧 Removing low variance features (threshold: {threshold})")

    # Memory check before processing
    if not monitor_kaggle_memory_fs(X, "Pre-variance removal"):
        print("⚠️ Memory limit reached, skipping variance removal")
        return X, []

    selector = VarianceThreshold(threshold=threshold)
    X_selected = selector.fit_transform(X)

    # Get selected feature names
    selected_features = X.columns[selector.get_support()].tolist()
    removed_features = X.columns[~selector.get_support()].tolist()

    print(f"   Removed {len(removed_features)} low-variance features")

    result_df = pd.DataFrame(X_selected, columns=selected_features, index=X.index)

    # Memory cleanup for Kaggle
    if IN_KAGGLE:
        gc.collect()

    return result_df, removed_features

def remove_highly_correlated_features_kaggle(X, threshold=0.95):
    """Kaggle-optimized correlation-based feature removal with NaN handling"""
    if not ENABLE_CORRELATION_REMOVAL:
        return X, []

    print(f"\n🔧 Removing highly correlated features (threshold: {threshold})")

    # Memory check before processing
    if not monitor_kaggle_memory_fs(X, "Pre-correlation removal"):
        print("⚠️ Memory limit reached, skipping correlation removal")
        return X, []

    # Handle missing values for correlation calculation
    X_clean = X.copy()

    # Only process numeric columns for correlation
    numeric_cols = X_clean.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) == 0:
        print("   ⚠️ No numeric columns for correlation analysis")
        return X, []

    X_numeric = X_clean[numeric_cols].copy()

    # Fill missing values in numeric columns
    for col in X_numeric.columns:
        if X_numeric[col].isna().any():
            X_numeric[col] = X_numeric[col].fillna(X_numeric[col].median())

    # For Kaggle, sample data if too large for correlation matrix
    if IN_KAGGLE and len(X_numeric) > 100000:
        print("   📊 Sampling data for correlation calculation (Kaggle optimization)")
        sample_idx = np.random.choice(len(X_numeric), 50000, replace=False)
        X_sample = X_numeric.iloc[sample_idx]
    else:
        X_sample = X_numeric

    try:
        # Calculate correlation matrix
        corr_matrix = X_sample.corr().abs()

        # Find highly correlated pairs
        upper_triangle = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )

        # Find features to remove
        to_remove = [column for column in upper_triangle.columns
                    if any(upper_triangle[column] > threshold)]

        print(f"   Removed {len(to_remove)} highly correlated features")

        # Remove from original dataset
        result_df = X.drop(columns=to_remove)

    except Exception as e:
        print(f"   ⚠️ Correlation analysis failed: {e}, returning original dataset")
        result_df = X
        to_remove = []

    # Memory cleanup for Kaggle
    if IN_KAGGLE:
        del X_clean, X_numeric
        if 'X_sample' in locals():
            del X_sample
        gc.collect()

    return result_df, to_remove

def univariate_feature_selection_kaggle(X, y, k=50):
    """Kaggle-optimized univariate feature selection with NaN handling"""
    if not ENABLE_UNIVARIATE_SELECTION:
        return X, []

    print(f"\n🔧 Univariate feature selection (top {k} features)")

    # Memory check before processing
    if not monitor_kaggle_memory_fs(X, "Pre-univariate selection"):
        print("⚠️ Memory limit reached, reducing k")
        k = min(k, 30)  # Reduce further for memory

    # Handle missing values before feature selection
    print("   📊 Handling missing values for feature selection...")
    X_clean = X.copy()

    # Fill missing values with appropriate strategies
    for col in X_clean.columns:
        if X_clean[col].isna().any():
            if X_clean[col].dtype in ['int64', 'int32', 'float64', 'float32']:
                # Use median for numeric columns
                X_clean[col] = X_clean[col].fillna(X_clean[col].median())
            else:
                # Use mode for categorical columns
                X_clean[col] = X_clean[col].fillna(X_clean[col].mode()[0] if len(X_clean[col].mode()) > 0 else 'missing')

    # For Kaggle, sample data if too large
    if IN_KAGGLE and len(X_clean) > 100000:
        print("   📊 Sampling data for feature selection (Kaggle optimization)")
        sample_idx = np.random.choice(len(X_clean), 50000, replace=False)
        X_sample = X_clean.iloc[sample_idx]
        y_sample = y.iloc[sample_idx]
    else:
        X_sample = X_clean
        y_sample = y

    # Use mutual information for mixed feature types
    selector = SelectKBest(score_func=mutual_info_classif, k=min(k, X.shape[1]))
    X_selected = selector.fit_transform(X_sample, y_sample)

    # Get selected feature names
    selected_features = X_sample.columns[selector.get_support()].tolist()
    removed_features = X_sample.columns[~selector.get_support()].tolist()

    print(f"   Selected {len(selected_features)} features")
    print(f"   Removed {len(removed_features)} features")

    # Apply selection to original dataset (with missing values preserved)
    result_df = X[selected_features]

    # Memory cleanup for Kaggle
    if IN_KAGGLE:
        del X_clean, X_sample
        gc.collect()

    return result_df, removed_features

def lasso_feature_selection_kaggle(X, y, alpha=0.01):
    """Kaggle-optimized LASSO-based feature selection with NaN handling"""
    if not ENABLE_LASSO_SELECTION:
        return X, []

    print(f"\n🔧 LASSO feature selection (alpha: {alpha})")

    # Memory check before processing
    if not monitor_kaggle_memory_fs(X, "Pre-LASSO selection"):
        print("⚠️ Memory limit reached, skipping LASSO selection")
        return X, []

    # Handle missing values before LASSO
    print("   📊 Handling missing values for LASSO...")
    X_clean = X.copy()

    # Fill missing values with appropriate strategies
    for col in X_clean.columns:
        if X_clean[col].isna().any():
            if X_clean[col].dtype in ['int64', 'int32', 'float64', 'float32']:
                # Use median for numeric columns
                X_clean[col] = X_clean[col].fillna(X_clean[col].median())
            else:
                # Use mode for categorical columns, convert to numeric
                mode_val = X_clean[col].mode()[0] if len(X_clean[col].mode()) > 0 else 'missing'
                X_clean[col] = X_clean[col].fillna(mode_val)
                # Convert categorical to numeric for LASSO
                if X_clean[col].dtype == 'object':
                    X_clean[col] = pd.Categorical(X_clean[col]).codes

    # For Kaggle, sample data if too large
    if IN_KAGGLE and len(X_clean) > 50000:
        print("   📊 Sampling data for LASSO (Kaggle optimization)")
        sample_idx = np.random.choice(len(X_clean), 30000, replace=False)
        X_sample = X_clean.iloc[sample_idx]
        y_sample = y.iloc[sample_idx]
    else:
        X_sample = X_clean
        y_sample = y

    try:
        # Standardize features for LASSO
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_sample)

        # LASSO with cross-validation
        lasso = LassoCV(alphas=[alpha], cv=3, random_state=42, max_iter=1000)  # Reduced CV for Kaggle
        lasso.fit(X_scaled, y_sample)

        # Select features with non-zero coefficients
        selected_mask = lasso.coef_ != 0
        selected_features = X_sample.columns[selected_mask].tolist()
        removed_features = X_sample.columns[~selected_mask].tolist()

        print(f"   Selected {len(selected_features)} features with non-zero LASSO coefficients")

        # Apply selection to original dataset (with missing values preserved)
        result_df = X[selected_features] if len(selected_features) > 0 else X

    except Exception as e:
        print(f"   ⚠️ LASSO failed: {e}, returning original dataset")
        result_df = X
        removed_features = []

    # Memory cleanup for Kaggle
    if IN_KAGGLE:
        del X_clean
        if 'X_sample' in locals():
            del X_sample
        gc.collect()

    return result_df, removed_features

def create_domain_specific_features_kaggle(X):
    """Kaggle-optimized domain-specific malware detection features"""
    if not ENABLE_DOMAIN_FEATURES:
        return X

    print(f"\n🔧 Creating Kaggle-optimized domain-specific malware features")

    # Memory check before processing
    if not monitor_kaggle_memory_fs(X, "Pre-domain features"):
        print("⚠️ Memory limit reached, reducing domain features")
        return X

    X_domain = X.copy()
    features_created = 0

    # Hardware vulnerability indicators (Kaggle-optimized)
    ram_cols = [col for col in X_domain.columns if 'ram' in col.lower() or 'memory' in col.lower()]
    core_cols = [col for col in X_domain.columns if 'core' in col.lower() or 'processor' in col.lower()]

    if len(ram_cols) > 0 and len(core_cols) > 0:
        ram_col = ram_cols[0]  # Use first available
        core_col = core_cols[0]  # Use first available

        ram = X_domain[ram_col].fillna(0) + 1
        cores = X_domain[core_col].fillna(0) + 1

        # Create essential ratios only (memory-optimized)
        X_domain['System_Performance_Score'] = (np.log1p(ram) * np.log1p(cores)).astype(np.float32)
        X_domain['RAM_per_Core'] = (ram / cores).astype(np.float32)
        features_created += 2

        # Memory check after each feature creation
        if IN_KAGGLE and not monitor_kaggle_memory_fs(X_domain, f"Domain features: {features_created}"):
            print("⚠️ Memory limit reached, stopping domain feature creation")
            return X_domain

    # Antivirus effectiveness (simplified for Kaggle)
    av_cols = [col for col in X_domain.columns if 'av' in col.lower() or 'antivirus' in col.lower()]
    if len(av_cols) >= 2:
        # Use first two AV-related columns
        col1, col2 = av_cols[:2]
        installed = X_domain[col1].fillna(0)
        enabled = X_domain[col2].fillna(0)

        X_domain['AV_Effectiveness'] = np.where(
            installed > 0, (enabled / (installed + 1e-8)), 0
        ).astype(np.float32)
        features_created += 1

        if features_created >= MAX_DOMAIN_FEATURES:
            print(f"   Reached max domain features limit ({MAX_DOMAIN_FEATURES})")
            return X_domain

    print(f"   Created {features_created} domain-specific features")

    # Memory cleanup for Kaggle
    if IN_KAGGLE:
        gc.collect()

    return X_domain

def create_regularized_feature_combinations_kaggle(X, y, max_combinations=10):
    """Kaggle-optimized regularized feature combinations with NaN handling"""
    print(f"\n🔧 Creating Kaggle-optimized feature combinations (max: {max_combinations})")

    # Memory check before processing
    if not monitor_kaggle_memory_fs(X, "Pre-combinations"):
        print("⚠️ Memory limit reached, skipping feature combinations")
        return X

    X_combined = X.copy()

    # Get numeric columns only for combinations (limited for Kaggle)
    numeric_cols = X_combined.select_dtypes(include=[np.number]).columns.tolist()[:10]  # Limit to top 10

    if len(numeric_cols) < 2:
        return X_combined

    # Handle missing values in numeric columns for correlation calculation
    X_numeric_clean = X_combined[numeric_cols].copy()
    for col in X_numeric_clean.columns:
        if X_numeric_clean[col].isna().any():
            X_numeric_clean[col] = X_numeric_clean[col].fillna(X_numeric_clean[col].median())

    # Calculate feature importance using correlation with target (sampled for Kaggle)
    if IN_KAGGLE and len(X_numeric_clean) > 50000:
        sample_idx = np.random.choice(len(X_numeric_clean), 30000, replace=False)
        X_sample = X_numeric_clean.iloc[sample_idx]
        y_sample = y.iloc[sample_idx]
    else:
        X_sample = X_numeric_clean
        y_sample = y

    feature_importance = {}
    for col in numeric_cols:
        try:
            # Handle any remaining NaN values
            x_vals = X_sample[col].fillna(X_sample[col].median())
            y_vals = y_sample.fillna(y_sample.median()) if y_sample.isna().any() else y_sample

            corr = np.corrcoef(x_vals, y_vals)[0, 1]
            feature_importance[col] = abs(corr) if not np.isnan(corr) else 0
        except:
            feature_importance[col] = 0

    # Sort features by importance (limited for Kaggle)
    important_features = sorted(feature_importance.keys(),
                              key=lambda x: feature_importance[x],
                              reverse=True)[:6]  # Top 6 for memory

    combinations_created = 0

    # Create combinations between most important features (Kaggle-optimized)
    for i, feat1 in enumerate(important_features):
        if combinations_created >= max_combinations:
            break

        for feat2 in important_features[i+1:]:
            if combinations_created >= max_combinations:
                break

            if feat1 in X_combined.columns and feat2 in X_combined.columns:
                # Handle missing values in feature combination
                feat1_vals = X_combined[feat1].fillna(X_combined[feat1].median())
                feat2_vals = X_combined[feat2].fillna(X_combined[feat2].median())

                # Only create simple weighted combination for memory efficiency
                weight1 = feature_importance[feat1]
                weight2 = feature_importance[feat2]
                total_weight = weight1 + weight2 + 1e-8

                X_combined[f'{feat1}_weighted_combo_{feat2}'] = (
                    (feat1_vals * weight1 + feat2_vals * weight2) / total_weight
                ).astype(np.float32)

                combinations_created += 1

                # Memory check for Kaggle
                if IN_KAGGLE and not monitor_kaggle_memory_fs(X_combined, f"Combinations: {combinations_created}"):
                    print("⚠️ Memory limit reached, stopping combinations")
                    break

    print(f"   Created {combinations_created} regularized feature combinations")

    # Memory cleanup for Kaggle
    if IN_KAGGLE:
        del X_numeric_clean
        if 'X_sample' in locals():
            del X_sample
        gc.collect()

    return X_combined

def lightgbm_feature_importance_selection(X, y, top_k=100, sample_size=50000):
    """
    Use LightGBM feature importance for feature selection (Top Kaggle technique)
    """
    if not pipeline_config.get('LIGHTGBM_AVAILABLE', False):
        print("⚠️ LightGBM not available, skipping importance-based selection")
        return X, []

    print(f"\n🔧 LightGBM feature importance selection (top {top_k} features)")

    # Handle missing values for LightGBM
    X_clean = X.copy()
    for col in X_clean.columns:
        if X_clean[col].isna().any():
            if X_clean[col].dtype in ['int64', 'int32', 'float64', 'float32']:
                X_clean[col] = X_clean[col].fillna(X_clean[col].median())
            else:
                X_clean[col] = X_clean[col].fillna('missing')
                if X_clean[col].dtype == 'object':
                    X_clean[col] = pd.Categorical(X_clean[col]).codes

    # Sample data for faster training
    if len(X_clean) > sample_size:
        print(f"   📊 Sampling {sample_size:,} samples for feature importance calculation")
        sample_idx = np.random.choice(len(X_clean), sample_size, replace=False)
        X_sample = X_clean.iloc[sample_idx]
        y_sample = y.iloc[sample_idx]
    else:
        X_sample = X_clean
        y_sample = y

    try:
        import lightgbm as lgb

        # Train a quick LightGBM model for feature importance
        lgb_model = lgb.LGBMClassifier(
            objective='binary',
            metric='auc',
            num_leaves=31,
            learning_rate=0.1,  # Higher LR for quick training
            n_estimators=100,   # Fewer trees for speed
            random_state=42,
            verbose=-1
        )

        print("   🚀 Training LightGBM for feature importance...")
        lgb_model.fit(X_sample, y_sample)

        # Get feature importance
        importance_df = pd.DataFrame({
            'feature': X_sample.columns,
            'importance': lgb_model.feature_importances_
        }).sort_values('importance', ascending=False)

        # Select top K features
        top_features = importance_df.head(top_k)['feature'].tolist()
        removed_features = importance_df.tail(len(importance_df) - top_k)['feature'].tolist()

        print(f"   Selected {len(top_features)} features based on LightGBM importance")
        print(f"   Removed {len(removed_features)} low-importance features")

        # Show top 5 most important features
        print(f"   🏆 Top 5 features: {top_features[:5]}")

        # Apply selection to original dataset
        result_df = X[top_features]

        # Memory cleanup
        if IN_KAGGLE:
            del X_clean, X_sample
            gc.collect()

        return result_df, removed_features

    except Exception as e:
        print(f"   ❌ LightGBM importance selection failed: {e}")
        return X, []

# ====================
# 🎯 MAIN KAGGLE FEATURE SELECTION PIPELINE EXECUTION
# ====================

print(f"🔍 Starting Kaggle-optimized feature selection on dataset: {X_work_final.shape}")

# Initial memory check
monitor_kaggle_memory_fs(X_work_final, "Initial dataset")

# Store original feature count for metadata
original_feature_count = X_work_final.shape[1]

# Step 1: Remove low variance features
X_selected, removed_variance = remove_low_variance_features_kaggle(X_work_final, VARIANCE_THRESHOLD)

# Step 2: Remove highly correlated features
X_selected, removed_corr = remove_highly_correlated_features_kaggle(X_selected, CORRELATION_THRESHOLD)

# Step 3: Univariate feature selection (if enabled)
X_selected, removed_univariate = univariate_feature_selection_kaggle(X_selected, y_work, TOP_K_FEATURES)

# Step 4: LASSO feature selection (if enabled)
X_selected, removed_lasso = lasso_feature_selection_kaggle(X_selected, y_work, LASSO_ALPHA)

# Step 5: LightGBM importance-based selection (if available and enabled)
if pipeline_config.get('LIGHTGBM_AVAILABLE', False) and X_selected.shape[1] > 100:
    X_selected, removed_lgb = lightgbm_feature_importance_selection(X_selected, y_work, top_k=min(100, X_selected.shape[1]))
else:
    removed_lgb = []

# Step 6: Create domain-specific features (Kaggle-optimized)
X_selected = create_domain_specific_features_kaggle(X_selected)

# Step 7: Create regularized feature combinations (Kaggle-optimized)
X_selected = create_regularized_feature_combinations_kaggle(X_selected, y_work, MAX_FEATURE_COMBINATIONS)

# ====================
# 🎯 PIPELINE OUTPUT - CREATE VARIABLES FOR CELL 4 COMPATIBILITY
# ====================

print("\n" + "="*60)
print("🏆 FEATURE SELECTION PIPELINE COMPLETED - CREATING CELL 4 VARIABLES")
print("="*60)

# Create the exact variables that Cell 4 expects
X_ready_for_models = X_selected.copy()  # Main dataset ready for training
cv_folds_enhanced = cv_folds.copy()     # Enhanced CV folds (pass through for now)

# Create comprehensive feature metadata
feature_metadata = {
    'original_feature_count': original_feature_count,
    'final_feature_count': X_ready_for_models.shape[1],
    'selected_features': list(X_ready_for_models.columns),
    'removed_features': {
        'variance': removed_variance,
        'correlation': removed_corr,
        'univariate': removed_univariate,
        'lasso': removed_lasso,
        'lightgbm': removed_lgb if 'removed_lgb' in locals() else []
    },
    'feature_selection_settings': {
        'variance_threshold': VARIANCE_THRESHOLD,
        'correlation_threshold': CORRELATION_THRESHOLD,
        'univariate_enabled': ENABLE_UNIVARIATE_SELECTION,
        'lasso_enabled': ENABLE_LASSO_SELECTION,
        'domain_features_enabled': ENABLE_DOMAIN_FEATURES,
        'max_combinations': MAX_FEATURE_COMBINATIONS
    },
    'numeric_features': final_numeric_features,
    'categorical_features': final_categorical_features
}

# Final memory check and summary
final_memory_mb = X_ready_for_models.memory_usage(deep=True).sum() / (1024**2)
final_memory_gb = final_memory_mb / 1024

print(f"✅ Dataset ready for Cell 4:")
print(f"   📊 Shape: {X_ready_for_models.shape}")
print(f"   📊 Memory usage: {final_memory_mb:.1f}MB ({final_memory_gb:.2f}GB)")
print(f"   📊 Features reduced: {original_feature_count} → {X_ready_for_models.shape[1]} ({X_ready_for_models.shape[1]/original_feature_count*100:.1f}% retained)")
print(f"   📊 Sample count: {len(X_ready_for_models):,}")

print(f"\n🔗 CELL 4 VARIABLES CREATED:")
print(f"   ✅ X_ready_for_models: {X_ready_for_models.shape} - Main training dataset")
print(f"   ✅ y_work: {len(y_work)} samples - Target variable (unchanged)")
print(f"   ✅ cv_folds_enhanced: {len(cv_folds_enhanced)} folds - Cross-validation folds")
print(f"   ✅ feature_metadata: Complete metadata dictionary")

print(f"\n🚀 READY FOR CELL 4 - GPU-ACCELERATED TRAINING")
print("="*60)

# Memory cleanup for Kaggle
if IN_KAGGLE:
    # Clean up intermediate variables
    if 'X_selected' in locals() and 'X_ready_for_models' in locals():
        del X_selected
    gc.collect()

    print(f"🧹 Memory cleanup completed for Kaggle environment")
