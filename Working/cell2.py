# ====================
# 🎯 CELL 2: KAGGLE-OPTIMIZED FEATURE ENGINEERING
# Microsoft Malware Prediction - Enhanced for Kaggle Environment
# ====================

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
import gc
import warnings
warnings.filterwarnings('ignore')

print(f"🔍 Starting Cell 2 - Kaggle-Optimized Feature Engineering")
print("="*60)

# ====================
# 🎯 KAGGLE-OPTIMIZED CONFIGURATION
# ====================

# Get Kaggle-specific settings from Cell 1
IN_KAGGLE = pipeline_config.get('IN_KAGGLE', False)
MAX_RAM_GB = pipeline_config.get('MAX_RAM_GB', 13.0)

# Kaggle-optimized feature engineering settings - BALANCED FOR 0.70-0.75 RECALL/ACCURACY
if IN_KAGGLE:
    print("🏆 KAGGLE OPTIMIZATIONS ACTIVE - BALANCED MODE FOR 0.70-0.75 RECALL/ACCURACY")
    # BALANCED SETTINGS to optimize recall and accuracy (not just AUC)
    ENABLE_TARGET_ENCODING = True
    ENABLE_FREQUENCY_ENCODING = True
    ENABLE_INTERACTION_FEATURES = True  # ENABLED - helps capture patterns for recall
    ENABLE_BINNING = True  # ENABLED - helps with decision boundaries for accuracy
    REGULARIZATION_ALPHA = 0.1  # Moderate regularization to prevent overfitting
    HIGH_CARDINALITY_THRESHOLD = 30  # Lower to reduce noise, improve accuracy
    MAX_INTERACTIONS = 10  # Limited interactions for important features
    MAX_BINNED_FEATURES = 5  # Strategic binning for key numeric features

    # NEW: Class balance awareness for better recall
    HANDLE_CLASS_IMBALANCE = True
    TARGET_BALANCE_RATIO = 0.5  # Target ratio for minority class sampling
else:
    # Original settings for other environments
    ENABLE_TARGET_ENCODING = True
    ENABLE_FREQUENCY_ENCODING = True
    ENABLE_INTERACTION_FEATURES = True
    ENABLE_BINNING = True
    REGULARIZATION_ALPHA = 0.1
    HIGH_CARDINALITY_THRESHOLD = 30
    MAX_INTERACTIONS = 10
    MAX_BINNED_FEATURES = 5
    HANDLE_CLASS_IMBALANCE = True
    TARGET_BALANCE_RATIO = 0.5

print(f"⚡ BALANCED Feature Engineering Settings for 0.70-0.75 Recall/Accuracy:")
print(f"   High cardinality threshold: {HIGH_CARDINALITY_THRESHOLD} (Noise reduction)")
print(f"   Interactions: {ENABLE_INTERACTION_FEATURES} (Pattern capture)")
print(f"   Binning: {ENABLE_BINNING} (Decision boundaries)")
print(f"   Regularization alpha: {REGULARIZATION_ALPHA} (Balanced)")
print(f"   Class imbalance handling: {HANDLE_CLASS_IMBALANCE}")
print(f"   🎯 TARGET: 0.70-0.75 Recall + Accuracy (not just AUC)")

def monitor_kaggle_memory(df, operation_name="Operation"):
    """Monitor memory usage for Kaggle limits"""
    memory_mb = df.memory_usage(deep=True).sum() / (1024**2)
    memory_gb = memory_mb / 1024

    print(f"   📊 {operation_name}: {memory_mb:.1f}MB ({memory_gb:.2f}GB)")

    if IN_KAGGLE and memory_gb > MAX_RAM_GB * 0.8:
        print(f"   ⚠️ WARNING: Approaching Kaggle memory limit!")
        return False
    return True

def audit_feature_leakage(X, y, threshold_critical=0.9, threshold_warning=0.7):
    """Detect potential feature leakage before processing"""
    print("🔍 LEAKAGE AUDIT - Critical for 0.75+ AUC")
    print("="*50)

    leakage_found = []
    warnings_found = []

    # Check numeric correlations (Kaggle-optimized)
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        print(f"📊 Checking {len(numeric_cols)} numeric features...")

        # Sample for memory efficiency in Kaggle
        sample_size = min(50000, len(X)) if IN_KAGGLE else len(X)
        sample_idx = np.random.choice(len(X), sample_size, replace=False)
        X_sample = X.iloc[sample_idx]
        y_sample = y.iloc[sample_idx]

        for col in numeric_cols:
            try:
                # Handle missing values
                mask = ~(X_sample[col].isna() | y_sample.isna())
                if mask.sum() < 100:  # Skip if too few valid values
                    continue

                corr = np.corrcoef(X_sample.loc[mask, col], y_sample.loc[mask])[0, 1]
                if not np.isnan(corr):
                    abs_corr = abs(corr)
                    if abs_corr > threshold_critical:
                        leakage_found.append((col, abs_corr))
                        print(f"❌ CRITICAL LEAKAGE: {col} = {corr:.4f}")
                    elif abs_corr > threshold_warning:
                        warnings_found.append((col, abs_corr))
                        print(f"⚠️ HIGH CORRELATION: {col} = {corr:.4f}")

            except Exception as e:
                print(f"⚠️ Could not check {col}: {e}")

    # Check suspicious feature names
    suspicious_keywords = ['target', 'label', 'hasdetections', 'malware', 'detection', 'outcome', 'result']
    for col in X.columns:
        if any(keyword in col.lower() for keyword in suspicious_keywords):
            print(f"⚠️ SUSPICIOUS NAME: {col}")
            warnings_found.append((col, 'suspicious_name'))

    # Check for ID-like features (high cardinality, unique values)
    for col in X.columns:
        if X[col].dtype == 'object' or X[col].dtype == 'int64':
            unique_ratio = X[col].nunique() / len(X)
            if unique_ratio > 0.8:  # >80% unique values
                print(f"⚠️ POTENTIAL ID: {col} ({unique_ratio:.1%} unique)")
                warnings_found.append((col, 'potential_id'))

    print(f"\n🎯 LEAKAGE AUDIT RESULTS:")
    print(f"   Critical issues: {len(leakage_found)}")
    print(f"   Warnings: {len(warnings_found)}")

    if leakage_found:
        print("❌ CRITICAL: Fix these before continuing!")
        return False, leakage_found, warnings_found
    else:
        print("✅ No critical leakage detected")
        return True, leakage_found, warnings_found

def frequency_based_pruning(X, categorical_features, min_frequency=20, rare_label="__rare__"):
    """
    Apply frequency-based pruning to reduce noise from rare categories - OPTIMIZED FOR RECALL/ACCURACY
    Increased min_frequency from 10 to 20 for better noise reduction
    """
    print(f"\n🔧 FREQUENCY-BASED PRUNING (min_frequency: {min_frequency})")
    print(f"   📊 Processing {len(categorical_features)} categorical columns on {len(X):,} rows")
    print(f"   🎯 Balanced mode: Optimize for recall + accuracy (not just AUC)")

    X_pruned = X.copy()
    pruning_stats = {}

    # Performance optimization: Sample for very large datasets to estimate pruning
    SAMPLE_SIZE = 100000
    USE_SAMPLING = len(X) > SAMPLE_SIZE and IN_KAGGLE

    # ADJUSTED: More aggressive pruning for noise reduction (helps accuracy)
    # Identify potentially important categorical features that need careful handling
    important_keywords = ['product', 'engine', 'os', 'version', 'av', 'defender', 'architecture', 'census']
    high_value_features = [col for col in categorical_features
                          if any(keyword in col.lower() for keyword in important_keywords)]

    print(f"   🎯 Identified {len(high_value_features)} high-value features for careful pruning")

    if USE_SAMPLING:
        print(f"   ⚡ Large dataset detected - using performance-preserving sampling")
        sample_idx = np.random.choice(len(X), SAMPLE_SIZE, replace=False)
        X_sample = X.iloc[sample_idx]

    for i, col in enumerate(categorical_features, 1):
        if col not in X_pruned.columns:
            continue

        print(f"   Processing {col} ({i}/{len(categorical_features)})...", end=" ", flush=True)

        # SAFEGUARD 2: Adaptive pruning thresholds based on feature importance
        current_min_freq = min_frequency
        if col in high_value_features:
            # More conservative pruning for important features
            current_min_freq = max(5, min_frequency // 2)  # At least 5, but more lenient
            print(f"[HIGH-VALUE: min_freq={current_min_freq}]...", end=" ", flush=True)

        try:
            # Performance optimization: Use sampling for initial estimation on large datasets
            if USE_SAMPLING and len(X) > SAMPLE_SIZE:
                # Get value counts on sample first to identify likely rare categories
                sample_counts = X_sample[col].value_counts()

                # SAFEGUARD 3: Conservative estimation - use stricter threshold for sampling
                # This ensures we don't accidentally prune important categories
                sample_threshold = max(2, (current_min_freq * SAMPLE_SIZE / len(X)) * 0.5)  # 50% safety margin
                sample_rare = sample_counts[sample_counts < sample_threshold].index

                # Only do full count if we found potential rare categories
                if len(sample_rare) > 0:
                    # Get full value counts only for verification
                    value_counts = X_pruned[col].value_counts()
                else:
                    # No rare categories detected in sample, skip this column
                    print("✓ (no rare categories)")
                    continue
            else:
                # Standard approach for smaller datasets
                value_counts = X_pruned[col].value_counts()

            # SAFEGUARD 4: Information-preserving pruning
            rare_categories = value_counts[value_counts < current_min_freq].index.tolist()

            # SAFEGUARD 5: Don't over-prune high-cardinality features
            original_cardinality = len(value_counts)
            if original_cardinality > 100:  # High cardinality feature
                # Limit pruning to avoid losing too much information
                max_prune_ratio = 0.80  # Don't prune more than 80% of categories
                max_categories_to_prune = int(original_cardinality * max_prune_ratio)

                if len(rare_categories) > max_categories_to_prune:
                    # Sort by frequency and only prune the rarest
                    rare_counts = value_counts[rare_categories].sort_values()
                    rare_categories = rare_counts.head(max_categories_to_prune).index.tolist()
                    print(f"[LIMITED-PRUNE: {max_categories_to_prune}/{len(rare_counts)}]...", end=" ", flush=True)

            if len(rare_categories) > 0:
                # SAFEGUARD 6: Efficient but careful replacement
                # Use vectorized operations but preserve data integrity
                mask = X_pruned[col].isin(rare_categories)
                rare_samples_count = mask.sum()

                # Don't prune if it affects too many samples (could hurt performance)
                if rare_samples_count > len(X) * 0.5:  # More than 50% of data
                    print("✓ (skipped: affects >50% of data)")
                    continue

                X_pruned.loc[mask, col] = rare_label

                pruning_stats[col] = {
                    'rare_categories_count': len(rare_categories),
                    'rare_samples_count': rare_samples_count,
                    'original_cardinality': original_cardinality,
                    'new_cardinality': X_pruned[col].nunique(),
                    'pruning_ratio': len(rare_categories) / original_cardinality,
                    'samples_affected_ratio': rare_samples_count / len(X),
                    'high_value_feature': col in high_value_features
                }

                print(f"✓ {len(rare_categories)} rare → '{rare_label}' ({rare_samples_count:,} samples, {len(rare_categories)/original_cardinality:.1%} pruned)")
            else:
                print("✓ (no rare categories)")

        except Exception as e:
            print(f"✗ Error: {str(e)[:50]}...")
            continue

        # Memory management for large datasets
        if IN_KAGGLE and i % 5 == 0:  # Every 5 columns
            gc.collect()

    # SAFEGUARD 7: Performance impact assessment
    total_rare_categories = sum(stats['rare_categories_count'] for stats in pruning_stats.values())
    total_samples_affected = sum(stats['rare_samples_count'] for stats in pruning_stats.values())
    high_value_processed = sum(1 for stats in pruning_stats.values() if stats['high_value_feature'])

    print(f"\n   🎯 PERFORMANCE-PRESERVING PRUNING SUMMARY:")
    print(f"   📊 Total rare categories pruned: {total_rare_categories:,}")
    print(f"   📈 Total samples affected: {total_samples_affected:,} ({total_samples_affected/len(X):.2%})")
    print(f"   🏆 High-value features processed: {high_value_processed}/{len(high_value_features)}")
    print(f"   ✅ Pruning optimized for 0.75+ AUC maintenance")

    # SAFEGUARD 8: Warning if excessive pruning detected
    if total_samples_affected > len(X) * 0.3:  # More than 30% of data affected
        print(f"   ⚠️ WARNING: High pruning impact detected ({total_samples_affected/len(X):.1%} of data)")
        print(f"   💡 Consider increasing min_frequency to preserve more information")

    # Final memory cleanup
    if IN_KAGGLE:
        gc.collect()

    return X_pruned, pruning_stats

def advanced_categorical_encoding(X, y, cv_folds, categorical_features):
    """Kaggle-optimized categorical encoding with memory management"""
    print("\n🎯 KAGGLE-OPTIMIZED CATEGORICAL ENCODING")
    print("="*50)

    X_encoded = X.copy()
    encoding_mappings = {}

    # Memory check before starting
    if not monitor_kaggle_memory(X_encoded, "Pre-encoding"):
        print("⚠️ Memory limit reached, reducing categorical features")
        categorical_features = categorical_features[:10]  # Limit features for Kaggle

    for col in categorical_features:
        if col not in X_encoded.columns:
            continue

        cardinality = X_encoded[col].nunique()
        print(f"   Encoding {col} (cardinality: {cardinality})")

        # Convert to string to handle mixed types
        X_encoded[col] = X_encoded[col].astype(str).fillna('missing')

        if cardinality <= 2:
            # Binary encoding for low cardinality
            le = LabelEncoder()
            X_encoded[col] = le.fit_transform(X_encoded[col])
            encoding_mappings[col] = {'type': 'label', 'encoder': le}

        elif cardinality <= HIGH_CARDINALITY_THRESHOLD:
            # Multi-encoding for medium cardinality
            if ENABLE_FREQUENCY_ENCODING:
                # Frequency encoding
                freq_map = X_encoded[col].value_counts().to_dict()
                X_encoded[f'{col}_freq'] = X_encoded[col].map(freq_map).astype(np.float32)

            # Target encoding with regularization
            if ENABLE_TARGET_ENCODING:
                target_encoded = target_encode_with_regularization(
                    X_encoded[col], y, cv_folds, alpha=REGULARIZATION_ALPHA
                )
                X_encoded[f'{col}_target'] = target_encoded.astype(np.float32)

            # Keep original as label encoded
            le = LabelEncoder()
            X_encoded[col] = le.fit_transform(X_encoded[col])
            encoding_mappings[col] = {'type': 'multi', 'encoder': le}

        else:
            # High cardinality features - Kaggle-optimized handling
            print(f"      High cardinality ({cardinality}) - using regularized encoding")

            if ENABLE_TARGET_ENCODING:
                target_encoded = target_encode_with_regularization(
                    X_encoded[col], y, cv_folds, alpha=REGULARIZATION_ALPHA * 2
                )
                X_encoded[f'{col}_target'] = target_encoded.astype(np.float32)

            if ENABLE_FREQUENCY_ENCODING and IN_KAGGLE:
                # More aggressive binning for Kaggle memory limits
                freq_map = X_encoded[col].value_counts().to_dict()
                freq_encoded = X_encoded[col].map(freq_map)
                X_encoded[f'{col}_freq_binned'] = pd.qcut(
                    freq_encoded, q=5, labels=False, duplicates='drop'  # Reduced bins for memory
                ).fillna(0).astype(np.uint8)

            # Drop original high-cardinality column
            X_encoded.drop(col, axis=1, inplace=True)
            encoding_mappings[col] = {'type': 'high_card', 'dropped': True}

        # Memory check after each encoding
        if IN_KAGGLE and not monitor_kaggle_memory(X_encoded, f"After encoding {col}"):
            print(f"⚠️ Memory limit reached, stopping categorical encoding")
            break

    # Force garbage collection for Kaggle
    if IN_KAGGLE:
        gc.collect()

    return X_encoded, encoding_mappings

def target_encode_with_regularization(series, y, cv_folds, alpha=0.1):
    """Target encoding with cross-validation and regularization"""
    encoded = np.zeros(len(series), dtype=np.float32)  # Use float32 for memory
    global_mean = y.mean()

    for fold_idx, (train_idx, val_idx) in enumerate(cv_folds):
        # Calculate target statistics on training fold only
        train_series = series.iloc[train_idx]
        train_y = y.iloc[train_idx]

        # Group statistics with regularization
        stats = pd.DataFrame({
            'sum': train_y.groupby(train_series).sum(),
            'count': train_y.groupby(train_series).count()
        }).fillna(0)

        # Regularized (smoothed) target encoding
        regularized_mean = (stats['sum'] + alpha * global_mean) / (stats['count'] + alpha)

        # Apply to validation fold
        val_encoded = series.iloc[val_idx].map(regularized_mean).fillna(global_mean)
        encoded[val_idx] = val_encoded.astype(np.float32)

    return encoded

def create_interaction_features_kaggle(X, numeric_features, max_interactions=MAX_INTERACTIONS):
    """Kaggle-optimized interaction feature creation"""
    if not ENABLE_INTERACTION_FEATURES or len(numeric_features) < 2:
        return X

    print(f"\n🔧 Creating Kaggle-optimized interaction features (max: {max_interactions})")

    X_interactions = X.copy()
    interactions_created = 0

    # Memory check before starting
    if not monitor_kaggle_memory(X_interactions, "Pre-interactions"):
        print("⚠️ Skipping interactions due to memory limits")
        return X_interactions

    # Create interactions between most important features (reduced for Kaggle)
    important_features = numeric_features[:min(4, len(numeric_features))]  # Top 4 for memory

    for i, feat1 in enumerate(important_features):
        if interactions_created >= max_interactions:
            break

        for feat2 in important_features[i+1:]:
            if interactions_created >= max_interactions:
                break

            if feat1 in X_interactions.columns and feat2 in X_interactions.columns:
                # Only multiplication interaction for memory efficiency
                X_interactions[f'{feat1}_x_{feat2}'] = (
                    X_interactions[feat1] * X_interactions[feat2]
                ).astype(np.float32)

                interactions_created += 1

                # Memory check after each interaction
                if IN_KAGGLE and not monitor_kaggle_memory(X_interactions, f"Interaction {interactions_created}"):
                    print("⚠️ Memory limit reached, stopping interactions")
                    break

    print(f"   Created {interactions_created} interaction features")

    # Garbage collection for Kaggle
    if IN_KAGGLE:
        gc.collect()

    return X_interactions

def create_binned_features_kaggle(X, numeric_features, n_bins=3):  # Reduced bins for memory
    """Kaggle-optimized binned feature creation"""
    if not ENABLE_BINNING:
        return X

    print(f"\n🔧 Creating Kaggle-optimized binned features ({n_bins} bins per feature)")

    X_binned = X.copy()
    features_binned = 0

    for col in numeric_features[:MAX_BINNED_FEATURES]:  # Limit for memory
        if col in X_binned.columns:
            try:
                # Create quantile-based bins with reduced bins for memory
                X_binned[f'{col}_binned'] = pd.qcut(
                    X_binned[col], q=n_bins, labels=False, duplicates='drop'
                ).fillna(n_bins // 2).astype(np.uint8)

                features_binned += 1

                # Memory check for Kaggle
                if IN_KAGGLE and not monitor_kaggle_memory(X_binned, f"Binned {features_binned} features"):
                    print("⚠️ Memory limit reached, stopping binning")
                    break

            except Exception as e:
                print(f"   Warning: Could not bin {col}: {e}")

    print(f"   Created {features_binned} binned features")

    # Garbage collection for Kaggle
    if IN_KAGGLE:
        gc.collect()

    return X_binned

def safe_dtype_optimization_kaggle(df):
    """Kaggle-optimized data type optimization"""
    print("\n🔧 KAGGLE DTYPE OPTIMIZATION:")

    original_memory = df.memory_usage(deep=True).sum() / 1024**2

    for col in df.columns:
        if df[col].dtype == 'int64':
            col_min = df[col].min()
            col_max = df[col].max()

            if col_min >= 0 and col_max < 255:
                df[col] = df[col].astype('uint8')
            elif col_min >= -128 and col_max < 127:
                df[col] = df[col].astype('int8')
            elif col_min >= -32768 and col_max < 32767:
                df[col] = df[col].astype('int16')
            else:
                df[col] = df[col].astype('int32')

        elif df[col].dtype == 'float64':
            df[col] = df[col].astype('float32')

        elif df[col].dtype == 'object':
            # Convert to category if reasonable cardinality
            unique_ratio = df[col].nunique() / len(df)
            if unique_ratio < 0.5:  # Less than 50% unique
                df[col] = df[col].astype('category')

    optimized_memory = df.memory_usage(deep=True).sum() / 1024**2
    print(f"   Memory: {original_memory:.1f}MB → {optimized_memory:.1f}MB ({optimized_memory/original_memory:.1%})")

    # Kaggle memory check
    monitor_kaggle_memory(df, "After optimization")

    # Force garbage collection
    if IN_KAGGLE:
        gc.collect()

    return df

def create_advanced_feature_engineering(X, categorical_features, numeric_features):
    """
    Create advanced features: cross-categorical, missingness flags, and ratios
    """
    print(f"\n🔧 ADVANCED FEATURE ENGINEERING")
    print("="*50)

    X_advanced = X.copy()
    features_created = 0

    # 1. MISSINGNESS FLAGS (highly predictive for malware)
    print("   📊 Creating missingness flags...")
    missing_flags_created = 0
    for col in X.columns:
        if X[col].isna().any():
            X_advanced[f'{col}_is_missing'] = X[col].isna().astype(np.uint8)
            missing_flags_created += 1

    print(f"      Created {missing_flags_created} missingness flags")
    features_created += missing_flags_created

    # 2. CROSS-CATEGORICAL FEATURES (OSVersion × AVProduct combinations)
    print("   🔗 Creating cross-categorical features...")
    cross_cat_created = 0

    # Important cross-categorical combinations for malware detection
    cross_combinations = []

    # Find OS-related columns
    os_cols = [col for col in categorical_features if 'os' in col.lower() or 'version' in col.lower()]
    av_cols = [col for col in categorical_features if 'av' in col.lower() or 'product' in col.lower()]
    census_cols = [col for col in categorical_features if 'census' in col.lower()]

    # Create meaningful cross-categorical features
    if len(os_cols) > 0 and len(av_cols) > 0:
        os_col = os_cols[0]
        av_col = av_cols[0]
        if os_col in X_advanced.columns and av_col in X_advanced.columns:
            # Create cross-categorical feature
            X_advanced[f'{os_col}_X_{av_col}'] = (
                X_advanced[os_col].astype(str) + '_cross_' + X_advanced[av_col].astype(str)
            )
            cross_combinations.append(f'{os_col}_X_{av_col}')
            cross_cat_created += 1

    if len(census_cols) >= 2:
        col1, col2 = census_cols[:2]
        if col1 in X_advanced.columns and col2 in X_advanced.columns:
            X_advanced[f'{col1}_X_{col2}'] = (
                X_advanced[col1].astype(str) + '_cross_' + X_advanced[col2].astype(str)
            )
            cross_combinations.append(f'{col1}_X_{col2}')
            cross_cat_created += 1

    print(f"      Created {cross_cat_created} cross-categorical features")
    features_created += cross_cat_created

    # 3. ADVANCED RATIOS (AV effectiveness, system ratios)
    print("   📈 Creating advanced ratio features...")
    ratios_created = 0

    # AV effectiveness ratios - only use numeric columns
    av_installed_cols = [col for col in numeric_features if 'installed' in col.lower() and 'av' in col.lower()]
    av_enabled_cols = [col for col in numeric_features if 'enabled' in col.lower() and 'av' in col.lower()]

    if len(av_installed_cols) > 0 and len(av_enabled_cols) > 0:
        installed_col = av_installed_cols[0]
        enabled_col = av_enabled_cols[0]

        if installed_col in X_advanced.columns and enabled_col in X_advanced.columns:
            try:
                # Ensure columns are numeric
                X_advanced[installed_col] = pd.to_numeric(X_advanced[installed_col], errors='coerce').fillna(0)
                X_advanced[enabled_col] = pd.to_numeric(X_advanced[enabled_col], errors='coerce').fillna(0)

                # AV effectiveness ratio
                X_advanced['AV_Effectiveness_Ratio'] = np.where(
                    X_advanced[installed_col] > 0,
                    X_advanced[enabled_col] / (X_advanced[installed_col] + 1e-8),
                    0
                ).astype(np.float32)
                ratios_created += 1

                # AV gap (security vulnerability indicator)
                X_advanced['AV_Security_Gap'] = np.maximum(
                    0, X_advanced[installed_col] - X_advanced[enabled_col]
                ).astype(np.float32)
                ratios_created += 1
            except Exception as e:
                print(f"      Warning: Could not create AV ratios: {e}")

    # System resource ratios - only use numeric columns
    ram_cols = [col for col in numeric_features if 'ram' in col.lower() or 'memory' in col.lower()]
    storage_cols = [col for col in numeric_features if 'disk' in col.lower() or 'storage' in col.lower()]

    if len(ram_cols) > 0 and len(storage_cols) > 0:
        ram_col = ram_cols[0]
        storage_col = storage_cols[0]

        if ram_col in X_advanced.columns and storage_col in X_advanced.columns:
            try:
                # Ensure columns are numeric
                X_advanced[ram_col] = pd.to_numeric(X_advanced[ram_col], errors='coerce').fillna(0)
                X_advanced[storage_col] = pd.to_numeric(X_advanced[storage_col], errors='coerce').fillna(0)

                # Memory to storage ratio (system profile indicator)
                X_advanced['Memory_Storage_Ratio'] = np.where(
                    X_advanced[storage_col] > 0,
                    X_advanced[ram_col] / (X_advanced[storage_col] + 1e-8),
                    0
                ).astype(np.float32)
                ratios_created += 1
            except Exception as e:
                print(f"      Warning: Could not create system ratios: {e}")

    print(f"      Created {ratios_created} advanced ratio features")
    features_created += ratios_created

    # 4. FREQUENCY/COUNT FEATURES for important categoricals
    print("   🔢 Creating frequency/count features...")
    freq_created = 0

    # Create frequency features for most important categorical columns
    important_categoricals = ['ProductName', 'EngineVersion', 'OSVersion', 'Census_OSArchitecture']

    for col in important_categoricals:
        matching_cols = [c for c in categorical_features if col.lower() in c.lower()]
        if len(matching_cols) > 0:
            actual_col = matching_cols[0]
            if actual_col in X_advanced.columns:
                try:
                    # Create frequency count feature
                    freq_map = X_advanced[actual_col].value_counts().to_dict()
                    X_advanced[f'{actual_col}_frequency_count'] = X_advanced[actual_col].map(freq_map).astype(np.float32)
                    freq_created += 1
                except Exception as e:
                    print(f"      Warning: Could not create frequency feature for {actual_col}: {e}")

    print(f"      Created {freq_created} frequency/count features")
    features_created += freq_created

    # 5. SYSTEM VULNERABILITY INDICATORS (malware-specific)
    print("   🛡️ Creating system vulnerability indicators...")
    vuln_created = 0

    # Security configuration weakness indicators - only use numeric columns
    security_keywords = ['firewall', 'defender', 'uac', 'smartscreen']
    security_cols = []

    # Find security-related columns that are actually numeric
    for col in X_advanced.columns:
        if any(sec in col.lower() for sec in security_keywords):
            # Check if column is numeric or can be converted to numeric
            try:
                pd.to_numeric(X_advanced[col], errors='coerce')
                security_cols.append(col)
            except:
                continue

    if len(security_cols) >= 2:
        try:
            # Ensure all security columns are numeric
            for col in security_cols:
                X_advanced[col] = pd.to_numeric(X_advanced[col], errors='coerce').fillna(0)

            # Create security strength score
            security_sum = X_advanced[security_cols].sum(axis=1)
            X_advanced['Security_Strength_Score'] = (security_sum / len(security_cols)).astype(np.float32)
            vuln_created += 1

            # Security gaps (areas with missing protection)
            X_advanced['Security_Gaps_Count'] = (X_advanced[security_cols] == 0).sum(axis=1).astype(np.uint8)
            vuln_created += 1
        except Exception as e:
            print(f"      Warning: Could not create security indicators: {e}")

    # OS and software update indicators - only use numeric version columns
    version_cols = [col for col in numeric_features if 'version' in col.lower()]
    if len(version_cols) > 0:
        version_col = version_cols[0]
        if version_col in X_advanced.columns:
            try:
                # Ensure column is numeric
                X_advanced[version_col] = pd.to_numeric(X_advanced[version_col], errors='coerce').fillna(0)

                # Create version freshness indicator (higher versions = more recent)
                version_percentile = X_advanced[version_col].rank(pct=True)
                X_advanced['Version_Freshness_Score'] = version_percentile.astype(np.float32)
                vuln_created += 1
            except Exception as e:
                print(f"      Warning: Could not create version freshness score: {e}")

    print(f"      Created {vuln_created} vulnerability indicators")
    features_created += vuln_created

    print(f"\n🎯 ADVANCED FEATURE ENGINEERING SUMMARY:")
    print(f"   📊 Missingness flags: {missing_flags_created}")
    print(f"   🔗 Cross-categorical: {cross_cat_created}")
    print(f"   📈 Advanced ratios: {ratios_created}")
    print(f"   🔢 Frequency features: {freq_created}")
    print(f"   🛡️ Vulnerability indicators: {vuln_created}")
    print(f"   🎯 TOTAL FEATURES ADDED: {features_created}")

    return X_advanced, cross_combinations

def enhanced_categorical_encoding_with_pruning(X, y, cv_folds, categorical_features, min_frequency=20):
    """
    Enhanced categorical encoding that combines frequency pruning with advanced encoding techniques
    """
    print(f"\n🎯 ENHANCED CATEGORICAL ENCODING WITH PRUNING")
    print("="*50)

    # Step 1: Apply frequency-based pruning first
    X_pruned, pruning_stats = frequency_based_pruning(X, categorical_features, min_frequency)

    # Step 2: Apply advanced categorical encoding
    X_encoded, encoding_mappings = advanced_categorical_encoding(X_pruned, y, cv_folds, categorical_features)

    # Combine statistics
    encoding_mappings['pruning_stats'] = pruning_stats

    return X_encoded, encoding_mappings

# Check if we have the required variables from Cell 1
required_vars = ['X_work', 'y_work', 'cv_folds', 'numeric_features', 'categorical_features']
missing_vars = [var for var in required_vars if var not in globals()]

if missing_vars:
    print(f"❌ Missing variables from previous cells: {missing_vars}")
    print("   Please run Cell 1 first!")
    raise RuntimeError("Required variables not found. Run Cell 1 first.")
else:
    print("✅ All required variables found from previous cells")

# ====================
# 🎯 MAIN KAGGLE PREPROCESSING PIPELINE
# ====================

# Initial memory check
monitor_kaggle_memory(X_work, "Initial dataset")

# Run leakage audit on the working data from Cell 1
print(f"🔍 Using working dataset: {X_work.shape[0]:,} rows, {X_work.shape[1]} columns")
leakage_safe, critical_leaks, warnings = audit_feature_leakage(X_work, y_work)

if not leakage_safe:
    print("🛑 CRITICAL LEAKAGE DETECTED - Pipeline stopped!")
    print("Fix these issues before continuing:")
    for feature, corr in critical_leaks:
        print(f"   Remove or investigate: {feature} (correlation: {corr:.4f})")
else:
    print("✅ Leakage audit passed - continuing with Kaggle-optimized preprocessing")

    # Optimize dtypes first for Kaggle
    X_work_optimized = safe_dtype_optimization_kaggle(X_work.copy())

    # Use the cv_folds from Cell 1 for consistency
    print(f"✅ Using CV folds from Cell 1: {len(cv_folds)} folds")

    # STEP: Apply advanced feature engineering BEFORE encoding
    X_work_advanced, cross_combinations = create_advanced_feature_engineering(
        X_work_optimized, categorical_features, numeric_features
    )

    # Update categorical features list to include cross-categorical features
    categorical_features_updated = categorical_features + cross_combinations

    # Advanced categorical encoding (now with frequency pruning)
    X_work_encoded, encoding_mappings = enhanced_categorical_encoding_with_pruning(
        X_work_advanced, y_work, cv_folds, categorical_features_updated, min_frequency=20
    )

    # Update feature lists after encoding
    numeric_features_updated = X_work_encoded.select_dtypes(include=[np.number]).columns.tolist()

    # Create interaction features (Kaggle-optimized)
    X_work_interactions = create_interaction_features_kaggle(
        X_work_encoded, numeric_features_updated[:8]  # Reduced for Kaggle
    )

    # Create binned features for regularization (Kaggle-optimized)
    X_work_final = create_binned_features_kaggle(X_work_interactions, numeric_features_updated[:10])

    # Final dtype optimization
    X_work_final = safe_dtype_optimization_kaggle(X_work_final)

    print(f"\n🎯 KAGGLE PREPROCESSING COMPLETE:")
    print(f"   Final shape: {X_work_final.shape}")
    print(f"   Features added: {X_work_final.shape[1] - X_work.shape[1]}")
    final_memory = X_work_final.memory_usage(deep=True).sum() / 1024**2
    print(f"   Final memory usage: {final_memory:.1f}MB")

    if IN_KAGGLE:
        print(f"   Kaggle memory utilization: {(final_memory/1024)/MAX_RAM_GB:.1%}")

    # ====================
    # 🎯 PREPARE FOR CELL 3
    # ====================

    # Update feature lists for next cell
    final_numeric_features = X_work_final.select_dtypes(include=[np.number]).columns.tolist()
    final_categorical_features = X_work_final.select_dtypes(include=['object', 'category']).columns.tolist()

    # Final memory cleanup for Kaggle
    if IN_KAGGLE:
        del X_work_optimized, X_work_encoded, X_work_interactions
        gc.collect()

    # CRITICAL FIX: Export variables to global scope for Cell 3
    globals()['X_work_final'] = X_work_final
    globals()['final_numeric_features'] = final_numeric_features
    globals()['final_categorical_features'] = final_categorical_features
    globals()['encoding_mappings'] = encoding_mappings

    print(f"\n🔧 EXPORTING VARIABLES TO GLOBAL SCOPE FOR CELL 3:")
    print(f"   ✅ X_work_final: {X_work_final.shape}")
    print(f"   ✅ final_numeric_features: {len(final_numeric_features)} features")
    print(f"   ✅ final_categorical_features: {len(final_categorical_features)} features")

    print(f"\n✅ Ready for Cell 3 with Kaggle-optimized features:")
    print(f"   📊 Variables prepared: X_work_final, y_work, cv_folds")
    print(f"   📋 Feature lists: final_numeric_features, final_categorical_features")
    print(f"   🔧 Encoding mappings saved for consistency")
    print(f"   🏆 Kaggle memory optimized: {final_memory:.1f}MB")
