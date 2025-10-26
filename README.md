Project: MiniHackthon - Microsoft Malware Prediction (Enhanced)

Overview
--------
This repository contains a multi-cell pipeline (cell1..cell5) built to train and tune classification models on a large dataset (~1.96M rows, 103 features). The code focuses on memory-aware preprocessing, stratified CV alignment with Cell 4, and kernel-safe hyperparameter tuning using LightGBM / XGBoost / CatBoost and Optuna.

Key goals
- Maintain training stability on limited memory (16 GB RAM)
- Preserve evaluation consistency with Cell 4 exact CV splits
- Keep AUC near baseline (~0.71) while improving recall/accuracy to ~0.70-0.75
- Avoid kernel crashes by sampling and memory optimizations during tuning

Repository structure
- Working/
  - cell1.py        # pipeline config + data load
  - cell2.py        # feature engineering (Kaggle-optimized)
  - cell3.py        # preprocessing + selection
  - cell4_gpu_optimized.py  # large-dataset CV splits & baseline models
  - (cell5)         # advanced CV tuning cell (not committed here)

Important variables produced by Cell 4 (available for Cell 5)
- trained_models: 3 models
- model_results: 3 result sets
- baseline_config: 3 baseline configs
- optimized_config: 2 optimized configs
- oof_predictions_dict: 3 out-of-fold prediction sets
- training_metadata: training summary
- X_for_advanced_cv: shape (1960000, 103)
- y_for_advanced_cv: 1960000 samples
- cv_folds_baseline: 5 folds
- Best baseline AUC: 0.7103 (example: LightGBM)

Memory & tuning strategy (what the code implements)
- Data memory optimization: dtype downcasting and memory-aware sampling
- Tuning uses a sampled subset to avoid kernel OOMs. Sampling preserves class distribution.
- Recommended safe tuning configs used in code:
  - Tuning sample: 200,000 rows (default in pipeline) for stable AUC retention
  - CV folds for tuning: 3 (reduced for speed)
  - Multi-stage Optuna tuning (example): Stage1 = 30 trials, Stage2 = 15 trials
  - Kernel-safe alternate: Sample 50k, Trials = 5 per model (fast but may hurt AUC)

Recommendations to avoid kernel crashes while preserving AUC
- Use a larger sample for tuning (100k-200k) instead of 50k if memory permits. 200k was used in the pipeline and kept AUC close to baseline.
- Reduce number of parallel worker threads for tree libraries (set n_jobs=1) to reduce peak memory.
- Reduce CV folds during tuning (3 folds) and use more folds only for final evaluation on full data.
- Use staged tuning: fewer trials for quick search, then refined smaller search on promising ranges.
- Use gc.collect() frequently and delete large temp variables after use.

Troubleshooting (errors seen & quick fixes)
- NameError: cv_strategy not defined
  - Ensure cv_strategy is set (e.g. cv_strategy = pipeline_config.get('CV_STRATEGY', 'cell4_exact')) before use.

- NameError: check_memory_status not defined
  - Implement a small helper that returns True/False based on available RAM (psutil.virtual_memory() or pipeline_config flag).

- NameError: OPTUNA_AVAILABLE not defined
  - Define OPTUNA_AVAILABLE = True if Optuna can be imported, else False. Example:
      try:
          import optuna
n         OPTUNA_AVAILABLE = True
      except Exception:
          OPTUNA_AVAILABLE = False

- Kernel crashes (OOM) during tuning
  - Lower sample size, reduce trials, reduce CV folds, set n_jobs=1, offload to disk where possible.
  - If still failing, use smaller models (fewer leaves / depth) during tuning.

How to run safely on 16GB RAM
1. Ensure pipeline_config has MAX_RAM_GB set to 13 or lower.
2. Use tuning_sample_size = 100000 or 200000 (balanced between memory and representativeness).
3. Use tuning_cv_folds = 3, trials per stage = 20/10 or 30/15 depending on time budget.
4. Set library workers to 1 (n_jobs=1 / thread_count=1) during tuning.
5. Run tuning in a separate process if possible to free notebook memory after each run.

Final notes
- The pipeline intentionally aligns with Cell 4 exact CV splits to keep final evaluation consistent.
- If you reduce tuning sample size or trials to save memory, expect some drop in AUC; prefer increasing sample (not folds/trials) when feasible.

Contact
- Internal project: MiniHackthon (NEVIN)

(Generated summary README based on current code and console logs.)

