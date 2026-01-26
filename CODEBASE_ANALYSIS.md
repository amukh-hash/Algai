# Codebase Analysis & Walkthrough: Algai Project

## 1. Executive Summary

This project ("Algai") is a sophisticated institutional-grade algorithmic trading platform designed to bridge the gap between advanced Machine Learning (ML) and practical financial engineering. It features a "Glass Box" AI approach, emphasizing explainability (XAI) alongside performance.

The system is architected as a hybrid application:
- **Backend**: Python-based (FastAPI) handling data ingestion, complex feature engineering (fractional differentiation, microstructure metrics), model training (PyTorch, Chronos, LoRA), and vectorized execution simulation.
- **Frontend**: React-based dashboard for visualization, strategy building, and monitoring (Greeks, Confidence Gauges, PnL).
- **Infrastructure**: A robust pipeline supporting "Central Hub" development, GPU-accelerated training, and an ensemble of models ("The Overlord", "The Judge").

## 2. Infrastructure Analysis

The infrastructure is designed for a high-performance, iterative research-to-production workflow.

*   **Environment**:
    *   **Windows/GPU Centric**: Scripts like `run_gpu.bat` and `run_training.bat` explicitly target Windows paths and CUDA environments (`venv_gpu`).
    *   **Dependency Management**: Uses `requirements.txt` with heavy hitters like `torch`, `pandas`, `polars` (for fast data processing), and `yfinance`/`alpaca` for data.
*   **Data Pipeline**:
    *   **Ingestion**: Scripts (`download_alpaca.py`, `unzip_data.py`) fetch raw data (bars, trade logs).
    *   **Storage**: Uses Parquet (`data_cache_alpaca/`) for efficient, columnar storage.
    *   **Processing**: `phase1_feature_engineering.py` uses Polars and multi-processing to compute complex features (OFI, Microprice, Fractional Diff) on tick-level data.
*   **Training Workflow**:
    *   **Multi-Stage Training**:
        1.  **Universal Pre-training**: `train_universal.py` trains a base model on all assets.
        2.  **Specialist Fine-tuning**: `train_specialist.py` adapts the base model to specific tickers.
        3.  **Ensemble ("The Overlord")**: `train_ensemble.py` generates diverse models via random search.
        4.  **Meta-Labeling ("The Judge")**: `train_metalabeler.py` and `train_stacking.py` train secondary models to filter signals.
    *   **Continuous Learning**: `continuous_learning.py` implements EWC (Elastic Weight Consolidation) to prevent catastrophic forgetting.
*   **Orchestration**:
    *   `orchestrate_overnight.py` automates long-running training tasks, monitoring for data availability.

## 3. Business Logic Analysis

The core business value lies in its advanced financial modeling and signal generation strategies.

*   **Financial Models**:
    *   **Triple Barrier Method**: Implemented in `backend/app/targets/triple_barrier.py`, this labels data based on profit-taking, stop-loss, and time-out horizons, strictly accounting for volatility.
    *   **Fractional Differentiation**: Preserves memory in time series while making them stationary (`backend/app/preprocessing/fractional.py`).
    *   **Greeks Calculation**: Black-Scholes implementation (`backend/app/math/greeks.py`) for options pricing and risk management.
*   **Trading Strategy**:
    *   **Hybrid Signal**: Combines Directional probability (Classification) and Volatility magnitude (Regression).
    *   **Vectorized Execution**: `VectorizedExecutioner` simulates trading logic (entry/exit, PnL tracking) efficiently across large datasets without loops.
    *   **Context Awareness**: Incorporates Market Breadth (AD Line) and Buying Pressure (BPI) to filter signals based on macro regimes.
*   **Machine Learning Architecture**:
    *   **PatchTST**: The core backbone (`backend/app/models/patchtst.py`) uses Transformer-based patching for time series forecasting.
    *   **Multimodal Fusion**: `backend/app/models/fusion.py` attempts to fuse text (sentiment) and price embeddings.
    *   **Chronos**: Integrates Amazon's Chronos (T5-based) foundation model for "Physics" based signals.
    *   **LoRA**: Low-Rank Adaptation (`backend/app/models/lora.py`) allows efficient fine-tuning of large models.

## 4. Code Quality Review

*   **Strengths**:
    *   **Modularity**: Clear separation of concerns (Engine vs. Models vs. Scripts).
    *   **Modern Tech Stack**: Usage of Fast and efficient libraries (Polars, FastAPI, PyTorch 2.0).
    *   **Sophistication**: Implementation of advanced concepts (Triple Barrier, EWC, LoRA) shows high domain expertise.
    *   **Async/Vectorized**: Good use of `asyncio` for I/O and vectorization for computation.
*   **Areas for Improvement**:
    *   **Error Handling**: Some scripts (e.g., `predict_ensemble.py`) have broad `try-except` blocks that might mask critical failures.
    *   **Hardcoded Paths**: Many scripts use absolute Windows paths (e.g., `C:\Users\Aishik\...`) or relative paths that assume a specific CWD. Using a config file or `pathlib` relative to `__file__` would be more robust.
    *   **Consistency**: Mixing of `pandas` and `polars` logic (e.g., `phase1` uses Polars, `train_global` uses Pandas). While Polars is faster, consistent interfaces help maintainability.
    *   **Documentation**: While code is generally readable, complex logic like `triple_barrier` could benefit from more inline comments explaining the "why".

---

## 5. Detailed File Walkthrough

### Root Directory

#### `WORKFLOW.md`
**Summary**: Defines the "Central Hub" development strategy, roles (Jules, Antigravity), and daily workflow for syncing code between local and remote environments.

#### `.julesrules`
**Summary**: Establishes coding standards (PEP 8), project structure, and tech stack preferences.

#### `check_data.py`
**Summary**: Utility script to validate integrity of Parquet files in the data cache. Checks for file size and readability.

#### `check_gpu.py`
**Summary**: Simple diagnostic script to verify PyTorch CUDA availability and device details.

#### `run_*.bat` (gpu, theta, training)
**Summary**: Windows batch scripts to launch environments. `run_gpu.bat` is the main menu for training tasks, forcing the use of `venv_gpu`.

### Backend App Core (`backend/app`)

#### `backend/app/main.py`
**Summary**: The FastAPI application entry point. Configures CORS and includes routers for Data, Hybrid, and XAI endpoints.

#### `backend/app/core/events.py`
**Summary**: Defines the Event-Driven architecture classes (`MarketEvent`, `SignalEvent`, `OrderEvent`, `FillEvent`) used for backtesting communication.

#### `backend/app/core/loss.py`
**Summary**: Custom Loss functions.
- `FocalLoss`: Handles class imbalance (e.g., rare market crashes).
- `UniversalLoss`: Combines Focal Loss (Direction) and MSE (Volatility).

#### `backend/app/core/trainer.py`
**Summary**: A standardized PyTorch training loop handling mixed-precision (AMP) training, validation, and early stopping.

#### `backend/app/core/dataset.py`
**Summary**: `TimeSeriesDataset` class for PyTorch. Handles sliding window creation from Pandas DataFrames.

### Backend App Models (`backend/app/models`)

#### `backend/app/models/fusion.py` (**Complex**)
**Summary**: Implements multimodal fusion of text and price data using a Gated Unit.
**Line-by-Line Analysis**:
- `TextEmbeddingMock`: Simulates a FinBERT model. In production, this would be a real Transformer.
- `GatedMultimodalUnit`:
    - **Init**: Defines projections (`W_t`, `W_p`) and a Gate mechanism (`W_z`).
    - **Forward**:
        - `h_text`, `h_price`: Tanh activations of projected inputs.
        - `z`: Sigmoid gate computed from concatenated inputs. Determines "how much text matters".
        - `h_fused`: Weighted sum `z * text + (1-z) * price`. This allows the model to dynamically ignore text if irrelevant.

#### `backend/app/models/patchtst.py` (**Complex**)
**Summary**: The core Hybrid PatchTST model for time series forecasting.
**Line-by-Line Analysis**:
- `PatchTSTBackbone`: Wraps standard `nn.TransformerEncoder`. Uses learnable positional embeddings.
- `HybridPatchTST`:
    - **Init**: Sets up RevIN (Normalization), PatchEmbedding, Backbone, and Heads.
    - **Heads**:
        - `direction_head`: Outputs logits for classification (Buy/Sell/Neutral).
        - `volatility_head`: Outputs scalar for volatility magnitude (Softplus for positivity).
    - **LoRA**: `_inject_lora` replaces specific Linear layers with `LoRALinear` for efficient fine-tuning.
    - **Forward**:
        1. Normalize input (RevIN).
        2. Patch embed (unfold time series).
        3. Encoder pass.
        4. Flatten and project to heads.

#### `backend/app/models/lora.py`
**Summary**: Implements Low-Rank Adaptation. `LoRALinear` wraps a frozen layer and adds a trainable low-rank side path (`Wx + BAx`).

### Backend App Math & Targets

#### `backend/app/targets/triple_barrier.py` (**Complex**)
**Summary**: Implements the Triple Barrier labeling method for supervised learning.
**Line-by-Line Analysis**:
- `get_daily_vol`: Computes dynamic volatility threshold.
- `apply_triple_barrier`:
    - Iterates through the price series.
    - For each point `i`, defines a window `[i : i + vertical_barrier]`.
    - Calculates returns relative to `price[i]`.
    - `hit_upper`/`hit_lower`: Finds indices where return exceeds `vol * multiplier`.
    - Logic:
        - If `upper` hit first -> Label 1 (Buy).
        - If `lower` hit first -> Label 2 (Sell).
        - If neither (timeout) -> Label 0 (Neutral).
- `get_purged_indices`: Prevents data leakage by removing samples that overlap with active trades of selected samples.

#### `backend/app/math/greeks.py`
**Summary**: Vectorized Black-Scholes calculator. Computes Delta, Gamma, Vega, Theta for options pricing.

### Backend App Engine

#### `backend/app/engine/vectorized.py` (**Complex**)
**Summary**: Simulates trade execution without event loops.
**Line-by-Line Analysis**:
- `update_positions`: Joins `open_positions` with `market_data` (Quotes) to update current PnL.
- `check_exits`: Vectorized boolean masks for Take Profit (`tp_mask`), Stop Loss (`sl_mask`), and Expiration. Moves exited trades to `closed_trades`.
- `select_contracts`:
    - Filters predictions by confidence.
    - Merges with Option Chain (Calls only).
    - Checks capital.
    - Appends new positions to `open_positions` dataframe.

#### `backend/app/engine/pipe.py`
**Summary**: `InferencePipe` orchestrates the Async workflow: Data Loading (Thread) -> Inference (GPU) -> Execution (CPU).

### Backend Scripts (`backend/scripts`)

#### `phase1_feature_engineering.py` (**Complex**)
**Summary**: Advanced feature engineering pipeline using Polars.
**Line-by-Line Analysis**:
- `process_daily_dbn`:
    - Loads Databento binary files.
    - Computes **Order Flow Imbalance (OFI)**: Delta of Bid/Ask sizes.
    - Computes **Microprice**: Volume-weighted mid-price.
    - Resamples to 1-second bars using Polars `group_by_dynamic`.
- `fast_fracdiff`: Implements Fixed-Window Fractional Differentiation to make series stationary without losing memory.
- `select_orthogonal_features`: Uses Hierarchical Clustering (Ward's method) on the correlation matrix to remove redundant features.

#### `train_universal.py`
**Summary**: Pre-trains the model on a massive dataset of all tickers to learn general market physics.

#### `train_ensemble.py`
**Summary**: "The Overlord". An infinite loop that spawns models with random configurations (Lookback, Heads, Features) to build a diverse ensemble.

#### `train_metalabeler.py`
**Summary**: Trains "The Judge" (XGBoost) to predict if the base model's signal is correct (`is_correct = Pred == True`).

#### `train_chronos_phase2.py`
**Summary**: Fine-tunes the Chronos T5 model using 4-bit Quantization (QLoRA) on the orthogonal dataset.

### Frontend & API

#### `backend/app/api/v1/endpoints/*.py`
**Summary**:
- `data.py`: Handling synthetic/yfinance data ingestion.
- `hybrid.py`: Triggering the inference pipeline.
- `xai.py`: Exposing Explainability (Integrated Gradients).

#### `frontend/src/components/Dashboard.js`
**Summary**: Main React component.
- Triggers `runPipeline` (axios post).
- Layouts "The Why" (Confidence), "The What" (Trade Details), and "The How" (PnL).
- State management for loading and selected trades.

#### `frontend/src/components/ConfidenceGauges.js`
**Summary**: Visualizes AI output using Recharts (Pie Chart for Direction, Text for Volatility).

---
**End of Analysis**
