# Codebase Walkthrough: ALGAI Architecture V2.0 (Teacher-Student Distillation)

This document explains the implementation of the V2.0 "Teacher-Student" trading architecture. The goal is to separate **Perception** (finding ground truth using heavy models and future data) from **Action** (live execution using lightweight models and causal filters).

## Overarching Intent

1.  **Teacher (Chronos T5)**: "The Sage". It runs overnight on high-quality L2 data. It uses *future information* (acausal smoothing) to determine what *should* have happened. It is too slow and heavy for live trading.
2.  **Student (Chronos Bolt)**: "The Agent". It runs live on L1 data. It is trained to mimic the Teacher's outputs while only seeing noisy live data. It learns to "hallucinate" L2 intelligence and smooth trends from L1 inputs.
3.  **Cycle**:
    *   **Daytime**: Student trades + adapts (SimTS).
    *   **Nighttime**: Teacher reviews the day's data, generates "Soft Labels" (logits), and trains the Student.

---

## 1. Core Modules

### `backend/app/features/signal_processing.py`
This file implements the advanced signal processing required to separate signal from noise.

*   **`apply_modwt_uks(data)`**: *Teacher Only*.
    *   **Goal**: Create the "Golden Label" trajectory.
    *   **Method**: Uses Maximal Overlap Discrete Wavelet Transform (MODWT) to decompose the signal, then applies an **Unscented Kalman Smoother (UKS)**.
    *   **Why**: UKS is "acausal" (uses future data to smooth past data), providing a mathematically superior view of the trend than any real-time filter could.
*   **`apply_sliding_wavelet_ukf(window)`**: *Student Only*.
    *   **Goal**: Estimate the current state in real-time without look-ahead bias.
    *   **Method**: Uses Sliding Window Wavelets + **Unscented Kalman Filter (UKF)**.
    *   **Why**: It must be strictly causal. It tries to approximate the Teacher's smooth line using only past data.
*   **`trend_scanning_labels(prices)`**: Generates t-statistic labels for regime detection.
*   **`triple_barrier_labels`**: Generates definitive Buy/Sell/Neutral labels based on volatility barriers.

### `backend/app/models/loss.py`
This file contains the custom loss function that aligns the Student with the Teacher and the Market.

*   **`StudentTradingLoss`**:
    *   **Task 1: Distillation (KL Divergence)**: Minimizes the difference between Student's probability distribution and Teacher's distribution. "Learn what the Teacher sees."
    *   **Task 2: Differentiable Sortino Ratio**: Directly optimizes the Risk-Adjusted Return (Sortino). "Learn to make money safely."
    *   **Task 3: Focal Loss**: Focuses learning on hard-to-predict market turns (Triple Barrier targets). "Focus on opportunities."
    *   **Homoscedastic Weighting**: Automatically balances these 3 tasks during training using learnable uncertainty weights.

### `backend/app/api/databento_client.py`
*   Handles fetching historical L2 data (MBP-10) for the Teacher and streaming L1 data for the Student.
*   Includes a `mock_mode` for testing without API keys.

---

## 2. Workflows & Scripts

### `backend/scripts/train_teacher_t5.py`
*   **Role**: Initial Setup / Periodic Re-training.
*   **Input**: Historical L2 Data.
*   **Process**:
    1.  Fetch raw price data.
    2.  Apply `apply_modwt_uks` to get the smoothed "Perception" signal.
    3.  Train `amazon/chronos-t5-large` to predict this smoothed signal.
*   **Output**: A saved Teacher model (`backend/models/teacher_t5_smoothed`).

### `backend/scripts/train_nightly_distill.py`
*   **Role**: The Nightly Knowledge Transfer.
*   **Input**: The previous day's trading data.
*   **Process**:
    1.  **Teacher Inference**: The Teacher looks at the data (simulating L2 visibility) and outputs probability distributions (Logits) for every timestep. These are "Soft Labels".
    2.  **Student Training**: The Student (`amazon/chronos-bolt-small`) takes the *same* data (simulating L1 noise).
    3.  **Optimization**: The Student updates its weights to minimize `StudentTradingLoss` (matching Teacher's logits + maximizing Sortino Ratio).
*   **Output**: An updated Student model (`backend/models/chronos_bolt_distilled`).

### `backend/scripts/run_live_bolt.py`
*   **Role**: The Live Execution Loop.
*   **Input**: Live Tick Stream.
*   **Process**:
    1.  **Tick Arrival**: Receives a new price tick.
    2.  **Causal Filter**: Updates the `apply_sliding_wavelet_ukf` state to filter noise.
    3.  **Inference**: The Student model predicts the next step based on the filtered window.
    4.  **Meta-Labeling ("The Judge")**: A placeholder XGBoost model evaluates if the Student's confidence is trustworthy.
    5.  **Portfolio**: Riskfolio-lib calculates allocation size (Kelly Criterion).
    6.  **Daytime SimTS**: Every N ticks, the script runs a self-supervised update to adapt the Student's encoder to the current day's volatility regime.
*   **Output**: Buy/Sell/Wait signals.

---

## 3. Summary of Data Flow

```mermaid
graph TD
    Market[Market Data] -->|Overnight Batch (L2)| Teacher[Chronos T5]
    Market -->|Live Stream (L1)| Student[Chronos Bolt]

    Teacher -->|Smoothed Logits| Loss[StudentTradingLoss]
    Student -->|Noisy Preds| Loss

    Loss -->|Gradients| Student

    subgraph Live Execution
    LiveTick --> Filter[Wavelet + UKF]
    Filter --> StudentInfer[Student Inference]
    StudentInfer --> Judge[Meta-Labeler]
    Judge --> Portfolio[Allocation]
    Portfolio --> Order[Execute]
    end
```
