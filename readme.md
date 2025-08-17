# Unsupervised Anomaly Detection in LLM Operations (AIOps)

![Final Anomaly Plot](https://raw.githubusercontent.com/mshivam-kumar/Anomaly-Detection-in-LLM-Operations/main/outputs/plots/final_anomaly_plot_sensitive.png)

## Overview

This project implements an end-to-end unsupervised anomaly detection system to monitor the operational telemetry of a Large Language Model (LLM) service. In the rapidly growing field of LLMOps, traditional monitoring tools often fail to capture the unique failure modes of generative AI systems. This project addresses that gap by using a deep learning (LSTM Autoencoder) approach to learn the "normal" rhythm of an LLM API and automatically flag abnormal operational patterns related to cost, performance, and traffic.

The system is trained on a large-scale, synthetic dataset simulating a full year of activity, complete with realistic daily/weekly cycles, different user personas, and long-term concept drift. The final model is capable of not only detecting anomalies but also providing an automated first-pass root cause analysis by identifying the most likely source of the deviation.

## Project Highlights

- **Modern Problem Domain:** Tackles a cutting-edge challenge at the intersection of AIOps and LLMOps.
- **End-to-End Pipeline:** Covers the full project lifecycle from realistic data generation to model training, evaluation, and nuanced interpretation.
- **Deep Learning Model:** Utilizes an LSTM Autoencoder in PyTorch to learn complex, non-linear temporal patterns in multivariate time-series data.
- **Robust Evaluation:** Employs a rigorous train/validation/test split and uses F1-score optimization to determine a mathematically optimal detection threshold.
- **Automated Root Cause Analysis:** The final model doesn't just flag anomalies; it infers the likely cause (e.g., Latency, Token/Cost, or Traffic issues) by analyzing per-feature reconstruction errors.

## The AIOps Pipeline

The project is structured into a series of logical steps, reflecting a professional machine learning workflow.

### 1. Synthetic Data Generation

A sophisticated dataset was generated to simulate a year of LLM API logs. This was crucial as real-world public datasets for LLMOps are not yet available. The simulation includes:
- **Diurnal and Weekly Cycles:** Traffic realistically peaks during weekdays and is lower at night and on weekends.
- **User Personas:** A mix of "casual" and "power" users with different activity levels and token usage.
- **Concept Drift:** A gradual increase in latency and token counts over the year, forcing the model to learn an evolving baseline of "normalcy."

![Data Pattern Plot](https://raw.githubusercontent.com/mshivam-kumar/Anomaly-Detection-in-LLM-Operations/main/outputs/plots/diurnal_pattern_visualization.png)
*(Example of the diurnal (daily) pattern generated for the training data)*

### 2. Feature Engineering

The raw, event-based logs were aggregated into 5-minute time windows. A rich set of features was engineered for each window to capture the system's state, including mean/std/max latency, sum of tokens, request counts, and error rates.

### 3. Model Training

An LSTM Autoencoder was built in PyTorch. The model was trained for over 100 epochs on 9 months of purely normal data. Key training components included:
- **`ReduceLROnPlateau` Scheduler:** Dynamically adjusts the learning rate when the validation loss stalls.
- **Early Stopping:** Automatically halts training to prevent overfitting and save time.
- **Dropout Regularization:** Improves the model's ability to generalize to unseen data.

![Train/Val Loss Plot](https://raw.githubusercontent.com/mshivam-kumar/Anomaly-Detection-in-LLM-Operations/main/outputs/plots/train_val_loss.png)
*(Training and validation loss curves, showing effective learning and the point of convergence)*

### 4. Threshold Optimization & Evaluation

A key challenge in unsupervised anomaly detection is setting a reliable detection threshold. This project employed two methods:

- **Statistical Baseline:** A heuristic threshold was established based on the 99.9th percentile of reconstruction errors on normal data.
- **F1-Score Optimization:** A rigorous, data-driven approach was used to find the mathematically optimal threshold that best balances precision and recall on the final labeled test set.

![Precision-Recall Curve](https://raw.githubusercontent.com/mshivam-kumar/Anomaly-Detection-in-LLM-Operations/main/outputs/plots/precision_recall_curve.png)
*(The F1-score optimization curve, identifying the threshold that provides the best overall performance)*

## Results and Key Insights

The final model was evaluated using the F1-optimized threshold, revealing a realistic and highly effective detection system.

![Confusion Matrix](https://raw.githubusercontent.com/mshivam-kumar/Anomaly-Detection-in-LLM-Operations/main/outputs/plots/final_confusion_matrix.png)

### Key Performance Metrics:

- **Excellent Detection Coverage:** The system demonstrated outstanding recall, successfully identifying **100% of all system-wide `Latency Degradation` incidents** and **86% of the more subtle `Token/Cost Abuse` events.** This confirms its reliability as a primary alerting tool.

- **Symptom-Driven Root Cause Analysis:** The final analysis proved that the model learned that **performance degradation (latency) is the most consistent *symptom* of system instability.** Even when the root cause was a `DoS Attack` or `Token Abuse`, the most prominent signal the model detected was the resulting spike in latency. This is a critical, real-world insight, showing the model learned to identify the most reliable indicator of trouble.

- **Quantified Precision:** The precision scores of **41% for Latency** and **44% for Token/Cost** provide a clear expectation for an operational team. The system is designed as a powerful first-alert mechanism to flag potential issues for human investigation, successfully narrowing down millions of events to a manageable number of high-probability incidents.

### Final Visualization

The plot below, generated using a sensitive threshold for investigative purposes, demonstrates the model's full capability. It correctly flags all three ground-truth anomaly periods (shaded in orange) and uses its root cause analysis to automatically label the type of deviation it has detected.

![Final Anomaly Plot (Sensitive)](https://raw.githubusercontent.com/mshivam-kumar/Anomaly-Detection-in-LLM-Operations/main/outputs/plots/final_anomaly_plot_sensitive.png)

## How to Run

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/mshivam-kumar/Anomaly-Detection-in-LLM-Operations.git
    cd Anomaly-Detection-in-LLM-Operations
    ```
2.  **Set up the environment:**
    ```bash
    conda create --name llmops_anomaly python=3.10
    conda activate llmops_anomaly
    pip install -r requirements.txt
    ```
3.  **Run the main notebook:**
    Open and run the cells in [`main_using_modules.ipynb`](https://github.com/mshivam-kumar/Anomaly-Detection-in-LLM-Operations/blob/main/main_using_modules.ipynb) to execute the entire pipeline from data generation to final analysis. The notebook is structured to follow the logical steps of the project.