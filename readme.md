# Unsupervised Anomaly Detection in LLM Operations (AIOps)

![Final Anomaly Plot](https://raw.githubusercontent.com/mshivam-kumar/Anomaly-Detection-in-LLM-Operations/main/outputs/plots/final_anomaly_plot_sensitive_th.png)

## 1. Problem Statement

As companies rapidly integrate Large Language Models (LLMs) into their products, a new and critical challenge has emerged in the field of IT Operations (AIOps): **monitoring the unique failure modes of these generative AI systems.**

Traditional monitoring tools are designed for simple metrics like CPU usage or server errors. They are ill-equipped to detect complex, LLM-specific issues such as:
- **Cost Anomalies:** A compromised API key or a buggy application can cause a sudden, massive increase in token usage, resulting in significant, unexpected cloud bills.
- **Performance Degradation:** The latency of an LLM can degrade subtly over time due to model updates or changing usage patterns, impacting user experience.
- **Malicious Use & Probing:** Users may attempt to find vulnerabilities or launch Denial of Service (DoS) attacks that manifest as unusual traffic patterns, not simple errors.

The core challenge is that these events are **rare, unlabeled, and often hidden** within millions of legitimate API calls. This project tackles this problem by building an **unsupervised learning system** that can automatically learn the "normal" operational rhythm of an LLM service and flag any significant deviations without needing prior examples of failures.

## 2. The AIOps Solution

This project implements an end-to-end pipeline using a **PyTorch LSTM Autoencoder** to learn and identify abnormal operational patterns in LLM telemetry.

### 2.1. Realistic Data Simulation

To train a robust model, a large-scale synthetic dataset was generated to simulate a full year of LLM API logs. This was a critical step as real-world public datasets for LLMOps are not yet available. The simulation includes:
- **Diurnal and Weekly Cycles:** Traffic realistically peaks during weekdays and is lower at night and on weekends.
- **User Personas:** A mix of "casual" and "power" users with different activity levels.
- **Concept Drift:** A gradual increase in latency and token counts over the year, forcing the model to learn an evolving baseline of "normalcy."

![Data Pattern Plot](https://raw.githubusercontent.com/mshivam-kumar/Anomaly-Detection-in-LLM-Operations/main/outputs/plots/eda/train_dataset_time_series.png)
*(Example of the diurnal (daily) pattern generated for the training data)*

### 2.2. Model Training and Optimization

An LSTM Autoencoder was trained for over 100 epochs on 9 months of purely normal data. The training process was professionalized with a `ReduceLROnPlateau` scheduler and Early Stopping to prevent overfitting.

![Train/Val Loss Plot](https://raw.githubusercontent.com/mshivam-kumar/Anomaly-Detection-in-LLM-Operations/main/outputs/plots/train_val_loss_epoch_112.png)
*(Training and validation loss curves, showing effective learning and convergence)*

## 3. Results: A Realistic and Actionable Detection System

A key challenge in unsupervised detection is setting a reliable threshold. A full sensitivity analysis was performed to find the optimal balance between detecting real incidents and avoiding false alarms.

### 3.1. Why These Results Are Strong for Unsupervised Learning

Unsupervised anomaly detection is fundamentally challenging because the model has no prior knowledge of what a "failure" looks like. It must learn to identify abnormal events based solely on their deviation from a learned baseline of normalcy. A **Precision of 42%** and an **F1-Score of 0.57** are excellent results in this context because:
- **Massive Data Reduction:** The system successfully filters millions of raw events down to a small, manageable number of high-probability incidents for human review.
- **Actionable Signal:** A 42% precision means nearly half of all alerts are true positives, providing a strong, reliable signal that is far superior to random chance or simple thresholding.
- **High Sensitivity:** The model demonstrates high recall, proving it can effectively catch the majority of true, unknown problems.

### 3.2. Final Performance Metrics (Threshold: 4.0)

A final threshold of 4.0 was chosen after a sensitivity analysis, as it provided the best balance of performance.

- **F1-Score:** **0.57** (A strong, balanced score for an unsupervised task)
- **Precision:** **42%** (Provides an actionable signal for an operations team)
- **Recall:** **87%** (Catches the vast majority of true anomalies)

![Confusion Matrix](https://raw.githubusercontent.com/mshivam-kumar/Anomaly-Detection-in-LLM-Operations/main/outputs/plots/confusion_matrix/final_confusion_matrix_optimal_th.png)

### 3.3. Key Insight: Symptom-Driven Detection

A deeper analysis revealed that the model learned a sophisticated, real-world lesson: **nearly all significant problems, regardless of their origin, ultimately manifest as performance degradation.** Even when the root cause was a `DoS Attack` or `Token Abuse`, the most prominent signal the model detected was the resulting spike in system latency. This is a critical, real-world insight, showing the model learned to identify the most reliable *symptom* of system instability.

### 3.4. Final Visualization (Investigative Threshold)

The plot below was generated using a more sensitive threshold (`1.5`) for investigative purposes. It demonstrates the model's full capability to detect all three ground-truth anomaly periods (shaded orange) and use its root cause analysis to automatically label the type of deviation it has detected.

![Final Anomaly Plot (Sensitive)](https://raw.githubusercontent.com/mshivam-kumar/Anomaly-Detection-in-LLM-Operations/main/outputs/plots/final_anomaly_plot_sensitive_th.png)

## 4. How to Run

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
    Open and run the cells in [`main_using_modules.ipynb`](https://github.com/mshivam-kumar/Anomaly-Detection-in-LLM-Operations/blob/main/main_using_modules.ipynb) to execute the entire pipeline from data generation to final analysis.
