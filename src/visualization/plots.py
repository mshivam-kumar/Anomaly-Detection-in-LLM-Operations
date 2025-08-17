import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import os
# --- Set a professional plotting style ---
sns.set_theme(style="whitegrid")

def plot_train_val_loss(history, show_plot=False, save_path=None):
    """
    Plots the training and validation loss curves over epochs.

    Args:
        history (dict): A dictionary containing 'train_loss' and 'val_loss' lists.
        save_path (str, optional): Path to save the figure. If None, shows the plot.
    """
    plt.figure(figsize=(12, 7))
    plt.plot(history['train_loss'], label='Training Loss', color='blue', linewidth=2)
    plt.plot(history['val_loss'], label='Validation Loss', color='orange', linewidth=2)
    plt.title('Model Loss Over Epochs', fontsize=16)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss (MSE)', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, which="both")
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved training loss plot to: {save_path}")
    if show_plot == True:
        plt.show()
    plt.close()

# def plot_reconstruction_error_histogram(train_errors, test_errors, threshold, show_plot=False, save_path=None):
#     """
#     Plots the distribution of reconstruction errors for train and test sets.

#     Args:
#         train_errors (np.array): Reconstruction errors from the training (normal) set.
#         test_errors (np.array): Reconstruction errors from the test set (contains anomalies).
#         threshold (float): The anomaly detection threshold to display.
#         save_path (str, optional): Path to save the figure.
#     """
#     plt.figure(figsize=(12, 7))
#     sns.histplot(train_errors, bins=50, kde=True, color='blue', label='Normal Data (Train Set)')
#     sns.histplot(test_errors, bins=50, kde=True, color='red', alpha=0.6, label='Test Set (with Anomalies)')
#     plt.axvline(threshold, color='purple', linestyle='--', linewidth=2.5, label=f'Threshold ({threshold:.2f})')
    
#     plt.title('Distribution of Reconstruction Errors', fontsize=16)
#     plt.xlabel('Reconstruction Error (MSE)', fontsize=12)
#     plt.ylabel('Frequency', fontsize=12)
#     plt.yscale('log')
#     plt.legend(fontsize=12)
    
#     if save_path:
#         plt.savefig(save_path, dpi=300, bbox_inches='tight')
#         print(f"Saved error histogram plot to: {save_path}")
#     if show_plot == True:
#         plt.show()
#     plt.close() # it is important to close the plot to free up memory. Otherwise after function calls it plots all the plots for all the function calls(in loop) in the notebook.

def plot_reconstruction_error_histograms_separate(train_errors, test_errors, threshold, show_plot=False, base_save_path=None):
    """
    Plots and saves two SEPARATE visualizations of reconstruction errors.
    - Image 1: A zoomed-in "Focus" view of the normal error distribution.
    - Image 2: A "Context" view showing the full scale of anomalies.

    Args:
        train_errors (np.array): Reconstruction errors from the training (normal) set.
        test_errors (np.array): Reconstruction errors from the test set (contains anomalies).
        threshold (float): The anomaly detection threshold to display.
        show_plot (bool): Whether to display the plots.
        base_save_path (str, optional): Base path for saving figures. Suffixes will be added.
    """
    
    # --- Plot 1: The "Focus" View (Zoomed In) ---
    plt.figure(figsize=(15, 8)) # Create the first figure
    
    sns.histplot(train_errors, bins=50, kde=True, color='blue', label='Normal Data (Train Set)')
    sns.histplot(test_errors, bins=500, kde=True, color='red', alpha=0.6, label='Test Set (with Anomalies)')
    plt.axvline(threshold, color='purple', linestyle='--', linewidth=2.5, label=f'Threshold ({threshold:.2f})')
    
    plt.title('Focus View: Distribution of Normal-Range Errors', fontsize=16)
    plt.xlabel('Reconstruction Error (MSE)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.legend(fontsize=12)
    
    # Set the x-axis limit to zoom in on the normal range
    zoom_limit = np.max(train_errors) * 1.5 
    plt.xlim(0, zoom_limit)
    plt.grid(True, which="both", linestyle='--')
    
    if base_save_path:
        # Create a specific path for this plot
        save_path_focus = base_save_path.replace('.png', '_focus_on_train_reconstruction_errors.png')
        plt.savefig(save_path_focus, dpi=300, bbox_inches='tight')
        print(f"Saved FOCUS plot to: {save_path_focus}")
    if show_plot:
        plt.show()
    plt.close() # Close the first figure to free memory

    # --- Plot 2: The "Context" View (Full Scale) ---
    plt.figure(figsize=(15, 8)) # Create the second, independent figure
    
    sns.histplot(train_errors, bins=50, kde=True, color='blue', label='Normal Data (Train Set)')
    sns.histplot(test_errors, bins=500, kde=True, color='red', alpha=0.6, label='Test Set (with Anomalies)')
    plt.axvline(threshold, color='purple', linestyle='--', linewidth=2.5, label=f'Threshold ({threshold:.2f})')

    plt.title('Context View: Full Distribution with Anomalies', fontsize=16)
    plt.xlabel('Reconstruction Error (MSE)', fontsize=12)
    plt.ylabel('Frequency (Log Scale)', fontsize=12)
    plt.yscale('log') # Use log scale to see the tail
    plt.legend(fontsize=12)
    plt.grid(True, which="both", linestyle='--')

    if base_save_path:
        # Create a specific path for this plot
        save_path_context = base_save_path.replace('.png', '_context.png')
        plt.savefig(save_path_context, dpi=300, bbox_inches='tight')
        print(f"Saved CONTEXT plot to: {save_path_context}")
    if show_plot:
        plt.show()
    plt.close() # Close the second figure

def plot_precision_recall_vs_threshold(y_true, test_errors, show_plot=False, save_path=None):
    """
    Calculates and plots Precision, Recall, and F1-Score for a range of thresholds.

    Args:
        y_true (np.array): The ground truth labels (0 for normal, 1 for anomaly).
        test_errors (np.array): Reconstruction errors for the test set.
        save_path (str, optional): Path to save the figure.

    Returns:
        float: The threshold that maximizes the F1-score.
    """
    thresholds = np.logspace(np.log10(min(test_errors)), np.log10(max(test_errors)), num=200)
    scores = []

    for t in thresholds:
        y_pred = (test_errors > t).astype(int)
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
        scores.append([t, precision, recall, f1])
        
    scores_df = pd.DataFrame(scores, columns=['threshold', 'precision', 'recall', 'f1_score'])
    optimal_idx = scores_df['f1_score'].idxmax()
    optimal_threshold = scores_df.loc[optimal_idx, 'threshold']
    best_f1 = scores_df.loc[optimal_idx, 'f1_score']

    plt.figure(figsize=(12, 7))
    plt.plot(scores_df['threshold'], scores_df['precision'], label='Precision', color='blue')
    plt.plot(scores_df['threshold'], scores_df['recall'], label='Recall', color='green')
    plt.plot(scores_df['threshold'], scores_df['f1_score'], label='F1-Score', color='red', linewidth=2.5)
    plt.axvline(x=optimal_threshold, color='purple', linestyle='--', label=f'Optimal Threshold ({optimal_threshold:.2f})')
    
    plt.title('Precision, Recall, and F1-Score vs. Threshold', fontsize=16)
    plt.xlabel('Anomaly Threshold', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.xscale('log')
    plt.legend(fontsize=12)
    plt.grid(True, which="both")

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved precision-recall curve plot to: {save_path}")
    if show_plot == True:
        plt.show()
    plt.close()
    
    return optimal_threshold

def plot_confusion_matrix(y_true, y_pred, show_plot=False, save_path=None):
    """
    Plots a confusion matrix for the anomaly detection results.

    Args:
        y_true (np.array): The ground truth labels.
        y_pred (np.array): The predicted labels after applying a threshold.
        save_path (str, optional): Path to save the figure.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Predicted Normal', 'Predicted Anomaly'],
                yticklabels=['Actual Normal', 'Actual Anomaly'])
    plt.title('Confusion Matrix', fontsize=16)
    plt.ylabel('Actual Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved confusion matrix plot to: {save_path}")
    if show_plot == True:
        plt.show()
    plt.close()

def plot_final_anomalies(results_df, anomalies_detected, threshold, anomaly_labels_df, show_plot=False, save_path=None):
    """
    Creates the final, comprehensive plot showing anomalies and their inferred types over time.

    This function is designed to be the main visual output of the anomaly detection project,
    showing the reconstruction error, the detection threshold, the ground truth, and the
    model's inferred root cause for each detected anomaly.

    Args:
        results_df (pd.DataFrame): DataFrame with the full timeline of reconstruction errors.
                                   Must have a datetime index and a 'reconstruction_error' column.
        anomalies_detected (pd.DataFrame): DataFrame containing only the detected anomalies.
                                           Must have a datetime index, and columns for 
                                           'reconstruction_error' and 'inferred_type'.
        threshold (float): The anomaly detection threshold value used. This will be plotted as a line.
        anomaly_labels_df (pd.DataFrame): DataFrame with the ground truth anomaly windows.
                                          Must have 'start' and 'end' datetime columns.
        show_plot (bool, optional): If True, displays the plot. Defaults to False.
        save_path (str, optional): The full path to save the figure image file. 
                                   If None, the plot is not saved. Defaults to None.
    """
    fig, ax = plt.subplots(figsize=(22, 10))
    
    # --- Plot the main time series and threshold ---
    
    # Plot the continuous reconstruction error
    ax.plot(results_df.index, results_df['reconstruction_error'], color='blue', alpha=0.7, label='Reconstruction Error', linewidth=1.5)
    
    # Plot the threshold line. We create a temporary series for this.
    threshold_series = pd.Series(threshold, index=results_df.index)
    ax.plot(threshold_series.index, threshold_series, color='red', linestyle='--', label=f'Anomaly Threshold ({threshold:.2f})', linewidth=2.5)

    # --- Plot the detected anomalies using inferred types for color and style ---
    if not anomalies_detected.empty:
        sns.scatterplot(
            data=anomalies_detected, 
            x=anomalies_detected.index, 
            y='reconstruction_error', 
            hue='inferred_type',    # Different colors for different types
            style='inferred_type',  # Different markers for different types
            s=150,                  # Make markers large and visible
            ax=ax, 
            zorder=5                # Ensure markers are plotted on top of other elements
        )

    # --- Shade the ground truth anomaly windows for comparison ---
    for idx, row in anomaly_labels_df.iterrows():
        ax.axvspan(
            row['start'], 
            row['end'], 
            color='orange', 
            alpha=0.3, 
            # Only add one label for the ground truth to avoid a cluttered legend
            label="Ground Truth Anomaly" if idx == 0 else "_nolegend_"
        )

    # --- Final plot styling ---
    ax.set_yscale('log')
    ax.legend(fontsize=12)
    ax.set_title('Final Anomaly Detection with Inferred Root Cause (Log Scale)', fontsize=16)
    ax.set_ylabel('Reconstruction Error (MSE) - Log Scale', fontsize=12)
    ax.set_xlabel('Timestamp', fontsize=12)
    plt.grid(True, which="both", linestyle='--', linewidth=0.5)
    plt.tight_layout()

    # --- Save and/or show the plot ---
    if save_path:
        # Ensure the directory for the save path exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved final anomaly plot to: {save_path}")
        
    if show_plot:
        plt.show()
        
    # Close the plot to free up memory, which is important in loops or notebooks
    plt.close()