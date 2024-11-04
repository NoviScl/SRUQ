import matplotlib.pyplot as plt
import numpy as np
import json

def reliability_plot_three_sets(confidences_logprob, accuracies_logprob, 
                                confidences_ensemble, accuracies_ensemble, 
                                confidences_sruq, accuracies_sruq, num_bins=10, filename="reliability_plot.png"):
    """
    Creates a reliability plot with three calibration lines to measure calibration.
    
    Parameters:
    - confidences_logprob, accuracies_logprob: confidences and accuracies for the logprob method
    - confidences_ensemble, accuracies_ensemble: confidences and accuracies for the ensemble method
    - confidences_sruq, accuracies_sruq: confidences and accuracies for the sruq method
    - num_bins: number of equal-width bins (default is 10 for 0.1 intervals)
    - filename: file name to save the plot (default is "reliability_plot.png")
    """
    
    def bin_data_and_get_averages(confidences, accuracies, num_bins):
        """Helper function to bin data and calculate average confidence and accuracy."""
        # Convert lists to numpy arrays if they're not already
        confidences = np.array(confidences)
        accuracies = np.array(accuracies)
        
        # Define bin edges slightly beyond 0.0 and 1.0 to include boundary values
        bin_edges = np.linspace(-1e-6, 1 + 1e-6, num_bins + 1)
        bin_indices = np.digitize(confidences, bin_edges) - 1  # zero-indexed

        bin_confidences = []
        bin_accuracies = []

        for i in range(num_bins):
            # Create a boolean mask for the current bin
            bin_mask = (bin_indices == i)
            bin_conf = confidences[bin_mask]
            bin_acc = accuracies[bin_mask]

            # Calculate the mean confidence and accuracy for the bin if it has at least 3 data points
            if len(bin_conf) > 5:
                bin_confidences.append(np.mean(bin_conf))
                bin_accuracies.append(np.mean(bin_acc))

        return bin_confidences, bin_accuracies
    
    # Get bin averages for each method
    logprob_confidences, logprob_accuracies = bin_data_and_get_averages(confidences_logprob, accuracies_logprob, num_bins)
    ensemble_confidences, ensemble_accuracies = bin_data_and_get_averages(confidences_ensemble, accuracies_ensemble, num_bins)
    sruq_confidences, sruq_accuracies = bin_data_and_get_averages(confidences_sruq, accuracies_sruq, num_bins)
    
    # Plot reliability diagram
    plt.figure(figsize=(8, 8))
    plt.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
    
    # Plot each method with different colors and symbols for points, and labels
    plt.scatter(logprob_confidences, logprob_accuracies, color='blue', marker='o', label='Logprob')
    for i in range(len(logprob_confidences) - 1):
        plt.plot([logprob_confidences[i], logprob_confidences[i + 1]],
                 [logprob_accuracies[i], logprob_accuracies[i + 1]], color='blue')
    
    plt.scatter(ensemble_confidences, ensemble_accuracies, color='green', marker='s', label='Ensemble')
    for i in range(len(ensemble_confidences) - 1):
        plt.plot([ensemble_confidences[i], ensemble_confidences[i + 1]],
                 [ensemble_accuracies[i], ensemble_accuracies[i + 1]], color='green')
    
    plt.scatter(sruq_confidences, sruq_accuracies, color='red', marker='^', label='SRUQ')
    for i in range(len(sruq_confidences) - 1):
        plt.plot([sruq_confidences[i], sruq_confidences[i + 1]],
                 [sruq_accuracies[i], sruq_accuracies[i + 1]], color='red')
    
    # Add labels, title, and legend with larger font sizes
    plt.xlabel("Confidence", fontsize=14)
    plt.ylabel("Accuracy", fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.title("Reliability Diagram with Three Methods", fontsize=16)
    plt.legend(fontsize=16)
    plt.grid(True)
    
    # Save the plot
    plt.savefig(filename)
    plt.show()


with open("predictions/stratqa_logprob_CoT.json", "r") as f:
    data = json.load(f)
    confidences_logprob = data["confidences"]
    accuracies_logprob = data["accuracies"]

with open("predictions/stratqa_ensemble_CoT.json", "r") as f:
    data = json.load(f)
    confidences_ensemble = data["confidences"]
    accuracies_ensemble = data["accuracies"]

with open("predictions/stratqa_sruq_CoT_word_pagerank.json", "r") as f:
    data = json.load(f)
    confidences_sruq = data["confidences"]
    accuracies_sruq = data["accuracies"]


reliability_plot_three_sets(confidences_logprob, accuracies_logprob, 
                            confidences_ensemble, accuracies_ensemble, 
                            confidences_sruq, accuracies_sruq, filename="reliability_plot_stratqa.pdf")