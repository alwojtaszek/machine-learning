from matplotlib import pyplot as plt


def plot_mismatched_labels(x_test, y_pred_label, y_pred_probability, y_test, n=5):
    """
      Plot Mismatched Labels after CNN Classification

      This method creates a plot to visualize instances where true labels and predicted labels do not match
      after CNN classification.

      Parameters:
      - x_test (numpy.ndarray): test dataset.
      - y_pred (numpy.ndarray): List of predicted labels from the CNN model.
      - y_test (list): List of true labels for the dataset.

      Returns:
      - None"""
    mismatched_labels = [(x, pred_label, pred_prob, true_label) for true_label, pred_label, pred_prob, x
                         in zip(y_test, y_pred_label, y_pred_probability, x_test)
                         if pred_label != true_label][:n]
    nr_of_items = len(mismatched_labels)
    fig, axs = plt.subplots(nr_of_items, sharex=True)
    for i, (x, pred_label, pred_prob, true_label) in enumerate(mismatched_labels):
        axs[i].imshow(x)
        axs[i].set_xticks([])
        axs[i].set_yticks([])
        axs[i].set_xlabel('{}'.format('Prediction: {}, Actual label: {}, Probability: {}'.
                                      format(pred_label, true_label, pred_prob)))