import matplotlib.pyplot as plt
from PIL import Image
import numpy as np


# Redefine the function to reduce the distance between the bars within each subplot
def generate_plot(data):
    """
    Generate a 6x6 grid of bar plots based on the provided data table.

    Parameters:
    - data: A 6x6 array-like structure where each element is a list or tuple of two values [goal, lose].
    """
    # Validate the input dimensions
    assert len(data) == 6, "Input data must be a 6x6 array."
    for row in data:
        assert len(row) == 6, "Each row in data must contain 6 elements."
        for value in row:
            assert isinstance(value, (list, tuple)) and len(
                value) == 2, "Each cell must be a list or tuple with 2 values."

    # Determine the global maximum value for y-axis scaling
    max_value = max(max(value) for row in data for value in row)

    # Create a 6x6 figure
    fig, axes = plt.subplots(6, 6, figsize=(12, 12))

    # Loop through each subplot position to create a bar plot
    for i in range(6):
        for j in range(6):
            # Extract 'goal' and 'lose' values from the data
            goal_value, lose_value = data[i][j]
            bar_width = 0.3  # Reduce the bar width to make them closer
            axes[i, j].bar([0], [goal_value], color='lightblue', edgecolor='black',
                           width=bar_width)  # Plot goal bar at x=0
            axes[i, j].bar([0.4], [lose_value], color='lightcoral', edgecolor='black',
                           width=bar_width)  # Plot lose bar at x=0.4
            axes[i, j].set_ylim(0, max_value)  # Set y-axis range using the determined maximum value
            axes[i, j].set_xticks([])  # Remove x-axis tick labels

            label_list = ['sl_cl_line', 'sl_cl_r', 'sl_origin', 'cl_line', 'cl_r', 'origin']
            # Add x-axis label only for the bottom row
            if i == 5:
                axes[i, j].set_xlabel(label_list[j], fontsize=15)

            # Add y-axis label only for the leftmost column
            if j == 0:
                axes[i, j].set_ylabel(label_list[i], fontsize=15)

    # Adjust the layout to ensure everything fits well
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
    plt.show()


def fault_plot(data):
    """
    Generate a bar plot with six bars based on the provided data list.

    Parameters:
    - data: A list containing six values representing the heights of the bars.
    """
    # Validate the input dimensions
    assert len(data) == 6, "Input data must be a list with six values."

    # Define bar labels
    bar_labels = ['sl_cl_line', 'sl_cl_r', 'sl_origin', 'cl_line', 'cl_r', 'origin']

    # Create the bar plot
    plt.figure(figsize=(18, 6))
    plt.bar(bar_labels, data, color='indianred', edgecolor='black')

    plt.xticks(fontsize=22)
    plt.yticks(fontsize=20)
    # Label the axes
    plt.ylabel('Faults rate', fontsize=25)

    plt.ylim(0, 1)

    min_value = min(data)
    plt.axhline(y=min_value, color='gray', linestyle='--', linewidth=1, label=f'Minimum Rate = {min_value:.2f}')

    plt.legend(fontsize=22)
    # Display the plot
    plt.show()


# Example data for demonstration
example_data = [77.2/511.7, 78.6/548.3, 93.6/533, 408/753.7, 114/550.6, 129.4/680.7]

# Call the function with example data
fault_plot(example_data)


# example_data = [
#     [(0, 0),    (14, 6.3),    (10, 5.7),  (12.7, 6),  (11.7, 4),     (11.7, 3)],
#     [(6.3, 14), (0, 0),       (8, 7.7),   (14, 7.7),  (12.3, 10.3),  (7, 6)],
#     [(5.7, 10), (7.7, 8),     (0, 0),     (14.3, 4.3),(8.3, 5.7),    (9, 6)],
#     [(6, 12.7), (7.7, 14),    (4.3, 14.3),(0, 0),    (8.7, 13),     (8.7, 9.7)],
#     [(4, 11.7), (10.3, 12.3), (5.7, 8.3), (13, 8.7),  (0, 0),        (7.7, 9.3)],
#     [(3, 11.7),  (6, 7),      (6, 9),     (9.7, 8.7), (9.3, 7.7),    (0, 0)]
# ]
# generate_plot(example_data)







