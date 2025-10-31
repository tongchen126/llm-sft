import json
import matplotlib.pyplot as plt
from collections import Counter
import os

def plot_label_distribution(json_path, title="Label Distribution", save_path='distribution.png'):
    """
    Plot the distribution of labels showing how many labels have 1, 2, 3, etc. records.
    
    Args:
        json_path: Path to JSON file containing list of records
        title: Title for the plot (default: "Label Distribution")
    
    Returns:
        dict: Distribution data (record_count -> number_of_labels)
    """
    # Load records from JSON file
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            records = json.load(f)
    except FileNotFoundError:
        print(f"Error: File not found at {json_path}")
        return {}
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in {json_path}")
        return {}
    
    if not records:
        print("No records to plot")
        return {}
    
    if not isinstance(records, list):
        print("Error: JSON file must contain a list of records")
        return {}
    
    # Count records per label
    label_counts = Counter(record['label'] for record in records)
    
    # Count how many labels have each record count
    # e.g., {1: 5, 2: 3, 3: 1} means 5 labels have 1 record, 3 labels have 2 records, etc.
    distribution = Counter(label_counts.values())
    
    # Sort by record count for plotting
    record_counts = sorted(distribution.keys())
    label_frequencies = [distribution[count] for count in record_counts]
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.bar(record_counts, label_frequencies, color='steelblue', edgecolor='black', alpha=0.7)
    
    plt.xlabel('Number of Records per Label', fontsize=12)
    plt.ylabel('Number of Labels', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add value labels on top of bars
    for x, y in zip(record_counts, label_frequencies):
        plt.text(x, y, str(y), ha='center', va='bottom', fontsize=10)
    
    # Add statistics text
    total_labels = len(label_counts)
    total_records = len(records)
    avg_records = total_records / total_labels if total_labels > 0 else 0
    
    stats_text = f'Total Labels: {total_labels}\nTotal Records: {total_records}\nAvg Records/Label: {avg_records:.2f}'
    plt.text(0.98, 0.97, stats_text, transform=plt.gca().transAxes,
             fontsize=10, verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(save_path)
    
    # Print summary
    print(f"\n{'='*50}")
    print(f"Label Distribution Summary - {json_path}")
    print(f"{'='*50}")
    print(f"Total unique labels: {total_labels}")
    print(f"Total records: {total_records}")
    print(f"Average records per label: {avg_records:.2f}")
    print(f"Distribution:")
    for count in record_counts:
        num_labels = distribution[count]
        percentage = (num_labels / total_labels) * 100
        print(f"  {num_labels} label(s) with {count} record(s) ({percentage:.1f}%)")
    print(f"{'='*50}\n")
    
    return dict(distribution)

def plot_losses_from_json(file_paths, names, output_path='loss_plot.png', 
                          x_axis='step', figsize=(10, 6)):
    """
    Extract loss values from JSON files and create a plot.
    
    Parameters:
    -----------
    file_paths : list of str
        List of paths to JSON files
    names : list of str
        List of names corresponding to each file path (for legend)
    output_path : str
        Path where the plot will be saved (default: 'loss_plot.png')
    x_axis : str
        What to use for x-axis: 'step' or 'epoch' (default: 'step')
    figsize : tuple
        Figure size (width, height) in inches
    """
    
    if len(file_paths) != len(names):
        raise ValueError("Number of file paths must match number of names")
    
    plt.figure(figsize=figsize)
    
    # Process each JSON file
    for file_path, name in zip(file_paths, names):
        # Read JSON file
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Extract log history
        log_history = data.get('log_history', [])
        
        # Extract x-axis values and losses
        x_values = []
        losses = []
        
        for entry in log_history:
            if 'loss' in entry:  # Only include entries that have loss
                x_values.append(entry.get(x_axis, 0))
                losses.append(entry['loss'])
        
        # Plot the line
        if x_values and losses:
            plt.plot(x_values, losses, marker='o', label=name, linewidth=2, markersize=4)
        else:
            print(f"Warning: No loss data found in {file_path}")
    
    # Customize the plot
    plt.xlabel(x_axis.capitalize(), fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training Loss Comparison', fontsize=14, fontweight='bold')
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_path}")
    
    # Optionally display the plot
    # plt.show()
    
    plt.close()

def list_directories(path):
    """
    List all directories in the given path.
    
    Args:
        path (str): The path to search for directories
        
    Returns:
        dict: A dictionary with directory names as keys and full paths as values
    """
    dir_dict = {}
    
    try:
        # Check if the path exists
        if not os.path.exists(path):
            raise ValueError(f"Path does not exist: {path}")
        
        # Check if the path is a directory
        if not os.path.isdir(path):
            raise ValueError(f"Path is not a directory: {path}")
        
        # List all items in the directory
        for item in os.listdir(path):
            full_path = os.path.join(path, item)
            
            # Check if the item is a directory
            if os.path.isdir(full_path):
                dir_dict[item] = os.path.abspath(full_path)
                
    except PermissionError:
        print(f"Permission denied: {path}")
    except Exception as e:
        print(f"Error: {e}")
    
    return dir_dict

if __name__ == '__main__':
    out_path = "data/pokemon1/"
    print(plot_label_distribution(out_path + "data.json", save_path = out_path + "data.png"))
    print(plot_label_distribution(out_path + "data_eval.json", save_path = out_path + "data_eval.png"))