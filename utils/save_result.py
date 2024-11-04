import json
import os


def save_accuracy(accuracies, tag, save_path, file_name, duration):
    os.makedirs(save_path, exist_ok=True)
    file_path = os.path.join(save_path, file_name)

    # Load existing data if file exists
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            saved_accuracies = json.load(file)
    else:
        saved_accuracies = {}

    # Update the dictionary with new metrics under the provided tag
    saved_accuracies[tag] = {
        'avg_loss': accuracies['avg_loss'],
        'accuracy': accuracies['accuracy'],
        'r2': accuracies['r2'],
        'mae': accuracies['mae'],
        'mse': accuracies['mse'],
        'rmse': accuracies['rmse'],
        'mape': accuracies['mape'],
        'duration': duration
    }

    # Save the updated dictionary to a JSON file
    with open(file_path, 'w') as file:
        json.dump(saved_accuracies, file, indent=2)