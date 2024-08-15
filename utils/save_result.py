import json
import os


def save_accuracy(accuracy, tag, save_path, file_name, duration):
    os.makedirs(save_path, exist_ok=True)
    file_path = os.path.join(save_path, file_name)

    # # Load existing data if file exists
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            accuracies = json.load(file)
    else:
        accuracies = {}


    # Update the model_accuracy dictionary
    accuracies[tag] = {
        'accuracy': accuracy,
        'duration': duration,
    }

    # Save the updated dictionary to a JSON file
    with open(file_path, 'w') as file:
        json.dump(accuracies, file, indent=2)