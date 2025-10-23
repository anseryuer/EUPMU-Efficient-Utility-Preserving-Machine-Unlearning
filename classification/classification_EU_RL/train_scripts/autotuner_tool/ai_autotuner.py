import subprocess
import re
import json
import csv
import os
import datetime
from google import genai

import ast
# ==============================================================================
# 1. HOW TO USE
# ==============================================================================
#
# STEP 1: SET YOUR API KEY
#   - It's best to set this as an environment variable for security.
#   - In your terminal (Linux/macOS): export GEMINI_API_KEY='YOUR_API_KEY_HERE'
#   - Or in PowerShell (Windows): $env:GEMINI_API_KEY='YOUR_API_KEY_HERE'
#   - If you prefer, you can uncomment the line below and paste your key directly.
#
# API_KEY = "YOUR_API_KEY_HERE" 

# STEP 2: DEFINE YOUR ORIGINAL MODEL METRICS
#   - The custom loss function requires baseline metrics from your original, pre-unlearning model.
#   - You need to provide these values for each task.
#
# STEP 3: CONFIGURE YOUR TUNING TASKS
#   - Go to the `TASKS` list below.
#   - Each dictionary in the list is a separate tuning job.
#   - 'name': A unique name for the task (used in the results CSV).
#   - 'base_command': The part of your python command that DOES NOT CHANGE during tuning.
#   - 'tunable_params': A dictionary of parameters to tune, with their INITIAL values.
#   - 'iterations': How many tuning loops to run for this task.
#   - 'original_model_metrics': The baseline metrics for the loss function.
#
# STEP 4: RUN THE SCRIPT
#   - Open your terminal in the same directory as this script and your main script.
#   - Run the command: python autotuner.py
#
# ==============================================================================
# 2. CONFIGURATION
# ==============================================================================

# Attempt to get API key from environment variables
API_KEY = os.getenv("GEMINI_API_KEY")

# --- DEFINE YOUR TUNING TASKS HERE ---
TASKS = [
    # {
    #     "name": "EUPMU_TinyImagenet_10_percent_forget",
    #     "base_command": [
    #         "python", "-u", "main_random.py",
    #         "--arch", "resnet18",
    #         "--dataset", "TinyImagenet",
    #         "--data", "../data/tiny-imagenet-200",
    #         "--unlearn", "RL",
    #         "--unlearn_epochs", "5",
    #         "--num_indexes_to_replace", "10000",
    #         "--mask", "pretrained_models/resnet18/TinyImagenet/model_SA_best.pth.tar",
    #         "--save_dir", "output",
    #         "--gpu", "0",
    #         "--mtl",
    #         "--mtl_method", "eu",
    #     ],
    #     "tunable_params": {
    #         "--eu_w_lr": 5,
    #         "--eu_error": 0.25,
    #         "--unlearn_lr": 3e-3,
    #     },
    #     "iterations": 30, # Number of times to ask Gemini for new parameters
    #     "original_model_metrics": {
    #         # !! IMPORTANT !!: You must fill these values from your original model's performance
    #         "train_acc": 95.29295293075,
    #         "test_acc": 66.35327055492886
    #     }
    # },
    # # --- ADD MORE TASKS HERE ---
    # # Example for another task (you would need to fill in the details)
    {
        "name": "EUPMU_TinyImagenet_10_percent_forget",
        "base_command": [
            "python", "-u", "main_random.py",
            "--arch", "resnet18",
            "--dataset", "TinyImagenet",
            "--data", "../data/tiny-imagenet-200",
            "--unlearn", "RL",
            "--unlearn_epochs", "5",
            "--num_indexes_to_replace", "30000",
            "--mask", "pretrained_models/resnet18/TinyImagenet/model_SA_best.pth.tar",
            "--save_dir", "output",
            "--gpu", "0",
            "--mtl",
            "--mtl_method", "eu",
        ],
        "tunable_params": {
            "--eu_w_lr": 5,
            "--eu_error": 0.25,
            "--unlearn_lr": 3e-3,
        },
        "iterations": 8, # Number of times to ask Gemini for new parameters
        "original_model_metrics": {
            # !! IMPORTANT !!: You must fill these values from your original model's performance
            "train_acc": 95.29295293075,
            "test_acc": 66.35327055492886
        }
    },
    {
        "name": "EUPMU_TinyImagenet_10_percent_forget",
        "base_command": [
            "python", "-u", "main_random.py",
            "--arch", "resnet18",
            "--dataset", "TinyImagenet",
            "--data", "../data/tiny-imagenet-200",
            "--unlearn", "RL",
            "--unlearn_epochs", "5",
            "--num_indexes_to_replace", "50000",
            "--mask", "pretrained_models/resnet18/TinyImagenet/model_SA_best.pth.tar",
            "--save_dir", "output",
            "--gpu", "0",
            "--mtl",
            "--mtl_method", "eu",
        ],
        "tunable_params": {
            "--eu_w_lr": 5,
            "--eu_error": 0.25,
            "--unlearn_lr": 3e-3,
        },
        "iterations": 8, # Number of times to ask Gemini for new parameters
        "original_model_metrics": {
            # !! IMPORTANT !!: You must fill these values from your original model's performance
            "train_acc": 95.29295293075,
            "test_acc": 66.35327055492886
        }
    },

]


# ==============================================================================
# 3. CORE LOGIC (You shouldn't need to change this)
# ==============================================================================

def run_experiment(base_command, parameters):
    """Runs the script with given parameters and captures output."""
    command = list(base_command)
    for key, value in parameters.items():
        command.extend([key, str(value)])
    
    print("="*80)
    print(f"EXECUTING: {' '.join(command)}")
    print("="*80)

    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        print("\n--- ERROR RUNNING SCRIPT ---")
        print(f"Return Code: {e.returncode}")
        print("\n--- STDOUT ---")
        print(e.stdout)
        print("\n--- STDERR ---")
        print(e.stderr)
        return None

def parse_output(output):
    """
    Parses the final accuracy block from the script output.
    """
    # This regex specifically targets the final summary line, e.g.,
    # accuracy : {'retain': 99.98, 'forget': 33.28, 'test': 67.75}
    pattern = r"accuracy\s*:\s*({.*})"
    
    match = re.search(pattern, output)
    
    if not match:
        print("Warning: Could not find the final 'accuracy : {...}' block in the output.")
        return None
        
    try:
        # The captured group is a string representation of a dictionary
        dict_str = match.group(1)
        # ast.literal_eval is a safe way to parse a Python literal
        parsed_dict = ast.literal_eval(dict_str)

        # The 'finetuned_train_acc' for your loss function is the 'retain' accuracy
        metrics = {
            'train_acc': float(parsed_dict['retain']),
            'forget_acc': float(parsed_dict['forget']),
            'test_acc': float(parsed_dict['test'])
        }
        
        # Verify that all keys were successfully parsed
        if all(key in metrics for key in ['train_acc', 'forget_acc', 'test_acc']):
            return metrics
        else:
            print("Warning: Parsed dictionary was missing required keys ('retain', 'forget', 'test').")
            return None

    except (SyntaxError, ValueError, KeyError) as e:
        print(f"Error parsing the accuracy dictionary string: {e}")
        print(f"Failed to parse: {match.group(1)}")
        return None
    
def _extract_json_dict(text):
    """Best-effort extraction of the first JSON object in a text blob."""
    cleaned = text.strip().replace("```json", "").replace("```", "").strip()

    if not cleaned:
        return None

    try:
        parsed = json.loads(cleaned)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        decoder = json.JSONDecoder()
        for idx, char in enumerate(cleaned):
            if char == '{':
                try:
                    parsed, _ = decoder.raw_decode(cleaned[idx:])
                    return parsed if isinstance(parsed, dict) else None
                except json.JSONDecodeError:
                    continue
    return None


def get_new_parameters_from_gemini(task_name, history, tunable_param_keys, original_metrics, max_attempts=3):
    """Asks Gemini for new parameter suggestions based on history and objective."""
    
    # The client automatically uses the GEMINI_API_KEY environment variable
    try:
        client = genai.Client()

    except Exception as e:
        print("\n--- ERROR INITIALIZING GEMINI CLIENT ---")
        print(f"This usually means the GEMINI_API_KEY is not set correctly.")
        print(f"Original error: {e}")
        return None

    # Build a readable history for the prompt
    history_str = "Experiment History (lower unlearning difference is better):\n"
    for record in history:
        # Select only tunable params from record
        params = {k: record[k] for k in tunable_param_keys if k in record}

        # Select only metrics (keys with "acc")
        metrics = {k: v for k, v in record.items() if "acc" in k}

        history_str += (
            f"- Parameters: {params}, Metrics: {metrics}, "
            f"Calculated unlearning difference: {record['loss']:.4f}\n"
        )

    prompt = f"""
You are an expert Machine Learning Engineer specializing in hyperparameter optimization.
I am performing a machine unlearning task called '{task_name}'.

My goal is to find the optimal hyperparameters to MINIMIZE a specific unlearning difference value.
The unlearning difference is calculated as:
unlearning difference = abs(original_train_acc - finetuned_train_acc) + abs(original_test_acc - finetuned_test_acc) + abs((100 - finetuned_forget_acc) - original_test_acc)

Notice that the forget accuracy is inverted in the last term: we want it to be as close to 100 - original_test_acc as possible.

The original model's performance was:
- original_train_acc: {original_metrics['train_acc']}
- original_test_acc: {original_metrics['test_acc']}

The parameters you can adjust are: {list(tunable_param_keys)}

Here is the history of the experiments I have run so far:
{history_str}

Analyze the history. See which parameter changes led to a lower loss.
Based on your analysis, suggest the BEST new set of hyperparameters for the next experiment.
Think step-by-step: if increasing a parameter made the loss worse, try decreasing it, and vice-versa. Explore the parameter space intelligently.

Provide your answer ONLY in a valid JSON format like this, with no other text or explanation:
{{
    "parameter1_name": value1,
    "parameter2_name": value2
}}
"""

    for attempt in range(1, max_attempts + 1):
        if attempt == 1:
            print("\nAsking Gemini for new parameter suggestions...")
        else:
            print(f"Retrying Gemini request ({attempt}/{max_attempts})...")

        response = None
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash",  # Using the newer, specified model
                contents=prompt
            )
        except Exception as e:
            print(f"\n--- ERROR WITH GEMINI API ---")
            print(f"An error occurred: {e}")
            if response is not None and getattr(response, "text", None):
                print(f"Received response: {response.text}")
            if attempt == max_attempts:
                return None
            continue

        suggested_params = _extract_json_dict(response.text)
        if suggested_params is not None:
            return suggested_params

        preview = response.text.strip().splitlines()
        preview = " ".join(preview[:3]) if preview else "<empty response>"
        print("Gemini returned a response that could not be parsed as JSON.")
        print(f"Response preview: {preview}")
        if attempt == max_attempts:
            print("Giving up after repeated invalid responses from Gemini.")
            return None


def save_results_to_csv(filepath, data_dict):
    """Appends a dictionary of results to a CSV file."""
    file_exists = os.path.isfile(filepath)
    with open(filepath, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=data_dict.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(data_dict)

def main():
    """Main orchestration loop for running all tuning tasks."""

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    results_filename = f"tuning_results_{timestamp}.csv"
    
    print(f"Starting hyperparameter tuning. Results will be saved to '{results_filename}'")
    
    for task in TASKS:
        print("\n" + "#"*40)
        print(f"### Starting Task: {task['name']}")
        print("#"*40 + "\n")

        current_parameters = task['tunable_params'].copy()
        experiment_history = []
        original_metrics = task['original_model_metrics']
        
        for i in range(task['iterations']):
            print(f"\n--- Task '{task['name']}' | Iteration {i+1}/{task['iterations']} ---")
            
            output = run_experiment(task['base_command'], current_parameters)
            
            if not output:
                print("Experiment failed to run. Skipping to next task.")
                break
                
            metrics = parse_output(output)
            
            if not metrics:
                print("Failed to parse output. Skipping to next task.")
                break

            # Calculate the custom loss
            loss = (abs(original_metrics['train_acc'] - metrics['train_acc']) +
                    abs(original_metrics['test_acc'] - metrics['test_acc']) +
                    abs(100 - (metrics['forget_acc']) - original_metrics['test_acc']))
            
            print(f"\nMetrics Found: {metrics}")
            print(f"Calculated Loss: {loss:.4f} (Lower is better)")

            # Record and save the result
            result_record = {
                "task_name": task['name'],
                "iteration": i + 1,
                **current_parameters,
                **metrics,
                "loss": round(loss, 4)
            }
            experiment_history.append(result_record)
            save_results_to_csv(results_filename, result_record)
            
            # Get new parameters for the next iteration
            new_params = get_new_parameters_from_gemini(
                task['name'],
                experiment_history,
                task['tunable_params'].keys(),
                original_metrics
            )
            
            if new_params:
                current_parameters.update(new_params)
            else:
                print("Could not get new parameters from Gemini. Ending this task.")
                break
    
    print("\n\n" + "="*50)
    print("ALL TASKS COMPLETED!")
    print(f"Final results are saved in '{results_filename}'")
    print("="*50)


if __name__ == "__main__":
    main()
