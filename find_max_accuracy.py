import re

def find_max_accuracy(log_file_path):
    try:
        with open(log_file_path, 'r') as file:
            log_contents = file.readlines()

        # Extract accuracy values
        accuracy_values = []
        for line in log_contents:
            match = re.search(r'Accuracy after epoch \d+: (\d+\.\d+)%', line)
            if match:
                accuracy_values.append(float(match.group(1)))

        if accuracy_values:
            max_accuracy = max(accuracy_values)
            return max_accuracy
        else:
            return "No accuracy values found in the log file."

    except FileNotFoundError:
        return f"File not found: {log_file_path}"
    except Exception as e:
        return f"An error occurred: {str(e)}"

# Example usage
if __name__ == "__main__":
    log_file_path = input("Please enter the log file path: ")
    max_accuracy = find_max_accuracy(log_file_path)
    print(f"The maximum Accuracy value in the log file is: {max_accuracy}")
