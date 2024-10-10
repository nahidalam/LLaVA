import pandas as pd

# Load the Excel file
file_path = './playground/data/eval/mmbench/answers_upload/mmbench_dev_20230712/maya_full_ft.xlsx'
excel_data = pd.read_excel(file_path)

# Extract the 'prediction' and 'answer' columns
predictions = excel_data['prediction']
answers = excel_data['answer']

# Calculate accuracy by comparing predictions to answers
accuracy = (predictions == answers).mean()

# Print the accuracy
print(f"Accuracy: {accuracy * 100:.2f}%")

