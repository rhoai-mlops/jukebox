import library

def quiz_eda():
    options = ["To ensure the model performs exactly as expected without any additional testing", "To directly tune hyperparameters for the ML model", "To understand the data, detect patterns, and identify anomalies"]
    correct_answer_idx = 2
    description = "Why is EDA important before building a machine learning model?"
    correct_answer_message = "Correct choice!👏👏👏"
    incorrect_answer_message = "Incorrect choice. Please try again."

    library.create_dropdown(
        options = options,
        correct_answer_idx=correct_answer_idx,
        description = description,
        correct_answer_message = correct_answer_message,
        incorrect_answer_message = incorrect_answer_message,
    )


def quiz_heatmap():
    options = ["To visualize the popularity of songs over time", "To identify and visualize relationships between numerical features", "To create a playlist based on user preferences."]
    correct_answer_idx = 1
    description = "What is the role of a heatmap in EDA?"
    correct_answer_message = "Correct choice! Well done. ✨"
    incorrect_answer_message = "Incorrect choice. Please try again."

    library.create_dropdown(
        options = options,
        correct_answer_idx=correct_answer_idx,
        description = description,
        correct_answer_message = correct_answer_message,
        incorrect_answer_message = incorrect_answer_message,
    )


def quiz_about_numbers():
    options = ["Many", "Few", "None"]
    start_value = "Many"
    correct_value = "Few"
    description = "Choose a good value "
    correct_answer_message: str = "Correct choice! Well done."
    incorrect_answer_message: str = "Incorrect choice. Please try again."

    library.create_slider(
        options=options,
        value=start_value,
        correct_value=correct_value,
        description = description,
        correct_answer_message = correct_answer_message,
        incorrect_answer_message = incorrect_answer_message,
    )
