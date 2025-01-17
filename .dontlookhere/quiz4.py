import library

def quiz_drift():
    options = ["How fast the model trains on new datasets", "Whether the model was trained on the wrong algorithm", "When the model becomes outdated because of changes in live data"]
    correct_answer_idx = 2
    description = "What does drift detection help us identify in machine learning models?"
    correct_answer_message = "Correct choice!🔦"
    incorrect_answer_message = "Incorrect choice. Please try again."

    library.create_dropdown(
        options = options,
        correct_answer_idx=correct_answer_idx,
        description = description,
        correct_answer_message = correct_answer_message,
        incorrect_answer_message = incorrect_answer_message,
    )