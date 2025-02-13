import library

def quiz_monitoring():
    options = ["Model accuracy and training speed", "Fairness, drift detection, and response to live data", "Dataset size and number of features"]
    correct_answer_idx = 1
    description = "Which key metrics should be tracked to detect if a model or data starts behaving strangely in production?"
    correct_answer_message = "Correct choice!🔦"
    incorrect_answer_message = "Incorrect choice. Please try again."

    library.create_dropdown(
        options = options,
        correct_answer_idx=correct_answer_idx,
        description = description,
        correct_answer_message = correct_answer_message,
        incorrect_answer_message = incorrect_answer_message,
    )

def quiz_drift():
    options = ["How fast the model trains on new datasets", "Whether the model was trained on the wrong algorithm", "When the model becomes outdated because of changes in live data"]
    correct_answer_idx = 2
    description = "What does drift detection help us identify in machine learning models?"
    correct_answer_message = "Correct choice!🎈"
    incorrect_answer_message = "Incorrect choice. Please try again."

    library.create_dropdown(
        options = options,
        correct_answer_idx=correct_answer_idx,
        description = description,
        correct_answer_message = correct_answer_message,
        incorrect_answer_message = incorrect_answer_message,
    )

def quiz_shap():
    options = ["It helps models remix their predictions into a top-chart hit", "It provides transparency into how features contribute to individual predictions", "It teaches models to sing in harmony with the data"]
    correct_answer_idx = 1
    description = "Why is SHAP particularly useful in machine learning?"
    correct_answer_message = "Correct choice!🍰"
    incorrect_answer_message = "Incorrect choice. Please try again."

    library.create_dropdown(
        options = options,
        correct_answer_idx=correct_answer_idx,
        description = description,
        correct_answer_message = correct_answer_message,
        incorrect_answer_message = incorrect_answer_message,
    )