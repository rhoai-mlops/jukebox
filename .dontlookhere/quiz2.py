import library

def quiz_model():
    options = ["Generative model", "Predictive model", "Descriptive model"]
    correct_answer_idx = 1
    description = "Which type of machine learning model did K.R.A.P. Records choose for Jukebox AI project?"
    correct_answer_message = "Correct choice!🤖"
    incorrect_answer_message = "Incorrect choice. Please try again."

    library.create_dropdown(
        options = options,
        correct_answer_idx=correct_answer_idx,
        description = description,
        correct_answer_message = correct_answer_message,
        incorrect_answer_message = incorrect_answer_message,
    )

def quiz_nn():
    options = ["Because they can generalize better and handle complex patterns", "Because they are faster than other models", "Because they require no training"]
    correct_answer_idx = 0
    description = "Why did K.R.A.P. Records decide to use neural networks for their predictive model?"
    correct_answer_message = "Correct choice!🐳"
    incorrect_answer_message = "Incorrect choice. Please try again."

    library.create_dropdown(
        options = options,
        correct_answer_idx=correct_answer_idx,
        description = description,
        correct_answer_message = correct_answer_message,
        incorrect_answer_message = incorrect_answer_message,
    )

def quiz_versioning(): 
    options = ["Always overwrite previous results to save storage", "Use random names like test1 or final_final_model for experiments", "Save model configurations, metrics, and training parameters for each experiment"]
    correct_answer_idx = 2
    description = "Which of the following is a good practice for managing machine learning experiments?"
    correct_answer_message = "Correct choice!😌"
    incorrect_answer_message = "Incorrect choice. Please try again."

    library.create_dropdown(
        options = options,
        correct_answer_idx=correct_answer_idx,
        description = description,
        correct_answer_message = correct_answer_message,
        incorrect_answer_message = incorrect_answer_message,
    )