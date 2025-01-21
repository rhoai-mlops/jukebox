import library

def quiz_data():
    options = ["Your model might end up remixing the wrong data", "The data starts making up its own rules", "You might lose track of what dataset was used for training"]
    correct_answer_idx = 2
    description = "What happens if you skip data versioning in your ML workflow?"
    correct_answer_message = "Correct choice!🐿️"
    incorrect_answer_message = "Incorrect choice. Please try again."

    library.create_dropdown(
        options = options,
        correct_answer_idx=correct_answer_idx,
        description = description,
        correct_answer_message = correct_answer_message,
        incorrect_answer_message = incorrect_answer_message,
    )

def quiz_versioning():
    options = ["Tagging datasets with dates or version numbers", "Uploading all datasets into a single folder called RandomStuff_NoBackup", "Keeping an immutable record of each version of the dataset"]
    correct_answer_idx = 1
    description = "If K.R.A.P. Records wants to version their top 50 songs dataset, what should they avoid doing?"
    correct_answer_message = "Correct choice!🎶📊"
    incorrect_answer_message = "Incorrect choice. Please try again."

    library.create_dropdown(
        options = options,
        correct_answer_idx=correct_answer_idx,
        description = description,
        correct_answer_message = correct_answer_message,
        incorrect_answer_message = incorrect_answer_message,
    )