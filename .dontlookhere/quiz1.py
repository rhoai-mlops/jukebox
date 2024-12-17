import library

def quiz_about_elephants():
    options = ['Option 1', 'AAAAAAAAAAAAAAAAAAA😂AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA\nBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB🐝', 'Option 3', 'Correct Option']
    correct_answer_idx = 3
    description = "What is your favorite choice? "
    correct_answer_message = "Correct choice! Well done."
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