import ipywidgets as widgets
from IPython.display import display, clear_output

empty_message = "Don't forget to answer the quiz!:D"

def create_dropdown(
    options: list,
    correct_answer_idx: int,
    description: str = "",
    correct_answer_message: str = "Correct choice! Well done.",
    incorrect_answer_message: str = "Incorrect choice. Please try again."
):
    dropdown = widgets.Dropdown(
        options=options,
        value=None,
        description="Choose:",
        disabled=False,
    )

    print(description)
    
    output = widgets.Output()

    with output:
        raise Exception(empty_message)
    
    def on_dropdown_change(change):
        if change['new'] is None:
            with output:
                clear_output(wait=True)
                print(empty_message)
        else:
            with output:
                clear_output(wait=True)
                if change['new'] == options[correct_answer_idx]:
                    print(correct_answer_message)
                else:
                    print(incorrect_answer_message)
    
    dropdown.observe(on_dropdown_change, names='value')
    
    display(dropdown, output)


def create_slider(
    options: list,
    value: any,
    correct_value: any,
    description: str = "",
    correct_answer_message: str = "Correct choice! Well done.",
    incorrect_answer_message: str = "Incorrect choice. Please try again."
):
    slider = widgets.SelectionSlider(
        options=options,
        value=value,
        description='Choose:',
        continuous_update=False
    )

    print(description)

    output = widgets.Output()

    with output:
        raise Exception(empty_message)

    def on_slider_change(change):
        with output:
            clear_output(wait=True)
            value = change['new']
            if value == correct_value:
                print(correct_answer_message)
            else:
                print(incorrect_answer_message)
    
    slider.observe(on_slider_change, names='value')
    
    display(slider, output)