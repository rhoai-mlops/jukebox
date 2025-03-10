import ipywidgets as widgets
from IPython.display import display, clear_output, HTML

empty_message = "Don't forget to answer the quiz!"

def create_dropdown(
    options: list,
    correct_answer_idx: int,
    description: str = "",
    correct_answer_message: str = "✅ Correct choice! Well done. ✅",
    incorrect_answer_message: str = "❌ Incorrect choice. Please try again. ❌"
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
        display(HTML(f'<h3 style="color: #ff0000; text-align: left; padding: 10px; font-weight: bold;">⚠️ {empty_message} ⚠️</h3>'))
    
    def on_dropdown_change(change):
        if change['new'] is None:
            with output:
                clear_output(wait=True)
                display(HTML(f'<h3 style="color: #ff0000; text-align: left; padding: 10px; font-weight: bold;">⚠️ {empty_message} ⚠️</h3>'))
        else:
            with output:
                clear_output(wait=True)
                if change['new'] == options[correct_answer_idx]:
                    display(HTML(f'<h3 style="color: green; text-align: left; padding: 10px; font-weight: bold;">{correct_answer_message}</h3>'))
                else:
                    display(HTML(f'<h3 style="color: red; text-align: left; padding: 10px; font-weight: bold;">{incorrect_answer_message}</h3>'))
    
    dropdown.observe(on_dropdown_change, names='value')
    
    display(dropdown, output)


def create_slider(
    options: list,
    value: any,
    correct_value: any,
    description: str = "",
    correct_answer_message: str = "✅ Correct choice! Well done. ✅",
    incorrect_answer_message: str = "❌ Incorrect choice. Please try again. ❌"
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
        display(HTML(f'<h3 style="color: #ff0000; text-align: left; padding: 10px; font-weight: bold;">⚠️ {empty_message} ⚠️</h3>'))

    def on_slider_change(change):
        with output:
            clear_output(wait=True)
            value = change['new']
            if value == correct_value:
                display(HTML(f'<h3 style="color: green; text-align: left; padding: 10px; font-weight: bold;">{correct_answer_message}</h3>'))
            else:
                display(HTML(f'<h3 style="color: red; text-align: left; padding: 10px; font-weight: bold;">{incorrect_answer_message}</h3>'))
    
    slider.observe(on_slider_change, names='value')
    
    display(slider, output)