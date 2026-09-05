import gradio as gr
import pandas as pd
from aimodel import Model

def ui(file_obj):
    if file_obj is None:
        return "No file uploaded.", None

    file_path = file_obj.name
    # Read file
    try:
        if not (file_path.endswith(".csv") or file_path.endswith((".xlsx", ".xls"))):
            return f"Unsupported file format: {file_path}", None
        output_file = Model()(file_path)
        df = pd.read_csv(output_file)
    except Exception as e:
        print(e)
        return f"Error: {str(e)}", None

    return output_file, df


# Build app using gr.Interface
Interface = gr.Interface(
    fn=ui,
    inputs=gr.File(
        label="Upload CSV or XLSX or XLS File", file_types=[".csv", ".xlsx", ".xls"]
    ),
    outputs=[
        gr.Textbox(label="Predicted Results File Path"),
        gr.Dataframe(label="Result Preview"),
    ],
    title="Freshers 404 Project",
    description="Upload a dataset to retrieve its Predictions.",
)

Interface.launch(inbrowser=True)