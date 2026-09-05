import gradio as gr
from aimodel import Model
import pandas as pd
def ui(file):
    file_path = file.name
    model = Model()
    output_file = model(file_path)
    df = pd.read_csv(output_file)
    return output_file , df
interface = gr.Interface(fn = ui, 
                         inputs = gr.File(label = "Upload csv or xlsx or xls.", 
                                          file_types = [".csv", ".xlsx", ".xls"]), 
                         outputs = [gr.Textbox(label = "Predicted results file path."), 
                                    gr.Dataframe(label = "Result Preview.")],
                         title = "Freshers 404 Project",
                         description= "Upload a dataet to retrieve its predictions." )
interface.launch(inbrowser= True)
