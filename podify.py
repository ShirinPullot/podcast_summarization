
import gradio as gr 
from podcast_summarization.genism_summarizer import get_genism_summary




title="Podify"
description="Get Podcast summarization"
block = gr.Blocks()

with block:
    gr.HTML(
        """     <center> 
                <h1>Podify</h1>
                <img src = 'https://www.allabtai.com/wp-content/uploads/2023/03/AllAboutAI_-_Kris_Female_Cyborg_Podcast_podcast_microphone_3fbca7de-0c5a-4f01-99e5-94e89c22e072.jpg' width = '50%'></img>
                </center>
        """
    )
    with gr.Group():
        with gr.Box():
          
          transcript = gr.Textbox(label="Podcasts transcript")

          with gr.Row().style(mobile_collapse=False, equal_height=True): 
              btn = gr.Button("Podify 🪄")
          
          text = gr.Textbox(
              label="Podcast summary", 
              placeholder="Podify Output",
              lines=5)
           
       
          
          btn.click(get_genism_summary, inputs=[transcript], outputs=[text])

block.launch(debug=True, share=True)
