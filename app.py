import gradio as gr
import sys
from starline import process

from utils import load_cn_model, load_cn_config, randomname
from convertor import pil2cv, cv2pil

from sd_model import get_cn_pipeline, generate, get_cn_detector
import cv2
import os
import numpy as np
from PIL import Image


path = os.getcwd()
output_dir = f"{path}/output"
input_dir = f"{path}/input"
cn_lineart_dir = f"{path}/controlnet/lineart"

load_cn_model(cn_lineart_dir)
load_cn_config(cn_lineart_dir)

class webui:
    def __init__(self):
        self.demo = gr.Blocks()

    def undercoat(self, input_image, pos_prompt, neg_prompt, alpha_th):
        org_line_image = input_image
        image = pil2cv(input_image)
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)

        index = np.where(image[:, :, 3] == 0)
        image[index] = [255, 255, 255, 255]
        input_image = cv2pil(image)

        pipe = get_cn_pipeline()
        detectors = get_cn_detector(input_image.resize((1024, 1024), Image.ANTIALIAS))
        

        gen_image = generate(pipe, detectors, pos_prompt, neg_prompt)
        output = process(gen_image.resize((image.shape[1], image.shape[0]), Image.ANTIALIAS) , org_line_image, alpha_th)

        output = output.resize((image.shape[1], image.shape[0]) , Image.ANTIALIAS)


        output = Image.alpha_composite(output, org_line_image)
        name = randomname(10)
        output.save(f"{output_dir}/output_{name}.png")
        #output = pil2cv(output)
        file_name = f"{output_dir}/output_{name}.png"

        return output, file_name



    def launch(self, share):
        with self.demo:
            with gr.Row():
                with gr.Column():
                    input_image = gr.Image(type="pil", image_mode="RGBA")

                    pos_prompt = gr.Textbox(max_lines=1000, label="positive prompt")                    
                    neg_prompt = gr.Textbox(max_lines=1000, label="negative prompt")

                    alpha_th = gr.Slider(maximum = 255, value=100, label = "alpha threshold")

                    submit = gr.Button(value="Start")
                with gr.Row():
                    with gr.Column():
                        with gr.Tab("output"):
                            output_0 = gr.Image()

                    output_file = gr.File()
            submit.click(
                self.undercoat, 
                inputs=[input_image, pos_prompt, neg_prompt, alpha_th], 
                outputs=[output_0, output_file]
            )

        self.demo.queue()
        self.demo.launch(share=share)


if __name__ == "__main__":
    ui = webui()
    if len(sys.argv) > 1:
        if sys.argv[1] == "share":
            ui.launch(share=True)
        else:
            ui.launch(share=False)
    else:
        ui.launch(share=False)
