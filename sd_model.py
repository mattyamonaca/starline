from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel, AutoencoderKL
import torch

device = "cuda"

def get_cn_pipeline():
    controlnets = [
        ControlNetModel.from_pretrained("./controlnet/lineart", torch_dtype=torch.float16, use_safetensors=True),
        ControlNetModel.from_pretrained("mattyamonaca/controlnet_line2line_xl", torch_dtype=torch.float16)
    ]

    vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
    pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
        "cagliostrolab/animagine-xl-3.1", controlnet=controlnets, vae=vae, torch_dtype=torch.float16
    )

    pipe.enable_model_cpu_offload()

    #if pipe.safety_checker is not None:
    #    pipe.safety_checker = lambda images, **kwargs: (images, [False])
    
    #pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    #pipe.to(device)

    return pipe

def invert_image(img):
    # 画像を読み込む
    # 画像をグレースケールに変換（もしもともと白黒でない場合）
    img = img.convert('L')
    # 画像の各ピクセルを反転
    inverted_img = img.point(lambda p: 255 - p)
    # 反転した画像を保存
    return inverted_img


def get_cn_detector(image):
    #lineart_anime = LineartAnimeDetector.from_pretrained("lllyasviel/Annotators")
    #canny = CannyDetector()
    #lineart_anime_img = lineart_anime(image)
    #canny_img = canny(image)
    #canny_img = canny_img.resize((lineart_anime(image).width, lineart_anime(image).height))
    re_image = invert_image(image)
    
    
    detectors = [re_image, image]
    print(detectors)
    return detectors

def generate(pipe, detectors, prompt, negative_prompt):
    default_pos = ""
    default_neg = ""
    prompt = default_pos + prompt 
    negative_prompt = default_neg + negative_prompt 
    print(type(pipe))
    image = pipe(
                prompt=prompt,
                negative_prompt = negative_prompt,
                image=detectors,
                num_inference_steps=50,
                controlnet_conditioning_scale=[1.0, 0.2],
            ).images[0]
    return image