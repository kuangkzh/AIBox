from diffusers import StableDiffusionPipeline
import torch

def stable_diffusion_v1_5(prompt, output_slot_id):
    # 输入字符串
    model_dir = "cache/model"

    model_id = "runwayml/stable-diffusion-v1-5"
    pipe = StableDiffusionPipeline.from_pretrained(model_id)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipe.to(device)

    image = pipe(prompt).images[0]  
        
    image.save(f"./cache/fileslots/{output_slot_id}.png")

    return f"./cache/fileslots/{output_slot_id}.png"
