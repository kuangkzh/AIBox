import torch
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
from diffusers.utils import export_to_video

def text_to_video(prompt,output_slot_id):

    model_dir = "cache/model"
    pipe = DiffusionPipeline.from_pretrained("damo-vilab/text-to-video-ms-1.7b", torch_dtype=torch.float16, variant="fp16")
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()

    #prompt = "Spiderman is surfing"
    video_frames = pipe(prompt, num_inference_steps=25).frames
    video_path = export_to_video(video_frames)

    return video_path
