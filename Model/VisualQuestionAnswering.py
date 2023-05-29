from transformers import ViltProcessor, ViltForQuestionAnswering
import requests
from PIL import Image

def visual_question_answer(prompt,input_slot_id):
    # prepare image + question
    image_path = f"./cache/fileslots/{input_slot_id}.jpg"
    image = Image.open(image_path)
    #text = "How many cats are there?"
    text=prompt

    processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
    model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")

    # prepare inputs
    encoding = processor(image, text, return_tensors="pt")

    # forward pass
    outputs = model(**encoding)
    logits = outputs.logits
    idx = logits.argmax(-1).item()
    return(model.config.id2label[idx])
