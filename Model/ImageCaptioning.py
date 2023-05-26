from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
import torch
from PIL import Image



def vit_gpt2_image_captioning(input_slot_ids):
  # 根据槽位号获取文件路径，文件名为槽位号+后缀
  model_dir = "cache/model"
  fileslot_dir = "./cache/fileslots/"
  file_paths = [fileslot_dir + str(sl_id) + '.png' for sl_id in input_slot_ids]
  
  # 以下是从huggingface上复制的调用示例代码
  model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning", cache_dir=model_dir)
  feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning", cache_dir=model_dir)
  tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning", cache_dir=model_dir)

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model.to(device)

  max_length = 16
  num_beams = 4
  gen_kwargs = {"max_length": max_length, "num_beams": num_beams}
  
  def predict_step(image_paths):
    images = []
    for image_path in image_paths:
      i_image = Image.open(image_path)
      if i_image.mode != "RGB":
        i_image = i_image.convert(mode="RGB")

      images.append(i_image)

    pixel_values = feature_extractor(images=images, return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(device)

    output_ids = model.generate(pixel_values, **gen_kwargs)

    preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    preds = [pred.strip() for pred in preds]
    return preds
    
  results = predict_step(file_paths)
  
  # 返回结果为文本的模型直接将返回结果的列表传给Controller即可
  # 结果为多模态的模型需要将结果装入文件中，文件命名为文件槽编号+后缀，保存在文件槽目录fileslot_dir下，最后返回文件槽编号列表
  return results
  
  
