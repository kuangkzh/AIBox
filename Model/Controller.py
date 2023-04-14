from transformers import AutoTokenizer, AutoModel

class Controller:
    def __call__(self, model_name, input_slot_ids, output_slot):
        model = AutoModel.from_pretrained(model_name)
        
        