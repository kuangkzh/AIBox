from transformers import AutoTokenizer, AutoModel, AutoProcessor
import re
import ImageCaptioning, SpeechRecognition, TextToImage, TextToVideo, VisualQuestionAnswering


class Controller:
    def __call__(self, input_str):
        if self.explain(input_str) == -1:
            return "Failed to find a proper model."

        if self.model_name == "vit-gpt2-image-captioning":
            return ImageCaptioning.vit_gpt2_image_captioning(self.model_args)
        if self.model_name == "whisper-base":
            return SpeechRecognition.whisper_base(*self.model_args)
        if self.model_name == "stable-diffusion-v1-5":
            return TextToImage.stable_diffusion_v1_5(*self.model_args)
        if self.model_name == "text-to-video-ms-1.7b":
            return TextToVideo.text_to_video(*self.model_args)
        if self.model_name == "vilt-b32-finetuned-vqa":
            return VisualQuestionAnswering.visual_question_answer(*self.model_args)
        if self.model_name == "python":
            return eval(self.model_args[0])

        # return("model name: {}; model args: {}.".format(self.model_name, self.model_args))

    def explain(self, input_str):
        start_pos = input_str.find("<|call|>")
        if start_pos < 0:
            return -1
        start_pos += 8
        end_pos = input_str.find("<|end|>")
        if end_pos < 0:
            return -1
        call_str = input_str[start_pos:end_pos]

        start_pos = call_str.find("[")
        if start_pos < 0:
            return -1
        end_pos = call_str.find("]")
        if end_pos < 0:
            return -1
        self.model_name = call_str[start_pos + 1 : end_pos]

        start_pos = call_str.find("(")
        if start_pos < 0:
            return -1
        end_pos = call_str.find(")")
        if end_pos < 0:
            return -1
        args = eval(call_str[start_pos + 1 : end_pos])
        if type(args) == int or type(args) == str:
            self.model_args = [args]
        else:
            self.model_args = list(args)
        return 1


if __name__ == "__main__":
    controller = Controller()
    print(
        controller(
            '<|call|>[stable-diffusion-v1-5]("sunset", 10)<|end|>./cache/fileslots/10.png<|endofcall|>'
        )
    )
