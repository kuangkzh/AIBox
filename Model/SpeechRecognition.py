import torchaudio
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from datasets import load_dataset

def whisper_base(input_slot_ids):
    # load model and processor
    processor = WhisperProcessor.from_pretrained("openai/whisper-base")
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base")
    model.config.forced_decoder_ids = None

    # load dummy dataset and read audio files
    ds = load_dataset(f"./cache/fileslots/{input_slot_ids}.mp3", "clean", split="validation")
    sample = ds[0]["audio"]
    wav, sr = torchaudio.load("C:\Program Files\MATLAB\R2020b\examples\deeplearning_shared\data\FemaleSpeech-16-4-mono-405secs.wav")
    input_features = processor(wav, sampling_rate=sr, return_tensors="pt").input_features 

    # generate token ids
    predicted_ids = model.generate(input_features)
    # decode token ids to text
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=False)

    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
