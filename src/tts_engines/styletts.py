from styletts2 import StyleTTS2
import torch

class StyleTTS:
    def __init__(self):
        self.model = StyleTTS2()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def tts(self, text, output_path):
        wav = self.model.inference(text)
        self.model.save_wav(wav, output_path)
