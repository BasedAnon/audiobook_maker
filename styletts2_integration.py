import torch
from styletts2 import StyleTTS2

class StyleTTS2Integration:
    def __init__(self):
        self.model = StyleTTS2()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def generate_speech(self, text, output_path):
        try:
            wav = self.model.inference(text)
            self.model.save_wav(wav, output_path)
            return True
        except Exception as e:
            print(f"Error generating speech with StyleTTS2: {str(e)}")
            return False
