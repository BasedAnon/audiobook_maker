# tts_engines/styletts.py

from styletts2 import StyleTTS2
import torch
import soundfile as sf
import os

class StyleTTS:
    def __init__(self):
        self.model = StyleTTS2()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def tts(self, text, output_path, **kwargs):
        try:
            # Generate audio
            wav = self.model.inference(text, **kwargs)

            # Ensure the output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            # Save the audio
            sf.write(output_path, wav.cpu().numpy(), self.model.config.data.sampling_rate)

            return True
        except Exception as e:
            print(f"Error in StyleTTS generation: {str(e)}")
            return False

    @staticmethod
    def get_supported_languages():
        return ["en"]  # StyleTTS2 is primarily designed for English

    @staticmethod
    def get_available_voices():
        return []  # StyleTTS2 doesn't have predefined voices in the same way as some other TTS systems

    @staticmethod
    def get_settings():
        return {
            "speaking_rate": {"type": "float", "default": 1.0, "min": 0.5, "max": 2.0},
            "noise_scale": {"type": "float", "default": 0.667, "min": 0.1, "max": 1.0},
            "noise_scale_w": {"type": "float", "default": 0.8, "min": 0.1, "max": 1.0},
            "length_scale": {"type": "float", "default": 1.0, "min": 0.1, "max": 2.0},
        }
