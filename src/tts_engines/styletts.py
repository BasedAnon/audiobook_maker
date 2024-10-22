# tts_engines/styletts.py

import torch
from styletts2 import StyleTTS2
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

    def load_model(self, model_path=None):
        # If a specific model path is provided, load it
        if model_path:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        # Otherwise, the default model is already loaded in __init__

    def set_speaker(self, speaker_id):
        # StyleTTS2 doesn't use speaker IDs in the same way as some other TTS systems
        # This method might not be necessary, or could be used to set a reference audio
        pass

    @staticmethod
    def get_supported_languages():
        # Return a list of supported languages, if applicable
        return ["en"]  # Assuming English is supported, adjust as needed

    @staticmethod
    def get_available_voices():
        # StyleTTS2 doesn't have predefined voices, so this might return an empty list
        return []

    @staticmethod
    def get_settings():
        # Return a dictionary of available settings for the UI
        return {
            "temperature": {"type": "float", "default": 0.667, "min": 0.0, "max": 1.0},
            "length_scale": {"type": "float", "default": 1.0, "min": 0.1, "max": 2.0},
        }
