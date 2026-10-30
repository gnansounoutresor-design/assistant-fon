import torch
import soundfile as sf
from transformers import VitsModel, AutoTokenizer

MODEL_NAME = "facebook/mms-tts-fon"
TEXT = "N mi kpɔn bo, nukun nyi na mi."

print("Chargement du modèle vocal...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = VitsModel.from_pretrained(MODEL_NAME)

print("Préparation du texte...")
inputs = tokenizer(text=TEXT, return_tensors="pt")

print("Génération audio...")
with torch.no_grad():
    output = model(**inputs).waveform

audio = output.squeeze().cpu().numpy()
sf.write("test_voix_fon.wav", audio, model.config.sampling_rate)

print("Succès : fichier test_voix_fon.wav créé.")
