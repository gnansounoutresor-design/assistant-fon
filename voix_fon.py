import torch
import soundfile as sf
from transformers import VitsModel, AutoTokenizer

MODEL_NAME = "facebook/mms-tts-fon"

print("Chargement du modèle vocal Fon...")
print("Le premier téléchargement peut prendre quelques minutes.")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = VitsModel.from_pretrained(MODEL_NAME)
model.eval()

texte = input("\nÉcris une phrase en Fon : ").strip()

if not texte:
    texte = "Bonjour, comment vas-tu ?"

print("\nGénération de la voix...")

inputs = tokenizer(text=texte, return_tensors="pt")

with torch.no_grad():
    waveform = model(**inputs).waveform.squeeze().cpu().numpy()

sf.write("reponse_fon.wav", waveform, model.config.sampling_rate)

print("\nSuccès ! Fichier audio créé :")
print("reponse_fon.wav")
