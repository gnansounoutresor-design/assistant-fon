import requests
import torch
import soundfile as sf
from transformers import VitsModel, AutoTokenizer

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen3:4b"
TTS_MODEL = "facebook/mms-tts-fon"

print("1. Interrogation d'Ollama...")
response = requests.post(
    OLLAMA_URL,
    json={
        "model": OLLAMA_MODEL,
        "prompt": "Réponds en français en une phrase courte : salue l'utilisateur.",
        "stream": False
    },
    timeout=180
)
response.raise_for_status()
texte_fr = response.json()["response"]
print("Réponse d'Ollama :", texte_fr)

print("\n2. Chargement du modèle vocal Fon de Facebook...")
tokenizer = AutoTokenizer.from_pretrained(TTS_MODEL)
model = VitsModel.from_pretrained(TTS_MODEL)
model.eval()

print("\n3. Pour ce premier test, saisis une phrase en Fon.")
print("Attention : la réponse française d'Ollama n'est pas automatiquement traduite.")
texte_fon = input("Phrase en Fon à prononcer : ").strip()

if not texte_fon:
    raise SystemExit("Aucune phrase en Fon fournie. Test arrêté.")

inputs = tokenizer(text=texte_fon, return_tensors="pt")
with torch.no_grad():
    audio = model(**inputs).waveform.squeeze().cpu().numpy()

sf.write("reponse_fon.wav", audio, model.config.sampling_rate)

print("\nTerminé ! Audio créé :")
print("reponse_fon.wav")
print("Réponse française d'Ollama :", texte_fr)
