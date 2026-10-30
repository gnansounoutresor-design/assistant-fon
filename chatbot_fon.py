import os
import requests
import torch
import soundfile as sf
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, VitsModel

OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "qwen:latest"
TRANSLATION_MODEL = "nellaw/nllb-fongbe-french-translation"
VOICE_MODEL = "facebook/mms-tts-fon"
AUDIO_FILE = "reponse_fon.wav"

print("\n=== CHATBOT FRANÇAIS - FON ===")
print("Chargement des modèles. Le premier lancement peut prendre du temps.\n")

print("1/3 - Chargement du traducteur...")
translator_tokenizer = AutoTokenizer.from_pretrained(
    TRANSLATION_MODEL,
    src_lang="fra_Latn",
    tgt_lang="fon_Latn"
)
translator = AutoModelForSeq2SeqLM.from_pretrained(TRANSLATION_MODEL)

print("2/3 - Chargement de la voix Fon...")
voice_tokenizer = AutoTokenizer.from_pretrained(VOICE_MODEL)
voice_model = VitsModel.from_pretrained(VOICE_MODEL)

print("3/3 - Vérification d'Ollama...")
try:
    response = requests.get("http://localhost:11434/api/tags", timeout=10)
    response.raise_for_status()
    available_models = [
        item["name"] for item in response.json().get("models", [])
    ]
    if OLLAMA_MODEL not in available_models:
        raise RuntimeError(
            f"Le modèle {OLLAMA_MODEL} n'est pas disponible. "
            f"Modèles détectés : {available_models}"
        )
except Exception as error:
    raise SystemExit(
        f"Impossible de joindre Ollama : {error}\n"
        "Vérifie qu'Ollama est lancé."
    )

history = [
    {
        "role": "system",
        "content": (
            "Tu es un assistant conversationnel utile, patient et honnête. "
            "L'utilisateur parle français. Réponds en français, clairement "
            "et naturellement. Tiens compte des messages précédents. "
            "N'invente pas de faits. Pour une question ambiguë, demande "
            "une précision. Fais des réponses assez courtes pour faciliter "
            "leur traduction en fon."
        )
    }
]

def traduire_en_fon(texte):
    inputs = translator_tokenizer(
        texte,
        return_tensors="pt",
        truncation=True,
        max_length=256
    )
    with torch.no_grad():
        outputs = translator.generate(
            **inputs,
            max_new_tokens=256,
            num_beams=3,
            forced_bos_token_id=translator_tokenizer.convert_tokens_to_ids(
                "fon_Latn"
            )
        )
    return translator_tokenizer.decode(
        outputs[0], skip_special_tokens=True
    )

def parler_fon(texte):
    inputs = voice_tokenizer(text=texte, return_tensors="pt")
    with torch.no_grad():
        audio = voice_model(**inputs).waveform.squeeze().cpu().numpy()

    sf.write(AUDIO_FILE, audio, voice_model.config.sampling_rate)
    print(f"\nAudio enregistré : {os.path.abspath(AUDIO_FILE)}")

    # Lecture automatique sous Windows si un lecteur par défaut est disponible.
    try:
        import winsound
        winsound.PlaySound(AUDIO_FILE, winsound.SND_FILENAME)
    except Exception as error:
        print("Lecture automatique indisponible :", error)

print("\nChatbot prêt ! Tape QUITTER pour terminer.")
print("Pose tes questions en français.\n")

while True:
    question = input("Toi : ").strip()

    if question.upper() == "QUITTER":
        print("Fin de la conversation.")
        break

    if not question:
        continue

    history.append({"role": "user", "content": question})

    try:
        result = requests.post(
            OLLAMA_URL,
            json={
                "model": OLLAMA_MODEL,
                "messages": history,
                "stream": False
            },
            timeout=300
        )
        result.raise_for_status()
        answer = result.json()["message"]["content"].strip()

        history.append({"role": "assistant", "content": answer})
        print("\nAssistant (français) :", answer)

        print("\nTraduction en cours...")
        fon = traduire_en_fon(answer)
        print("Assistant (fon) :", fon)

        print("\nGénération de la voix...")
        parler_fon(fon)

    except Exception as error:
        # Retire la question de l'historique si la génération a échoué,
        # afin d'éviter de conserver une conversation incohérente.
        if history and history[-1].get("role") == "user":
            history.pop()
        print(f"\nErreur : {error}")
        print("Tu peux réessayer avec une autre question.\n")
