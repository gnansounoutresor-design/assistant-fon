import os
import re
import uuid
import threading
import logging
from pathlib import Path

import requests
import torch
from flask import Flask, jsonify, render_template, request, send_from_directory
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, VitsModel

BASE_DIR = Path(__file__).resolve().parent
AUDIO_DIR = BASE_DIR / "static" / "audio"
AUDIO_DIR.mkdir(parents=True, exist_ok=True)

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen:latest")
TRANSLATION_MODEL = "nellaw/nllb-fongbe-french-translation"
VOICE_MODEL = "facebook/mms-tts-fon"

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("assistant_fon")

# Les modèles sont chargés à la demande, puis conservés en mémoire.
translator_tokenizer = None
translator = None
voice_tokenizer = None
voice_model = None

translation_lock = threading.Lock()
voice_lock = threading.Lock()
history_lock = threading.Lock()

# Historique temporaire en mémoire : un historique par conversation.
conversations = {}
MAX_HISTORY_MESSAGES = 20
MAX_CONVERSATIONS = 100

SYSTEM_PROMPT = (
    "Tu es un assistant conversationnel utile, patient et honnête. "
    "L'utilisateur parle français. Réponds en français, clairement et "
    "naturellement. Tiens compte des messages précédents. N'invente pas "
    "de faits. Si une question est ambiguë, demande une précision. "
    "Réponds de manière concise quand c'est possible. "
    "Pour les calculs simples, donne un résultat exact."
)


def charger_traducteur():
    global translator_tokenizer, translator

    if translator is not None and translator_tokenizer is not None:
        return

    with translation_lock:
        if translator is not None and translator_tokenizer is not None:
            return

        logger.info("Chargement du modèle de traduction...")
        tokenizer = AutoTokenizer.from_pretrained(
            TRANSLATION_MODEL,
            src_lang="fra_Latn",
            tgt_lang="fon_Latn"
        )
        model = AutoModelForSeq2SeqLM.from_pretrained(TRANSLATION_MODEL)
        model.eval()

        translator_tokenizer = tokenizer
        translator = model
        logger.info("Traducteur prêt.")


def charger_voix():
    global voice_tokenizer, voice_model

    if voice_model is not None and voice_tokenizer is not None:
        return

    with voice_lock:
        if voice_model is not None and voice_tokenizer is not None:
            return

        logger.info("Chargement du modèle vocal...")
        tokenizer = AutoTokenizer.from_pretrained(VOICE_MODEL)
        model = VitsModel.from_pretrained(VOICE_MODEL)
        model.eval()

        voice_tokenizer = tokenizer
        voice_model = model
        logger.info("Modèle vocal prêt.")


def calcul_simple(question):
    """
    Calcule seulement une expression arithmétique explicite.
    Aucun eval() n'est utilisé.
    """
    texte = question.strip().lower()

    motifs = [
        r"combien\s+font\s+(-?\d+(?:[.,]\d+)?)\s*([+\-*/×÷])\s*(-?\d+(?:[.,]\d+)?)",
        r"combien\s+fait\s+(-?\d+(?:[.,]\d+)?)\s*([+\-*/×÷])\s*(-?\d+(?:[.,]\d+)?)",
        r"calcule(?:r)?\s+(-?\d+(?:[.,]\d+)?)\s*([+\-*/×÷])\s*(-?\d+(?:[.,]\d+)?)",
    ]

    for motif in motifs:
        match = re.fullmatch(motif, texte)
        if not match:
            continue

        a = float(match.group(1).replace(",", "."))
        op = match.group(2)
        b = float(match.group(3).replace(",", "."))

        if abs(a) > 1e12 or abs(b) > 1e12:
            return None

        try:
            if op == "+":
                resultat = a + b
            elif op == "-":
                resultat = a - b
            elif op in ("*", "×"):
                resultat = a * b
            elif op in ("/", "÷"):
                if b == 0:
                    return "La division par zéro est impossible."
                resultat = a / b
            else:
                return None

            if not (-1e15 < resultat < 1e15):
                return None

            if float(resultat).is_integer():
                affichage = str(int(resultat))
            else:
                affichage = f"{resultat:.10g}".replace(".", ",")

            return f"Le résultat est {affichage}."
        except (ValueError, OverflowError, ZeroDivisionError):
            return None

    return None


def interroger_ollama(messages):
    response = requests.post(
        f"{OLLAMA_URL}/api/chat",
        json={
            "model": OLLAMA_MODEL,
            "messages": messages,
            "stream": False
        },
        timeout=(10, 240)
    )
    response.raise_for_status()

    data = response.json()
    answer = data.get("message", {}).get("content", "")

    if not isinstance(answer, str) or not answer.strip():
        raise RuntimeError("Le modèle n'a fourni aucune réponse.")

    return answer.strip()


def traduire_en_fon(texte):
    charger_traducteur()

    with translation_lock:
        inputs = translator_tokenizer(
            texte,
            return_tensors="pt",
            truncation=True,
            max_length=256
        )

        target_id = translator_tokenizer.convert_tokens_to_ids("fon_Latn")
        if target_id is None:
            raise RuntimeError("Le code de langue Fon est indisponible.")

        with torch.inference_mode():
            outputs = translator.generate(
                **inputs,
                max_new_tokens=180,
                num_beams=3,
                forced_bos_token_id=target_id
            )

        resultat = translator_tokenizer.decode(
            outputs[0],
            skip_special_tokens=True
        ).strip()

    if not resultat:
        raise RuntimeError("La traduction est vide.")

    return resultat


def generer_audio(texte_fon):
    """
    La voix est facultative : une erreur ici ne doit jamais supprimer
    la réponse textuelle.
    """
    if not texte_fon or not texte_fon.strip():
        raise ValueError("Aucun texte Fon à prononcer.")

    charger_voix()

    with voice_lock:
        inputs = voice_tokenizer(
            text=texte_fon.strip(),
            return_tensors="pt"
        )

        if inputs.get("input_ids") is None:
            raise RuntimeError("Impossible de préparer le texte pour la voix.")

        # Certaines phrases très courtes, notamment des nombres seuls,
        # peuvent provoquer une erreur dans le modèle vocal.
        if inputs["input_ids"].shape[-1] < 2:
            raise RuntimeError(
                "Texte trop court pour la synthèse vocale. "
                "La réponse textuelle reste disponible."
            )

        with torch.inference_mode():
            output = voice_model(**inputs)
            audio = output.waveform.squeeze().cpu().numpy()

        if audio.size == 0:
            raise RuntimeError("Le modèle vocal a produit un audio vide.")

        import soundfile as sf

        filename = f"{uuid.uuid4().hex}.wav"
        destination = AUDIO_DIR / filename
        sf.write(
            str(destination),
            audio,
            voice_model.config.sampling_rate
        )

    return f"/static/audio/{filename}"


@app.get("/")
def accueil():
    return render_template("index.html")


@app.get("/api/health")
def health():
    result = {
        "site": "ok",
        "ollama": "indisponible",
        "modele_ollama": OLLAMA_MODEL,
        "traducteur": "non chargé",
        "voix": "non chargée"
    }

    try:
        response = requests.get(f"{OLLAMA_URL}/api/tags", timeout=3)
        response.raise_for_status()
        names = [
            item.get("name", "")
            for item in response.json().get("models", [])
        ]
        result["ollama"] = (
            "ok" if OLLAMA_MODEL in names else "modèle absent"
        )
    except Exception:
        pass

    if translator is not None:
        result["traducteur"] = "chargé"

    if voice_model is not None:
        result["voix"] = "chargée"

    return jsonify(result)


@app.post("/api/chat")
def chat():
    data = request.get_json(silent=True)

    if not isinstance(data, dict):
        return jsonify({"error": "Requête invalide."}), 400

    question = data.get("question")
    conversation_id = data.get("conversation_id")

    if not isinstance(question, str):
        return jsonify({"error": "La question doit être du texte."}), 400

    question = question.strip()

    if not question:
        return jsonify({"error": "Écris une question avant d'envoyer."}), 400

    if len(question) > 4000:
        return jsonify({"error": "La question est trop longue (maximum 4 000 caractères)."}), 413

    if not isinstance(conversation_id, str) or len(conversation_id) > 80:
        conversation_id = uuid.uuid4().hex

    warnings = []
    audio_url = None

    # Réponse arithmétique déterministe pour les formats reconnus.
    answer_fr = calcul_simple(question)

    with history_lock:
        if conversation_id not in conversations:
            if len(conversations) >= MAX_CONVERSATIONS:
                oldest_key = next(iter(conversations))
                conversations.pop(oldest_key, None)

            conversations[conversation_id] = [
                {"role": "system", "content": SYSTEM_PROMPT}
            ]

        history = conversations[conversation_id]
        messages = list(history)

    if answer_fr is None:
        try:
            messages.append({"role": "user", "content": question})
            answer_fr = interroger_ollama(messages)
        except requests.exceptions.Timeout:
            return jsonify({
                "error": "Ollama met trop de temps à répondre. Réessaie."
            }), 504
        except requests.exceptions.ConnectionError:
            return jsonify({
                "error": "Ollama est inaccessible. Vérifie qu'il est lancé."
            }), 503
        except Exception:
            logger.exception("Erreur pendant la génération de la réponse")
            return jsonify({
                "error": "La réponse n'a pas pu être générée. Réessaie."
            }), 502

    # Enregistrer la question et la réponse française uniquement après
    # la réussite de la génération. La traduction et la voix sont séparées.
    with history_lock:
        history = conversations.setdefault(
            conversation_id,
            [{"role": "system", "content": SYSTEM_PROMPT}]
        )
        history.extend([
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer_fr}
        ])

        # Garder le message système et les derniers échanges.
        conversations[conversation_id] = (
            [history[0]] + history[1:][-MAX_HISTORY_MESSAGES:]
        )

    answer_fon = None

    try:
        answer_fon = traduire_en_fon(answer_fr)
    except Exception:
        logger.exception("Erreur de traduction")
        warnings.append(
            "La traduction en Fon est momentanément indisponible. "
            "La réponse française est conservée."
        )

    if answer_fon:
        try:
            audio_url = generer_audio(answer_fon)
        except Exception:
            logger.exception("Erreur de synthèse vocale")
            warnings.append(
                "La voix n'a pas pu être générée pour cette réponse. "
                "Tu peux quand même lire le texte Fon."
            )

    return jsonify({
        "conversation_id": conversation_id,
        "answer_fr": answer_fr,
        "answer_fon": answer_fon,
        "audio_url": audio_url,
        "warnings": warnings
    })


@app.errorhandler(413)
def fichier_trop_grand(_error):
    return jsonify({"error": "La requête est trop volumineuse."}), 413


@app.errorhandler(500)
def erreur_interne(_error):
    logger.exception("Erreur interne du serveur")
    return jsonify({
        "error": "Une erreur interne est survenue. Réessaie."
    }), 500


if __name__ == "__main__":
    print("\n=== ASSISTANT FRANÇAIS–FON ===")
    print(f"Modèle conversationnel : {OLLAMA_MODEL}")
    print("Site local : http://127.0.0.1:5000")
    print("Les modèles de traduction et de voix seront chargés à la demande.\n")
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)
