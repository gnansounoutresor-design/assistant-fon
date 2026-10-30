import torch
import soundfile as sf
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import VitsModel

TRANSLATION_MODEL = "nellaw/nllb-fongbe-french-translation"
VOICE_MODEL = "facebook/mms-tts-fon"

print("Chargement du traducteur français-Fon...")
print("Le premier téléchargement peut prendre du temps.")

translator_tokenizer = AutoTokenizer.from_pretrained(
    TRANSLATION_MODEL,
    src_lang="fra_Latn",
    tgt_lang="fon_Latn"
)
translator = AutoModelForSeq2SeqLM.from_pretrained(TRANSLATION_MODEL)
translator.eval()

print("\nChargement de la voix Fon...")
voice_tokenizer = AutoTokenizer.from_pretrained(VOICE_MODEL)
voice_model = VitsModel.from_pretrained(VOICE_MODEL)
voice_model.eval()

while True:
    texte_fr = input("\nÉcris un texte en français (ou tape QUITTER) : ").strip()

    if texte_fr.upper() == "QUITTER":
        break

    if not texte_fr:
        continue

    print("\nTraduction en Fon...")
    inputs = translator_tokenizer(
        texte_fr,
        return_tensors="pt",
        max_length=256,
        truncation=True
    )

    with torch.no_grad():
        translated = translator.generate(
            **inputs,
            max_length=256,
            num_beams=4,
            forced_bos_token_id=translator_tokenizer.convert_tokens_to_ids("fon_Latn")
        )

    texte_fon = translator_tokenizer.decode(
        translated[0],
        skip_special_tokens=True
    )

    print("Traduction Fon :", texte_fon)
    print("\nGénération du son...")

    audio_inputs = voice_tokenizer(text=texte_fon, return_tensors="pt")

    with torch.no_grad():
        audio = voice_model(**audio_inputs).waveform.squeeze().cpu().numpy()

    sf.write("reponse_fon.wav", audio, voice_model.config.sampling_rate)

    print("Audio créé : reponse_fon.wav")
    print("Ouvre le fichier pour écouter la prononciation.")
