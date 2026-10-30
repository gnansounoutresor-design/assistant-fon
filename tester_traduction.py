from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

MODEL = "nellaw/nllb-fongbe-french-translation"

print("Chargement du modèle de traduction...")
tokenizer = AutoTokenizer.from_pretrained(
    MODEL,
    src_lang="fra_Latn",
    tgt_lang="fon_Latn"
)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL)

texte = "Bonjour, comment allez-vous ?"
inputs = tokenizer(texte, return_tensors="pt", truncation=True, max_length=128)

print("Traduction en cours...")
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=128,
        num_beams=3,
        forced_bos_token_id=tokenizer.convert_tokens_to_ids("fon_Latn")
    )

print("\nFrançais :", texte)
print("Fon      :", tokenizer.decode(outputs[0], skip_special_tokens=True))
