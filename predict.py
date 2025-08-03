from transformers import RobertaTokenizer, RobertaForSequenceClassification
import torch

model = RobertaForSequenceClassification.from_pretrained("results/checkpoint-XXXX")  # Replace with real checkpoint
tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")

code = """
const A = "AIza";
const B = "SyD9-example";
const key = A + B;
fetch("https://api.example.com?key=" + key);
"""

inputs = tokenizer(code, return_tensors="pt", truncation=True, padding=True, max_length=512)
outputs = model(**inputs)
pred = torch.argmax(outputs.logits, dim=1).item()

print("🔴 Leak Detected!" if pred == 1 else "🟢 Code is Clean.")
