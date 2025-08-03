from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments
from dataset import BugDataset

tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
model = RobertaForSequenceClassification.from_pretrained("microsoft/codebert-base", num_labels=2)

train_dataset = BugDataset("train.jsonl", tokenizer)
valid_dataset = BugDataset("valid.jsonl", tokenizer)

training_args = TrainingArguments(
    output_dir=".",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=4,
    weight_decay=0.01,
    logging_dir='./logs',
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
)

trainer.train()
