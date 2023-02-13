import torch
import torchvision
import transformers
import evaluate
from datasets import load_dataset, load_metric
from transformers import AutoTokenizer
import numpy as np
from transformers import create_optimizer, AdamWeightDecay
from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer
import os
from transformers import DataCollatorForSeq2Seq
import nltk
nltk.download('punkt')


def preprocess_data(data_to_process):
  #get all the dialogues
  inputs = [dialogue for dialogue in data_to_process['dialogue']]
  #tokenize the dialogues
  model_inputs = tokenizer(inputs,  max_length=max_input, padding='max_length', truncation=True)
  #tokenize the summaries
  with tokenizer.as_target_tokenizer():
    targets = tokenizer(data_to_process['summary'], max_length=max_target, padding='max_length', truncation=True)
    
  #set labels
  model_inputs['labels'] = targets['input_ids']
  #return the tokenized data
  #input_ids, attention_mask and labels
  return model_inputs


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)

    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
    result["gen_len"] = np.mean(prediction_lens)

    return {k: round(v, 4) for k, v in result.items()}
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    from huggingface_hub import notebook_login
    notebook_login()
    dataset = load_dataset('knkarthick/dialogsum') 
    metric = load_metric("rouge")
    max_input = 512
    max_target = 128
    batch_size = 3
    model_checkpoints = "facebook/bart-large-xsum"
    dataset = load_dataset('knkarthick/dialogsum') 
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoints)
    tokenized_data = dataset.map(preprocess_data, batched=True)
    model = transformers.AutoModelForSeq2SeqLM.from_pretrained(model_checkpoints)
    model.to('cuda')
    collator = transformers.DataCollatorForSeq2Seq(tokenizer, model=model)
    rouge = evaluate.load("rouge")

    training_args = Seq2SeqTrainingArguments(
        output_dir="finetuned_bart_on_dialogsum",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=2,
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=4,
        predict_with_generate=True,
        eval_accumulation_steps=1,
        fp16=True,
        push_to_hub=True,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_data["train"],
        eval_dataset=tokenized_data["validation"],
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics,
    )
    trainer.train()
