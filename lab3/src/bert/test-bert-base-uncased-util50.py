# 'eval_accuracy': 0.92932

import torch
import torch.nn as nn
import numpy as np

from transformers import AutoTokenizer, BertModel
from transformers import Trainer, TrainingArguments
from transformers.modeling_outputs import SequenceClassifierOutput

from datasets import load_dataset
import evaluate


# Prepare Dataset
def prepare_dataset(dataset_name, tokenizer, utilize_ratio, train_val_split, seed):

    # Raw dataset load
    dataset = load_dataset(dataset_name)
    dataset = dataset.map(lambda x: tokenizer(x["text"], truncation=True, padding="max_length"), batched=True)

    # Separate the training and validation sets
    raw_train_dataset = dataset["train"].shuffle(seed)
    raw_val_dataset = dataset["test"].shuffle(seed)

    # Calculate the number of examples to include in training and validation sets
    new_examples = int(len(raw_train_dataset) * utilize_ratio)
    num_train_examples = int(new_examples * train_val_split)
    num_val_examples = new_examples - num_train_examples

    # Split the training data into a smaller training set and a validation set
    new_dataset = raw_train_dataset.train_test_split(
        train_size=num_train_examples,
        test_size=num_val_examples,
        shuffle=True,
    )

    # Get new train/val/test dataset
    new_train_dataset = new_dataset["train"]
    new_val_dataset = new_dataset["test"]
    new_test_dataset = raw_val_dataset

    return new_train_dataset, new_val_dataset, new_test_dataset

# Prepare Model
class BERT(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = BertModel.from_pretrained("./results/bert-base-uncased-util50/checkpoint-6000/")
        self.model.init_weights()
        self.classifier = nn.Linear(768, 2)


    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0]
        logits = self.classifier(pooled_output)
        loss = None
        if labels is not None:
            criterion = nn.CrossEntropyLoss()
            loss = criterion(logits.view(-1, 2), labels.view(-1))
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


if __name__ == "__main__":
    
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"

    bert_pretrain_path = "./results/bert-base-uncased-util50/checkpoint-6000/pytorch_model.bin"
    
    bert_ckpt = torch.load(bert_pretrain_path)
    for key in list(bert_ckpt.keys()): # remove the prefix "model."
        if key.startswith("model."):
            new_key = key[6:]
            bert_ckpt[new_key] = bert_ckpt.pop(key)
    torch.save(bert_ckpt, bert_pretrain_path)
    
    for key in list(bert_ckpt.keys()): 
        if not "classifier" in key:
            bert_ckpt.pop(key)
        else:
            new_key = key.split(".")[-1]
            bert_ckpt[new_key] = bert_ckpt.pop(key)
    
    bert_model = BERT()
    print("Begin load pretrain weight of classifier")
    bert_model.classifier.load_state_dict(bert_ckpt)
    bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    print("End load pretrain weight of classifier")
    
    trainset, valset, testset = prepare_dataset(dataset_name="imdb",
                                                tokenizer=bert_tokenizer,
                                                utilize_ratio=1.00,
                                                train_val_split=0.8,
                                                seed=2023)
    
    # import ipdb
    # ipdb.set_trace()
    
    def compute_metrics(eval_pred):
        metric = evaluate.load("accuracy")
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)
    
    # args
    training_args = TrainingArguments(
        logging_steps=20,
        output_dir="./results/test-bert-base-uncased-util50",
        num_train_epochs=5,
        per_device_train_batch_size=16//4,
        per_device_eval_batch_size=32//4,
        optim="adamw_torch",
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_epsilon=1e-8,
        learning_rate=5e-5,
        warmup_steps=200,
        lr_scheduler_type="linear",
        evaluation_strategy="steps",
        eval_steps=100
        )
    
    trainer = Trainer(model=bert_model, 
                      tokenizer=bert_tokenizer, 
                      train_dataset=trainset, 
                      eval_dataset=valset, 
                      args=training_args, 
                      compute_metrics=compute_metrics)
    
    out = trainer.evaluate(testset)
    print(out)