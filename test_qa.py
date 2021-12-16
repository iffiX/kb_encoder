from encoder.trainer.openbook_qa_trainer import OpenBookQATrainer

if __name__ == "__main__":
    trainer = OpenBookQATrainer.load_from_checkpoint(
        "/home/muhan/data/workspace/kb_encoder/train_ob_qa/0/checkpoint/epoch=07-accuracy-accuracy=0.892.ckpt"
    )
    while True:
        question = input("Q?")
        answer = input("A?")
        trainer.model.to("cuda:0")
        encoding = trainer.tokenizer(
            question,
            answer,
            padding="max_length",
            max_length=256,
            truncation=True,
            return_tensors="pt",
        ).to("cuda:0")
        out = trainer.model.generate(
            encoding["input_ids"],
            max_length=32,
            attention_mask=encoding["attention_mask"],
            early_stopping=True,
        )
        print("Result:")
        print(trainer.tokenizer.decode(out[0].to("cpu"), skip_special_tokens=True))
