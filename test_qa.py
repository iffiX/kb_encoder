import torch as t
from encoder.trainer.openbook_qa_trainer import OpenBookQATrainer

if __name__ == "__main__":
    trainer = OpenBookQATrainer.load_from_checkpoint(
        # "/home/muhan/data/workspace/kb_encoder/train_ob_qa/0/checkpoint/epoch=14-test_accuracy-test_accuracy=0.850.ckpt"
        "/home/muhan/data/workspace/kb_encoder/train_ob_qa_deberta_v3_match/0/checkpoint/epoch=16-test_accuracy-test_accuracy=0.862.ckpt"
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
        if trainer.config.base_type.startswith("t5"):
            out = trainer.model.generate(
                encoding["input_ids"],
                max_length=32,
                attention_mask=encoding["attention_mask"],
                early_stopping=True,
            )
            print("Result:")
            print(trainer.tokenizer.decode(out[0].to("cpu"), skip_special_tokens=True))
        else:
            choice_num = trainer.model.choice_num
            out = trainer.model.predict(
                input_ids=encoding["input_ids"]
                .unsqueeze(0)
                .repeat(1, choice_num, 1)
                .to("cuda:0"),
                attention_mask=encoding["attention_mask"]
                .unsqueeze(0)
                .repeat(1, choice_num, 1)
                .to("cuda:0"),
                token_type_ids=encoding["token_type_ids"]
                .unsqueeze(0)
                .repeat(1, choice_num, 1)
                .to("cuda:0"),
            ).cpu()
            print("Result:")
            print(t.sigmoid(out[0, 0].to("cpu")).item())
