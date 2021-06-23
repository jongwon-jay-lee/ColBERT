import argparse

from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch

QA_BERT_LARGE_UNCASED = "bert-large-uncased-whole-word-masking-finetuned-squad"


def load_model(args):
    if not hasattr(args, "model_type_or_path"):
        args.model_type_or_path = QA_BERT_LARGE_UNCASED
    tokenizer = AutoTokenizer.from_pretrained(args.model_type_or_path)
    model = AutoModelForQuestionAnswering.from_pretrained(args.model_type_or_path)

    return model, tokenizer


def get_qa():
    text = r"""
    🤗 Transformers (formerly known as pytorch-transformers and pytorch-pretrained-bert) provides general-purpose
    architectures (BERT, GPT-2, RoBERTa, XLM, DistilBert, XLNet…) for Natural Language Understanding (NLU) and Natural
    Language Generation (NLG) with over 32+ pretrained models in 100+ languages and deep interoperability between
    TensorFlow 2.0 and PyTorch.
    """

    questions = [
        "How many pretrained models are available in 🤗 Transformers?",
        "What does 🤗 Transformers provide?",
        "🤗 Transformers provides interoperability between which frameworks?",
    ]
    return text, questions


def extractive_answer(question, text, model, tokenizer):

    inputs = tokenizer(question, text, add_special_tokens=True, return_tensors="pt")
    input_ids = inputs["input_ids"].tolist()[0]

    outputs = model(**inputs)
    answer_start_scores = outputs.start_logits
    answer_end_scores = outputs.end_logits

    answer_start = torch.argmax(answer_start_scores)
    answer_end = torch.argmax(answer_end_scores) + 1

    answer = tokenizer.convert_tokens_to_string(
        tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end])
    )
    print(f"Question: {question}")
    print(f"Context: {text}")
    print(f"Answer: {answer}")


def main():
    parser = argparse.ArgumentParser()
    """
    parser.add_argument(
        "--model_type_or_path",
        type=str,
        default=QA_BERT_LARGE_UNCASED,
    )
    """
    args = parser.parse_args()

    model, tokenizer = load_model(args)
    text, questions = get_qa()

    for question in questions:
        extractive_answer(question, text, model, tokenizer)
        print("-" * 50)


if __name__ == "__main__":
    main()
