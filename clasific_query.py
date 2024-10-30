
from transformers import pipeline
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

def classify_task(query,model):
    # Задачи для классификации
    tasks = ["3 summary", "1 translation","2 write article","4 extract keywords"]
# Создаем классификационный pipeline на основе предобученной модели
    from transformers import pipeline

    classifier = pipeline("zero-shot-classification",model="MoritzLaurer/mDeBERTa-v3-base-mnli-xnli",device="cuda:0")
    result = classifier(str(query), tasks)
    # print(f"Result:{result}")
    max_score_index = result['scores'].index(max(result['scores']))
    return result['labels'][max_score_index]

# https://huggingface.co/facebook/bart-large-mnli

