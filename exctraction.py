
from transformers import pipeline
from transformers import T5ForConditionalGeneration, T5Tokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
def generate_extract(quiry, model_name):
    torch.manual_seed(3)
    prompt = "Текст:"+ quiry+"""Выведи только НАУЧНЫЕ термины из текста или НАУЧНЫЕ ключевые слова нужно выдавать только научные слова из текста, обычные повседневные не связанные с темой не надо.Ответ дай в форме списка
с названием.Всё что ты выводишь должно быть из моего текста все слова найденные должны быть от сюда"""+quiry+""""
    Термины: список слов (каждое слово с новой строки) 
    """
    pipe = pipeline(

    "text-generation",

    model=model_name,

    tokenizer=AutoTokenizer.from_pretrained(model_name),

    torch_dtype=torch.bfloat16,

    device_map="auto",

    )
    sequences = pipe(

    prompt,

    max_new_tokens=100,

    do_sample=True,

    top_k=10,

    return_full_text = False,

)
    a=[]
    for seq in sequences:
        a.append(seq['generated_text'])
    return ''.join(a)