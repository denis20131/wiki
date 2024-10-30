import requests
import torch
import sys
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from retriver import process_wiki_data
from additional import *
from clasific_query import *
from langdetect import detect
from summurize import *
from exctraction import *
from translate import *
from write_article import *

# Функция для генерации ответа с использованием модели

    # # Разделение ссылок
    # links = articles_text.split("\n")
    
    # text_summary = ""
    # for link in links:
    #     # Здесь можно добавить код для обработки каждой статьи и создания краткого изложения
    #     # Например, можно использовать дополнительные API или библиотеки для извлечения текста статьи
    #     text_summary += f"Summary of the article from {link}: ключевые моменты... \n"
    
    # if text_summary:
    #     answer = f"На основе статьи ключевые моменты: \n{text_summary}\n\nОтвет на вопрос: ..."
    # else:
    #     answer = "Недостаточно информации для ответа на вопрос."
    
    # return answer

# def generate_response_add(model_name, dialogue_history, search_query):
#     torch.manual_seed(0)
#     prompt = '''Отвечая по предыдущей теме из историй диалога f"{dialogue_history}"выполни запрос
#     Это может быть просьба пересказать основные моменты темы или выделить главные слова из статьи'''
#     pipe = pipeline("text-generation", model=model_name, device="cuda")

#     sequences = pipe(
#         prompt,
#         max_new_tokens=1000,
#         do_sample=True,
#         top_k=10,
#         return_full_text=False,
#     )

    # generated_text = ' '.join([seq['generated_text'] for seq in sequences])

    # return generated_text

def dialogue():
    api_key = "AIzaSyC1AdfloL-KSq253rERj3ZqqrbLeFIZqDE"
    cse_id = "a753502dcc5624aa0"
    
    model_llama = "rndteam41/tsum_trained"  
    tokenizer = AutoTokenizer.from_pretrained(model_llama)
    model = AutoModelForCausalLM.from_pretrained(model_llama)
    model.half()
    device = torch.device("cuda")
    while True:
        print('''
        Введите одну из задач:
            1.перевод
            2.написать статью
            3.пересказ
            4.извлечь слова из текста
        ''')
        class_query = input("User: ")
        if detect(class_query)=="ru":
            process_query=translate_text(class_query,"ru")
        else:
             process_query=class_query
            # Поиск статей через Google Custom Search API
        results = google_search(process_query, api_key, cse_id)
        articles = []
        for item in results.get('items', []):
            articles.append(f"Title: {item['title']}\nURL: {item['link']}\nSnippet: {item['snippet']}\n")
        print(articles)

        #     # Генерация ответа на основе статей
        # response = generate_response(search_query, articles_text, model, tokenizer)
        user_choise=classify_task(process_query,model_llama)

        if (user_choise=="3 summary"):
                print("Пересказываю статью\nВведите текст")
                text=str(input().strip())
                if text=="q": continue
                print(f"Ваш пересказанный текст\n{summ(text,detect(text))}")
                continue

        if (user_choise=="1 translation"):
                print("Перевод\nВведите текст")
                search_text=str(input().strip())
                if search_text=="q": 
                     continue
                language=detect(search_text)
                text=translate_text(search_text,language)
                text = text.rstrip(" by") if text.endswith(" by") else text
                print(f"Ваш переведённый текст:\n{text}")
                continue

        if (user_choise=="4 extract keywords"):
            print("Вывожу ключеввые слова\nВведите текст")
            text=input().strip()
            if text=="q": continue
            print(f"Ключевые слова: {generate_extract(text,model_llama)}")
            continue
        if (user_choise=="2 write article") :
            print("Пишу статью\nВведите текст")
            text=input().strip()
            if text=="q": continue
            continue
        print(user_choise)
# Запуск диалога
if __name__ == "__main__":
    dialogue()
