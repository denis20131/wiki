from transformers import pipeline
from transformers import T5ForConditionalGeneration, T5Tokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch,requests

def generate_write_article(quiry, model_name):
    torch.manual_seed(3)
    prompt =  """Ответьте на вопрос, используя приведенный ниже контекст.
    Контекст: Гаспачо — это холодный суп и напиток из сырых смешанных овощей. 
    В состав большинства гаспачо входит черствый хлеб, помидоры, огурцы, лук, сладкий перец, чеснок, оливковое масло, винный уксус, вода и соль. 
    Северные рецепты часто включают тмин и/или пиментон (копченую сладкую паприку). 
    Традиционно гаспачо готовили путем растирания овощей в ступке пестиком; 
    этот более трудоемкий метод до сих пор иногда используется, поскольку он помогает сохранить гаспачо прохладным и позволяет избежать пены и 
    шелковистой консистенции смузи, приготовленных в блендере или кухонном комбайне.
    Текст: Какой современный инструмент используется для приготовления гаспачо?
    Отвечать:
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



# Функция для выполнения поиска через Google Custom Search API
def google_search(query, api_key, cse_id, num_results=5):
    search_url = "https://www.googleapis.com/customsearch/v1"
    params = {
        "q": query,
        "key": api_key,
        "cx": cse_id,
        "num": num_results,
    }
    try:
        response = requests.get(search_url, params=params, timeout=10)  # Добавлен тайм-аут
        response.raise_for_status()  # Проверка на ошибки HTTP
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error during Google search: {e}")
        return {}

def generate_response_main_article(model_name, dialogue_history, search_query, articles):
    # extract_from_wikidata=process_wiki_data(search_query)
    # print (f"from json{extract_from_wikidata}")
    torch.manual_seed(0)
    art=articles[0].split('Snippet: ')[1].split('\n')[0]
    art_url=articles[0].split('URL: ')[1].split('\n')[0]
    prompt = f"""
    Правило: Всегда говори на русском языке. Должен быть только русский язык.
    История диалога:\n{dialogue_history}
    Вы эксперт по обобщению. Всегда переводите информацию на русский.
    {"При ответе опирайтесь на эту информацию: " + art}
    Когда просят дописать или дополнить текст, делайте это, добавляя заголовки и абзацы.
    На основе о '{search_query}', пожалуйста, предоставьте полноценную статью в формате на русском языке.
    В конце выведите одну ссылку: {art_url} 
    """


    pipe = pipeline("text-generation", model=model_name, device="cuda")

    sequences = pipe(
        prompt,
        max_new_tokens=500,
        do_sample=True,
        top_k=10,
        return_full_text=False,
    )

    generated_text = ' '.join([seq['generated_text'] for seq in sequences])

    return generated_text