# from transformers import AutoTokenizer, AutoModelForCausalLM

# model_name = "rndteam41/tsum_trained"  # Укажите путь к вашей модели
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForCausalLM.from_pretrained(model_name)
# model=model.cuda()
# conversation = [
#     {"role": "user", "content": "What has Man always dreamed of?"}
# ]

# # Define documents for retrieval-based generation
# documents = [
#     {
#         "title": "The Moon: Our Age-Old Foe", 
#         "text": "Man has always dreamed of destroying the moon. In this essay, I shall..."
#     },
#     {
#         "title": "The Sun: Our Age-Old Friend",
#         "text": "Although often underappreciated, the sun provides several notable benefits..."
#     }
# ]
# tokenizer.pad_token_id = tokenizer.eos_token_id
# # Tokenize conversation and documents using a RAG template, returning PyTorch tensors.
# input_ids = tokenizer.apply_chat_template(
#     conversation=conversation,
#     documents=documents,
#     chat_template="rag",
#     tokenize=True,
#     add_generation_prompt=True,
#     return_tensors="pt").to("cuda")

# # Generate a response 
# gen_tokens = model.generate(
#     input_ids,
#     max_new_tokens=100,
#     do_sample=True,
#     temperature=0.3,
#     )

# # Decode and print the generated text along with generation prompt
# gen_text = tokenizer.decode(gen_tokens[0])
# print(gen_text)
from googletrans import Translator
def tranl_text(text):
    translator = Translator()
    print(
     '''
af: африкаанс
ar: арабский
bn: бенгальский
bg: болгарский
cs: чешский
da: датский
de: немецкий
el: греческий
en: английский
es: испанский
et: эстонский
fa: фарси (персидский)
fi: финский
fr: французский
gu: гуджарати
he: иврит
hi: хинди
hr: хорватский
hu: венгерский
id: индонезийский
it: итальянский
ja: японский
ka: грузинский
kk: казахский
ko: корейский
lt: литовский
lv: латышский
ms: малайский
nl: нидерландский
no: норвежский
pl: польский
pt: португальский
ro: румынский
ru: русский
sk: словацкий
sl: словенский
sr: сербский
sv: шведский
ta: тамильский
te: телугу
th: тайский
tr: турецкий
uk: украинский
ur: урду
vi: вьетнамский
zh-cn: китайский (упрощённый)
zh-tw: китайский (традиционный)
    '''
    )
    s=str(input().strip())
    translated = translator.translate(text, src='ru', dest='en')
    # print("Переведенный текст:", translated.text)
    return translated.text

from rake_nltk import Rake
from stop_words import get_stop_words
import nltk


from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from collections import defaultdict
import nltk
from keybert import KeyBERT
def rake_keyword_extraction(text):
    kw_model = KeyBERT()
    text=tranl_text(text)
    print(text)
    keywords =  kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 1), stop_words='english',
                              use_maxsum=True, nr_candidates=20, top_n=5)
    return keywords
if __name__ == "__main__":
    print(rake_keyword_extraction(
        '''
Большие языковые модели (также называемые LLM) – это очень большие модели глубокого обучения, которые предварительно обучены на огромных объемах данных. Лежащий в основе трансформер – это набор нейронных сетей, каждая из которых состоит из кодера и декодера с возможностью самонаблюдения. Кодер и декодер извлекают значения из последовательности текста и понимают отношения между имеющимися в ней словами и фразами.
Трансформеры LLM способны обучаться без наблюдения, хотя точнее будет сказать, что трансформеры осуществляют самообучение. Именно благодаря этому процессу трансформеры учатся понимать базовую грамматику и языки, а также усваивать знания.
В отличие от предыдущих рекуррентных нейронных сетей (RNN), которые последовательно обрабатывают входные данные, трансформеры обрабатывают целые последовательности параллельно. Это позволяет специалистам по обработке данных использовать графические процессоры для обучения LLM на основе трансформеров, что значительно сокращает время обучения.
'''
    ))