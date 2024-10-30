import requests
from transformers import pipeline
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

def translate_text(text,source_lang):
    answer=[]
    model = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_1.2B")
    tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_1.2B")
    tokenizer.src_lang = source_lang
    if source_lang=="ru":
        result = split_into_pairs(text)
        for i in result:
            encoded_ = tokenizer(str(i), return_tensors="pt")
            generated_tokens = model.generate(**encoded_, forced_bos_token_id=tokenizer.get_lang_id("en"))
            answer.append(tokenizer.batch_decode(generated_tokens, skip_special_tokens=True))
    else:
        for i in result:
            encoded_ = tokenizer(str(i), return_tensors="pt")
            generated_tokens = model.generate(**encoded_, forced_bos_token_id=tokenizer.get_lang_id("en"))
            answer.append(tokenizer.batch_decode(generated_tokens, skip_special_tokens=True))    
    if isinstance(answer, list):
        flattened = []
        for item in answer:
            if isinstance(item, list):
                flattened.append(' '.join(item))  
            elif isinstance(item, str):
                flattened.append(item)
        return ''.join(flattened)  #
    else:
        return answer

import re

def split_into_pairs(text):
    sentences = re.split(r'(?<=[.!?]) +', text)
    
    # Группируем предложения по 3
    paired_sentences = [sentences[i:i + 3] for i in range(0, len(sentences), 3)]
    
    result = [' '.join(pair) for pair in paired_sentences]
    
    return result