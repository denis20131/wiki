from transformers import pipeline
from transformers import T5ForConditionalGeneration, T5Tokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
def generate_summary(quiry, model_name):
    torch.manual_seed(3)

    prompt = quiry+"""
    Напиши пересказ этого текста

    Пересказ:

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


def summ(quiry,lang):
    if lang=="en":
        summarizer = pipeline("summarization", model="facebook/bart-large-cnn",device="cuda:0")
        text=summarizer(quiry, max_length= (len(quiry)//4), min_lentgth=(len(quiry)//4), do_sample=False)
        return text[0]['summary_text']
    else:
        model_name = "rndteam41/tsum_trained"
        # tokenizer = AutoTokenizer.from_pretrained(model_name)
        # model = AutoModelForCausalLM.from_pretrained(model_name)
        return generate_summary(quiry,model_name)
        # prefix = 'summary brief: '
        # src_text = prefix + quiry
        # input_ids = tokenizer(src_text, return_tensors="pt")

        # generated_tokens = model.generate(**input_ids)

        # result = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        # return result

