from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, BitsAndBytesConfig
from datasets import load_dataset
from peft import get_peft_model, LoraConfig, TaskType
import torch

# Укажите путь к вашей модели
model_name = "rndteam41/tsum_trained"   
tokenizer = AutoTokenizer.from_pretrained(model_name)
device = torch.device("cpu")
# Загружаем модель с квантованием, но переместим её на CPU
# quantization_config = BitsAndBytesConfig(load_in_8bit=True)
model = AutoModelForCausalLM.from_pretrained(model_name, pad_token_id=tokenizer.eos_token_id).to("cpu")
model.cpu()
# Настройка целевых модулей
target_modules = ["q_proj", "k_proj", "v_proj", "out_proj", "fc_in", "fc_out", "wte"]

# Конфигурация LoRA
peft_config = LoraConfig(
    r=4, 
    lora_alpha=16, 
    target_modules=target_modules,
    lora_dropout=0.1, 
    bias="none", 
    task_type="CAUSAL_LM"
)

# Получение LoRA модели и перемещение на CPU
lora_model = get_peft_model(model, peft_config)
lora_model.cpu()
lora_model.print_trainable_parameters()

# Загрузка датасета
dataset = load_dataset("wikimedia/wikipedia", "20231101.en")
dataset = dataset.remove_columns(['id', 'url'])

# Получение общего количества примеров в тренировочном датасете
total_samples = len(dataset['train'])
fraction = 0.00001
num_samples = int(total_samples * fraction)
subset_dataset = dataset['train'].select(range(num_samples))

# Токенизация
def tokenize_function(example):
    return tokenizer(example['text'], truncation=True, padding='max_length', max_length=2048)

tokenized_datasets = subset_dataset.map(tokenize_function, batched=True)

# Переместите данные на CPU
tokenized_datasets.set_format(type='torch', columns=['title', 'text', 'input_ids', 'attention_mask'])

# Настройки обучения
training_args = TrainingArguments(
    output_dir="./results",
    overwrite_output_dir=True,
    per_device_train_batch_size=8,
    num_train_epochs=3,
    save_total_limit=2,
    fp16=False,  # Отключите fp16, если используете CPU
    gradient_accumulation_steps=1,  
    dataloader_num_workers=4,  
    logging_dir='./logs',  
    logging_steps=10,  
    evaluation_strategy="epoch",  
    save_strategy="epoch",  
    load_best_model_at_end=True,  
    report_to="wandb",  
    use_cpu=True
)

# Настройка тренера и обучение на CPU
trainer = Trainer(
    model=lora_model,
    args=training_args,
    train_dataset=tokenized_datasets,
)

# Начало обучения
trainer.train()

# Сохранение модели и токенизатора
lora_model.save_pretrained("./results/final_model")
tokenizer.save_pretrained("./results/final_model")
