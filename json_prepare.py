from datasets import load_dataset
import pandas as pd

# Загрузка вашего датасета с Hugging Face
dataset = load_dataset("wikimedia/wikipedia", "20231101.ru")  # замените на название вашего датасета
subset_dataset = dataset['train']
df = pd.DataFrame(subset_dataset)  

df.to_json("subset_dataset.json", orient="records", lines=True, force_ascii=False)

print(f"Подмножество датасета сохранено в файл subset_dataset.json ")
