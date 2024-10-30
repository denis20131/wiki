import pandas as pd,torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModel
def process_wiki_data(question):
    # Загрузка JSON файла с несколькими объектами на отдельных строках
    # df = pd.read_json("subset_dataset.json", lines=True)

    # # Удаление колонок 'id' и 'url'
    # df = df.drop(columns=["id"])
    # comments_dataset = Dataset.from_pandas(df)
    # print(comments_dataset)

    model_ckpt = "sentence-transformers/multi-qa-mpnet-base-dot-v1"
    tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
    model = AutoModel.from_pretrained(model_ckpt)

    device = torch.device("cuda")
    model.to(device)

    def cls_pooling(model_output):
        return model_output.last_hidden_state[:, 0]

    def get_embeddings(text_list):
        encoded_input = tokenizer(
            text_list, padding=True, truncation=True, return_tensors="pt"
        )
        encoded_input = {k: v.to(device) for k, v in encoded_input.items()}
        model_output = model(**encoded_input)
        return cls_pooling(model_output)

    # embedding = get_embeddings(comments_dataset["text"][0])

    # embeddings_dataset = comments_dataset.map(
    #     lambda x: {"embeddings": get_embeddings(x["text"]).detach().cpu().numpy()[0]}
    # )
    # #embeddings_dataset.save_to_disk("saved_embeddings_dataset")
    embeddings_dataset = Dataset.load_from_disk("saved_embeddings_dataset")
    embeddings_dataset.add_faiss_index(column="embeddings")

    question_embedding = get_embeddings([question]).cpu().detach().numpy()

    scores, samples = embeddings_dataset.get_nearest_examples(
        "embeddings", question_embedding, k=5
    )

    samples_df = pd.DataFrame.from_dict(samples)
    samples_df["scores"] = scores
    samples_df.sort_values("scores", ascending=False, inplace=True)
    return samples_df
    
# print(process_wiki_data("что таоке LLM?"))