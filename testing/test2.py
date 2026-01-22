from utlis.config_loader import load_config


config = load_config()

collection_name = config["vector_db"]["index_name"]
embedding_model_name = config["embedding_model"]["model_name"]
llm_name=config["llm"]["model"]
top_k = config["retriever"]["top_k"]

print(collection_name)          # test
print(embedding_model_name)     # BAAI/bge-large-en-v1.5
print(llm_name)
print(top_k)                    # 3