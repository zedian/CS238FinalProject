import openai
import pandas as pd
import os
from config import OAI_API_KEY
from concurrent.futures import ThreadPoolExecutor

openai.api_key = OAI_API_KEY

def get_embedding(dialogue):
    try:
        response = openai.Embedding.create(input=dialogue, engine="text-embedding-ada-002")
        return response['data'][0]['embedding']
    except Exception as e:
        print(f"Error getting embedding for dialogue: {dialogue[:30]}... Error: {e}")
        return None

def get_embeddings_concurrent(dialogues, output_file, max_workers=15):
    if os.path.exists(output_file):
        print(f"Reading embeddings from {output_file}")
        embeddings_df = pd.read_csv(output_file)
        embeddings = embeddings_df.drop(columns=['index']).values.tolist()
        indices = embeddings_df['index'].tolist()
    else:
        print(f"Generating embeddings for {len(dialogues)} dialogues")
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            embeddings = list(executor.map(get_embedding, dialogues))
        embeddings_df = pd.DataFrame(embeddings)
        embeddings_df['index'] = range(len(dialogues))
        embeddings_df.to_csv(output_file, index=False)
        indices = embeddings_df['index'].tolist()
    return embeddings, indices

def main():
    print("Starting script")
    df = pd.read_csv('output_dialogues_all.csv')
    dialogues = df['dialogue'].tolist()
    embeddings, indices = get_embeddings_concurrent(dialogues, 'embeddings.csv')
    print("Embeddings generated")

if __name__ == "__main__":
    main()
