import os
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq
import numpy as np
from sentence_transformers import SentenceTransformer


model_embedding=SentenceTransformer("all-MiniLM-L6-v2") #384

load_dotenv()

my_api_key=os.getenv("GROQ_API_KEY")

if not my_api_key:
    raise ValueError("API key is not there")
client=Groq(api_key=my_api_key)
groqmodel='llama-3.3-70b-versatile'

documents = [
    "Pizza is my favorite food because I love melted cheese and a crispy crust.",
    "I really enjoy eating pizza with extra cheese and a crunchy crust.",
    "Pizza tastes best when it is hot, cheesy, and freshly baked.",

    "The Himalayan mountains are a great destination for trekking and adventure.",
    "I enjoy hiking through mountains and exploring beautiful natural landscapes.",
    "Mountains are perfect for people who love trekking, hiking, and outdoor adventures.",

    "Python is a programming language commonly used for machine learning and data science.",
    "Python provides many useful libraries for building machine learning applications.",

    "The stock market can be highly volatile and prices may change rapidly.",
    "I watched a comedy movie last night and laughed throughout the entire film.",

    "My laptop battery is draining quickly and needs to be charged frequently."
]

document_embeddings =model_embedding.encode(documents)



def cosine_similarity(a,b):
    return np.dot(a,b)/(np.linalg.norm(a) * np.linalg.norm(b))

query='Where can I go to experience breathtaking peaks and an exciting outdoor expedition?'

q_embedding=model_embedding.encode(query)


def retrive(q_embedding):
    scores=[]
    for i,document in enumerate(document_embeddings):  #i == index and document for each array
        score=cosine_similarity(q_embedding,document)
        scores.append((score,documents[i]))
    scores.sort(reverse=True)
    return scores[0]


def ask_llm(question,context):
    #how to access knowledge throuhg LLM

    sys_prompt=(f'''answer the question in one line
    ,Answer only on based on this context.donot hallucinate.
Context:{context}''')
    system_message={
        'role':'system',
        'content':sys_prompt
    }
    message={
        'role':'user',
        'content':question
    }
    messages=[system_message,message]
    response=client.chat.completions.create(model=groqmodel,messages=messages)
    answer=response.choices[0].message.content
    return answer


score,context=retrive(q_embedding)

answer=ask_llm(query,context)
print(answer)
