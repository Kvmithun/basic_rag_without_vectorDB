# Basic RAG Without Vector Database

This project demonstrates how a **basic Retrieval-Augmented Generation (RAG)** system works from scratch, without using a vector database.

The main idea is simple:

> Instead of directly asking an LLM a question, we first retrieve relevant information from our own documents and then give that information to the LLM to generate an answer.

## What This Project Does

We start with a small collection of documents:

```text
Documents
   ↓
Convert documents into embeddings
   ↓
Store embeddings in memory
   ↓
User asks a question
   ↓
Convert question into an embedding
   ↓
Compare query with every document
   ↓
Find the most similar document
   ↓
Send the retrieved document + question to LLM
   ↓
Generate final answer
```

## 1. Creating Embeddings

The project uses the `all-MiniLM-L6-v2` model from Sentence Transformers.

```python
model_embedding = SentenceTransformer("all-MiniLM-L6-v2")
```

This model converts text into a **384-dimensional numerical vector**.

For example:

```text
"Python is used for machine learning"
                ↓
        Sentence Transformer
                ↓
     [0.21, -0.08, 0.73, ...]
                ↓
           384 values
```

We generate embeddings for all our documents:

```python
document_embeddings = model_embedding.encode(documents)
```

## 2. Creating a Query Embedding

When the user asks a question, we also convert the question into an embedding.

```python
q_embedding = model_embedding.encode(query)
```

Now both the documents and the query are represented as vectors.

```text
Document → Embedding
Document → Embedding
Document → Embedding
       ...
Query    → Embedding
```

## 3. Finding Similar Documents

We need to determine which document is most relevant to the query.

For this, we use **cosine similarity**.

```python
def cosine_similarity(a, b):
    return np.dot(a, b) / (
        np.linalg.norm(a) * np.linalg.norm(b)
    )
```

Cosine similarity compares the **direction of two vectors**.

A higher score means the two embeddings are more semantically similar.

For example:

```text
Query:
"Where can I go for an exciting outdoor expedition?"

Document:
"The Himalayan mountains are a great destination for trekking and adventure."
```

Even though the exact words are different, their **meaning is similar**, so their embeddings should have a relatively high cosine similarity.

## 4. Retrieving the Most Relevant Document

The query embedding is compared with every document embedding:

```python
for i, document in enumerate(document_embeddings):
    score = cosine_similarity(q_embedding, document)
    scores.append((score, documents[i]))
```

Here:

* `i` → index of the document
* `document` → embedding of that document
* `documents[i]` → original text corresponding to that embedding
* `score` → cosine similarity between the query and document

We then sort the results:

```python
scores.sort(reverse=True)
```

and select the document with the highest similarity:

```python
return scores[0]
```

So the retrieval process is:

```text
Query
  ↓
Query Embedding
  ↓
Compare with Document 1 → Score
Compare with Document 2 → Score
Compare with Document 3 → Score
...
  ↓
Highest Score
  ↓
Most Relevant Document
```

## 5. Passing Retrieved Information to the LLM

After retrieving the most relevant document, we pass it to the Groq LLM as **context**.

The LLM receives:

```text
Question + Retrieved Context
```

The system prompt tells the LLM to answer **only using the provided context** and not to hallucinate information.

This is the **Generation** part of RAG.

```text
Retrieved Context
       +
User Question
       ↓
      LLM
       ↓
Final Answer
```

## Why Is This Called RAG?

RAG stands for:

**Retrieval-Augmented Generation**

It has two main stages:

### Retrieval

Find relevant information from the stored documents.

```text
Documents → Embeddings → Similarity Search → Relevant Context
```

### Generation

Give the retrieved context to an LLM and generate an answer.

```text
Question + Context → LLM → Answer
```

Therefore:

```text
        RAG
         │
   ┌─────┴─────┐
   ↓           ↓
Retrieval   Generation
   ↓           ↓
Find data    LLM answer
```

## Why No Vector Database?

For this project, the number of documents is very small, so we can keep the embeddings in memory and compare the query with each document directly.

```text
Small Dataset
     ↓
Python + NumPy
     ↓
Cosine Similarity
```

This is useful for **learning how RAG works internally**.

However, if we have millions of documents and embeddings, comparing a query with every single vector becomes expensive. That's where a **vector database and vector indexing** become useful for scalable retrieval.

## Technologies Used

* Python
* Sentence Transformers
* `all-MiniLM-L6-v2`
* NumPy
* Groq API
* Llama 3.3 70B

## Complete RAG Flow

```text
                 DOCUMENTS
                     ↓
            Sentence Transformer
                     ↓
             Document Embeddings
                     ↓
                Store in memory
                     │
                     │
USER QUERY ──────────┘
     ↓
Query Embedding
     ↓
Cosine Similarity
     ↓
Compare with document embeddings
     ↓
Highest similarity score
     ↓
Relevant document
     ↓
Question + Context
     ↓
Groq LLM
     ↓
FINAL ANSWER
```

### In one sentence

> **This project shows how to build a basic RAG system where text is converted into embeddings, cosine similarity is used to retrieve the most relevant document, and the retrieved context is given to an LLM to generate a grounded answer.**
