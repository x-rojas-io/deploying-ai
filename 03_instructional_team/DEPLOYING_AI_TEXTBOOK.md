# Deploying AI: The Complete Guide

Welcome to the comprehensive theoretical guide for the **Deploying AI** module. This document serves as a textbook for understanding the principles, patterns, and technologies used to build production-grade AI systems.

---

## Table of Contents
1. [Chapter 1: The Foundations of AI Engineering](#chapter-1-the-foundations-of-ai-engineering)
2. [Chapter 2: Representing Meaning (Embeddings & VectorDBs)](#chapter-2-representing-meaning-embeddings--vectordbs)
3. [Chapter 3: Reliability & Evaluation](#chapter-3-reliability--evaluation)
4. [Chapter 4: Structured Data Extraction](#chapter-4-structured-data-extraction)
5. [Chapter 5: Agents & Orchestration](#chapter-5-agents--orchestration)
6. [Chapter 6: Connecting to the World (MCP)](#chapter-6-connecting-to-the-world-mcp)
7. [Appendix A: Frequently Asked Questions (FAQ)](#appendix-a-frequently-asked-questions-faq)
8. [Appendix B: Glossary](#appendix-b-glossary)

---

## Chapter 1: The Foundations of AI Engineering

AI Engineering is the discipline of integrating Large Language Models (LLMs) into practical software applications. Unlike traditional software, where logic is deterministic (code), AI software involves probabilistic components (models) that require careful orchestration.

### 1.1 The API Economy
Modern AI development often relies on consuming models via Application Programming Interfaces (APIs). Understanding API styles is crucial.

*   **REST (Representational State Transfer)**: The standard for web APIs. It uses standard HTTP verbs (`GET` for reading, `POST` for creating/sending data) and is resource-oriented. Most AI providers (OpenAI, Anthropic) use REST-like APIs.
*   **SOAP & RPC**: Older or more specialized protocols. SOAP is strict and XML-based. RPC (Remote Procedure Call) focuses on actions rather than resources.
*   **WebSockets**: Enables persistent, two-way communication. Essential for real-time AI voice or chat interfaces where latency matters.

### 1.2 The Stateless "Completions" Model
The core primitive of LLM interaction is the **Chat Completion**.
*   **Statelessness**: The model typically does not "remember" previous interactions. The developer must send the *entire* conversation history (context) with every new request.
*   **Roles**:
    *   `System`: Sets the behavior/persona of the model.
    *   `User`: The human input.
    *   `Assistant`: The machine output.

### 1.3 Security & Environment Variables
**Never commit API keys to version control.**
*   **The `.env` Pattern**: Secrets are stored in a local file named `.env` (or `.secrets`) which is added to `.gitignore`.
*   **Loading Secrets**: Libraries like `python-dotenv` load these file values into the system's environment variables at runtime, making them accessible to your code without hardcoding them.

---

## Chapter 2: Representing Meaning (Embeddings & VectorDBs)

Computers don't understand text; they understand numbers. To perform semantic search (finding documents based on *meaning* rather than keywords), we need to convert text into numeric vectors.

### 2.1 Tokenization
Before a model sees text, it is chopped into chunks called **tokens**.
*   **What is a token?** It can be a word, part of a word, or even a space. A common rule of thumb: 1000 tokens ≈ 750 words.
*   **Algorithms**: Models use specific algorithms (e.g., Byte-Pair Encoding or WordPiece) to optimize this vocabulary. For example, "tokenization" might become `["token", "ization"]`.

### 2.2 Embeddings
An embedding is a list of floating-point numbers (a vector) that represents the semantic meaning of a piece of text.
*   **The Concept**: In a high-dimensional space, concepts that are similar (e.g., "Cat" and "Kitten") will have vectors that are mathematically close to each other. Concepts that are unrelated (e.g., "Cat" and "Microscope") will be far apart.
*   **BERT**: A seminal model for generating these embeddings, capturing context from both left and right directions of a word.

### 2.3 Vector Databases (VectorDBs)
A traditional SQL database searches for exact matches (`WHERE title = 'Cat'`). A VectorDB searches for *nearest neighbors*.
*   **Workflow**:
    1.  **Ingestion**: Split documents into chunks.
    2.  **Embedding**: Run chunks through an embedding model (e.g., `text-embedding-3-small`).
    3.  **Storage**: Save the vector + original text in the VectorDB (e.g., **ChromaDB**).
    4.  **Query**: Embed the user's question and find the top $k$ vectors that are closest to it.
*   **Docker**: We often run VectorDBs in Docker containers to ensure consistent environments and easy deployment.

---

## Chapter 3: Reliability & Evaluation

 LLMs are stochastic (random). How do we know if a system is working correctly?

### 3.1 Retrieval Augmented Generation (RAG)
RAG combines a retrieval system (VectorDB) with a generative model (LLM).
*   **Goal**: Reduce hallucinations by grounding the model in specific data.
*   **The Flow**: User Query -> Search VectorDB -> Context + Query -> LLM -> Answer.

### 3.2 Evaluation Metrics
*   **Log Probabilities (Logprobs)**: The probability associated with each token generated. Low probabilities indicate the model is unsure, which can be a signal to flag the answer for human review.
*   **G-Eval (AI as a Judge)**: Using a highly capable model (like GPT-4o) to grade the outputs of your system.
    *   **Faithfulness**: Does the answer come *only* from the provided context?
    *   **Answer Relevancy**: Did the system actually answer the user's question?
    *   **Contextual Recall**: Did the retrieval step find the right documents?

---

## Chapter 4: Structured Data Extraction

LLMs naturally speak in prose (paragraphs). Software systems need specific data structures (JSON).

### 4.1 The Problem
Asking an LLM for a list of names might result in: "Sure! Here is the list: 1. Alice, 2. Bob."
A Python script tries to parse this and fails because of the extra text ("Sure! Here...").

### 4.2 The Solution
*   **Pydantic**: A Python library that defines strict data schemas (types).
*   **Function Calling / Tools**: Modern APIs allow you to pass a JSON schema (or Pydantic class) to the model. The model is fine-tuned to output *only* valid JSON conforming to that schema.
*   **LangChain Integration**: `model.with_structured_output(MyPydanticClass)` handles the translation between Python objects and API schemas automatically.

---

## Chapter 5: Agents & Orchestration

An **Agent** is a system where the LLM acts as a reasoning engine to decide *what to do next*, rather than just generating text.

### 5.1 LangGraph
LangGraph is a framework for building stateful, multi-step agent applications.
*   **State**: A shared dictionary (e.g., containing the conversation history) that is passed around.
*   **Nodes**: Python functions that perform work. Example: A node that calls the LLM, or a node that executes a Google Search.
*   **Edges**: The control flow.
    *   *Normal Edge*: Go from Node A to Node B.
    *   *Conditional Edge*: LLM decides "If I found the answer, go to END. If I need more info, go back to SEARCH."

### 5.2 The LLM Compiler Architecture
For complex tasks, running steps sequentially is too slow. The **LLM Compiler** pattern optimizes this:
1.  **Planner**: Decomposition of the user request into a dependency graph of tasks.
2.  **Task Fetching**: Executing tasks in parallel (e.g., searching for "Weather in NY" and "Weather in London" simultaneously).
3.  **Joiner**: Synthesizing the results into a final answer.

---

## Chapter 6: Connecting to the World (MCP)

**Model Context Protocol (MCP)** is an open standard that decouples AI assistants from the data they access.

### 6.1 The Architecture
*   **MCP Server**: A specialized connector for a specific resource (e.g., a "Google Drive Server", a "Postgres Server"). It exposes:
    *   **Resources**: Data meant to be read (files, logs).
    *   **Tools**: Functions meant to be executed (calculators, API calls).
    *   **Prompts**: Validated templates for interaction.
*   **MCP Client**: The connector embedded in the host application.
*   **MCP Host**: The application the user interacts with (e.g., Claude Desktop, IDE, or your custom app).

### 6.2 Why it matters
Instead of building a custom "Google Drive Integration" for every single AI app, developers build *one* MCP Server for Google Drive, and *any* MCP-compliant app can use it.

---

## Appendix A: Frequently Asked Questions (FAQ)

**Q: Why do we need a Vector Database? Can't we just use `grep` or SQL?**
**A:** Keyword search (`grep`, `LIKE %...%`) fails on synonyms. If you search for "automobile", keyword search misses documents containing "car". Vector search understands that "automobile" and "car" are semantically close and retrieves both.

**Q: What is the difference between RAG and Fine-tuning?**
**A:**
*   **RAG (Retrieval Augmented Generation)**: Giving the model a textbook during the exam. Good for factual knowledge, private data, and data that changes often.
*   **Fine-tuning**: Sending the model to school to learn a new skill. Good for teaching specific formats, tones, or specialized "languages" (like medical shorthand). It typically does *not* teach new facts reliably.

**Q: Why use `uv` instead of `pip`?**
**A:** `uv` is a modern, extremely fast package manager that replaces `pip`, `venv`, and other tools. It resolves dependencies much faster and handles Python version management natively.

**Q: What is the difference between a Chain and an Agent?**
**A:**
*   **Chain**: A hardcoded sequence of steps (A -> B -> C). It always happens in the same order.
*   **Agent**: A system where the LLM decides the sequence. It might go A -> C -> B, or loop on A until satisfied. It is more flexible but harder to control.

---

## Appendix B: Glossary

*   **Chunking**: Breaking a large document into smaller pieces (chunks) suitable for embedding.
*   **Context Window**: The maximum amount of text (in tokens) an LLM can process at one time.
*   **Embedding**: A vector (list of numbers) representing the meaning of text.
*   **Hallucination**: When an LLM generates a plausible-sounding but factually incorrect answer.
*   **JSONL**: JSON Lines. A file format where each line is a valid JSON object. Preferred for large AI datasets.
*   **LangGraph**: A library for building stateful, graph-based agent applications.
*   **LLM (Large Language Model)**: A deep learning model trained on massive datasets to understand and generate text.
*   **Logprobs**: Logarithmic probabilities. A score indicating how confident the model is in its token choice.
*   **MCP (Model Context Protocol)**: A standard for connecting AI models to external data and tools.
*   **Prompt Engineering**: The art of crafting inputs (prompts) to guide the LLM towards the desired output.
*   **RAG (Retrieval Augmented Generation)**: A technique that enhances LLMs by retrieving relevant data from external sources before generation.
*   **Schema**: A definition of the structure of data (e.g., "This object must have a 'name' field which is a string").
*   **Temperature**: A parameter controlling the randomness of LLM output. High temperature = more creative/random; Low temperature = more deterministic/focused.
*   **Token**: The basic unit of text processing for an LLM (roughly 0.75 words).
*   **Vector Database**: A database optimized for storing and searching vector embeddings.
