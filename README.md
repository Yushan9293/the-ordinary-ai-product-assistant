# The Ordinary AI Product Assistant  
**Custom RAG Chatbot · FastAPI · Pinecone · OpenAI · n8n Ready**

This project is a **production-ready AI product assistant** built for skincare and e-commerce use cases, demonstrated with **The Ordinary** public product data.

It is designed to answer customer questions **only based on real product knowledge**, avoiding hallucinations, and can be easily connected to **n8n, Telegram, Instagram, or other messaging platforms**.

---

## 💡 What This Chatbot Does

- Answers customer questions about skincare products in a **natural, human-like way**
- Retrieves **real product information** (name, category, price, official URL)
- Distinguishes between:
  - casual questions
  - buying intent
  - product comparison requests
- Uses **Retrieval-Augmented Generation (RAG)** to ensure accuracy

This is **not a generic ChatGPT wrapper**, but a grounded, data-backed AI assistant.

---

## 🚀 Key Capabilities

- 🔍 **Accurate Product Retrieval**
  - Product data comes from structured JSON files
  - Prices and URLs are never invented by the AI

- 🧠 **Knowledge-Based Answers**
  - Brand knowledge and usage guidance come from Markdown knowledge files
  - The AI only answers based on retrieved content

- ⚡ **API-First Design**
  - Built with FastAPI
  - Ready to plug into n8n Cloud, Telegram bots, or web apps

- 🔗 **Automation Ready**
  - Designed to work seamlessly with n8n HTTP workflows
  - Ideal for customer support, pre-sales chat, or product recommendation flows

---

## 🏗️ Project Structure (Simplified)

```text
.
├── api.py
│   FastAPI service exposing the chatbot API
│
├── rag.py
│   Core AI logic (retrieval + response generation)
│
├── pinecone_store.py
│   Vector database connection and search logic
│
├── ingest.py
│   Script used to load product data and knowledge into Pinecone
│
├── data/
│   ├── products/
│   │   Public product JSON data (name, price, URL, category)
│   └── knowledge/
│       Markdown knowledge base for brand and usage guidance
│
├── requirements.txt
│   Python dependencies
│
└── README.md


All data included in this repository is **publicly available** and used for demonstration purposes.

---

## 🔁 How the System Works (Simple Explanation)

1. Product and knowledge data are ingested into a vector database (Pinecone)
2. When a user asks a question:

   * Relevant products and knowledge are retrieved
   * The AI generates an answer **only using retrieved data**
3. If the user shows buying intent, the chatbot returns:

   * product name
   * category
   * price
   * official product URL

This ensures **accuracy, consistency, and trustworthiness**.

---

## 🔌 n8n Integration Example

This chatbot is designed to be called from **n8n Cloud** using an HTTP Request node.

**Request**

```json
{
  "question": "I want a gentle exfoliator for beginners",
  "user_id": "telegram_user_id",
  "message_id": "123"
}
```

**Response**

```json
{
  "answer": "A gentle option for beginners is ...",
  "sources": [
    {
      "title": "Lactic Acid 5% + HA",
      "price": "€7.90",
      "url": "https://theordinary.com/..."
    }
  ]
}
```

---

## 🧪 Data Ingestion (One-Time Setup)

Before using the chatbot, product and knowledge data must be loaded once:

```bash
python ingest.py
```

This step is only required when the data changes.

---

## 📦 Deployment

The project is tested and deployable on:

* Linux VPS (Hetzner)
* FastAPI + Uvicorn
* n8n Cloud (HTTP Request node)
* Telegram Bot workflows

It can be adapted easily for:

* e-commerce websites
* Instagram DM automation
* customer support chatbots
* internal product assistants

---

## 👩‍💻 About This Project

This repository demonstrates:

* real-world RAG architecture
* clean backend design
* automation-friendly AI integration

It is suitable as:

* a production starter template
* a client demo
* a portfolio project for AI automation and chatbot development

---

## 📄 License

Demo / portfolio project
Product data belongs to their respective brand owners.

```

---

