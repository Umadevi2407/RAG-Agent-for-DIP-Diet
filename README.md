# 🥗 RAG Agent for DIP Diet
**Problem Statement No. 9 — Disciplined & Intelligent Plan Diet Assistant**

An intelligent diet assistant built with **Vector Store RAG**, **IBM Granite on watsonx.ai**, and **Streamlit** — delivering personalized DIP Diet recommendations through conversational AI.

---

## 🎯 Objectives
1. **Ingest & Index** DIP Diet documents into a ChromaDB vector store for semantic search
2. **Conversational Q&A** — natural-language questions answered by IBM Granite (70B)
3. **Personalized Agentic Layer** — adapts meal plans to user profile, BMI, medical conditions, and goals
4. **Visualize Diet Guidance** — 4-tab Streamlit dashboard with charts, meal plans, and lifestyle tips

---

## 🏗️ Project Structure
```
dip_diet_rag/
├── app.py               # Streamlit dashboard (4 tabs: Chat, Meal Plan, Nutrition, Lifestyle)
├── rag_agent.py         # DIPDietRAGAgent — IBM Granite + ChromaDB RAG pipeline
├── ingest.py            # Document ingestion → chunking → embedding → ChromaDB
├── requirements.txt     # Python dependencies
├── .env.example         # Credential template (copy to .env and fill in)
├── data/
│   ├── dip_core_principles.txt     # DIP philosophy, rules, approved/prohibited foods
│   ├── dip_meal_timing.txt         # Meal schedules, two-plate rule, prep guidelines
│   └── dip_disease_protocols.txt   # Diabetes, hypertension, obesity, cardiac protocols
└── vector_store/
    └── chroma_db/       # Persisted ChromaDB index (auto-created by ingest.py)
```

---

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install -r dip_diet_rag/requirements.txt
```

### 2. Configure credentials
Copy `.env.example` to `.env` and fill in your watsonx.ai credentials:
```env
WATSONX_API_KEY=your_api_key_here
WATSONX_PROJECT_ID=your_project_id_here
WATSONX_URL=https://us-south.ml.cloud.ibm.com
GRANITE_MODEL=meta-llama/llama-3-3-70b-instruct
```

### 3. Build the vector store
```bash
python dip_diet_rag/ingest.py
```

### 4. Launch the dashboard
```bash
python -m streamlit run dip_diet_rag/app.py --server.port 8501
```
Open **http://localhost:8501** in your browser.

---

## 🛠️ Technology Stack

| Layer | Technology | Version |
|---|---|---|
| LLM | IBM Granite (Llama-3-3-70B) on watsonx.ai | 70B params |
| LLM SDK | ibm_watsonx_ai | 1.8.0 |
| Vector Store | ChromaDB | 1.5.9 |
| Embeddings | ONNX MiniLM-L6-v2 (built-in) | onnxruntime 1.31.0 |
| UI Framework | Streamlit | 1.66.0 |
| Charts | Plotly | 7.1.0 |
| Data | Pandas | 2.3.3 |
| Config | python-dotenv | 1.2.4 |
| Runtime | Python | 3.12.10 |

---

## 🤖 Agentic AI Features
- **Autonomous initialization** — agent self-connects, self-tests, and reports health status
- **Semantic retrieval** — top-5 cosine-similar chunks from DIP knowledge base
- **Personalized prompting** — user profile injected into every IBM Granite call
- **Condition-aware meal plans** — adapts for diabetes, hypertension, weight loss
- **Graceful fallback** — serves retrieved context if LLM is unavailable
- **Session memory** — chat history persists across conversation turns

---

## 📊 Dashboard Tabs
| Tab | Features |
|---|---|
| 💬 AI Chat | RAG-powered Q&A with IBM Granite, quick questions, source citations |
| 📅 Meal Plan | 7-day personalized meal plan with calorie tracking |
| 📊 Nutrition | Macro pie, calorie bar, weekly line chart, nutrient radar |
| 🌿 Lifestyle | Morning/evening routines, hydration rules, food substitutions |

---

## 📄 License
For educational and research purposes — FDP Project, Problem Statement No. 9.

> ⚠️ For informational purposes only. Consult a healthcare professional for medical advice.
