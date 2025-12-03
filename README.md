# 📚 Mathematical Research Multi-Agent System

An intelligent AI assistant that uses 5 specialized agents, RAG technology, and adaptive learning to provide high-quality mathematical explanations.

![Status](https://img.shields.io/badge/status-active-success) ![Python](https://img.shields.io/badge/python-3.10%2B-blue) ![Streamlit](https://img.shields.io/badge/streamlit-app-red)

---

## 🚀 Quick Start

```bash
git clone https://github.com/bydursun/Mathematical-Research-Multi-Agent-System.git
cd Mathematical-Research-Multi-Agent-System
pip install -r requirements.txt
streamlit run math_rag_crewai_app.py
```

Enter your OpenAI API key → Ask a math question → Get detailed answers in 60-90 seconds.

---

## ✨ Key Features

- **🤖 Multi-Agent System** - 5 specialized agents working together via CrewAI
- **🔍 Smart RAG** - ChromaDB vector search with ChatGPT fallback
- **💡 Adaptive Learning** - Learns from user feedback to improve responses
- **🎨 Customizable** - Control answer style and tone
- **🌍 Multi-Language** - Auto-detects and responds in your language
- **🛡️ Quality Assured** - Built-in reflection validates accuracy and clarity

---

## 🤖 Agent Architecture

```
┌─────────────────────────────────────┐
│      MANAGER AGENT (Planner)       │
│   Coordinates workflow & tasks     │
└──────────────┬──────────────────────┘
               │
    ┌──────────┼──────────┬──────────┐
    ▼          ▼          ▼          ▼
┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐
│  RAG   │ │Research│ │Reflect │ │Summary │
│ Agent  │ │ Agent  │ │ Agent  │ │ Agent  │
└────────┘ └────────┘ └────────┘ └────────┘
```

### Agent Roles

1. **Manager Agent** - Coordinates all agents and delegates tasks
2. **RAG Agent** - Retrieves relevant information from knowledge base
3. **Research Agent** - Analyzes and explains mathematical concepts
4. **Reflection Agent** - Evaluates answer quality (accuracy, clarity, completeness)
5. **Summary Agent** - Creates final polished answer

---

## 🔍 RAG Implementation

- **Vector Database**: ChromaDB with persistent storage
- **Embeddings**: HuggingFace `all-MiniLM-L6-v2` (384 dimensions)
- **Retrieval**: Top-3 similarity search with relevance scores
- **Confidence Threshold**: 0.3 (falls back to ChatGPT if below)

### How It Works

1. User asks a question
2. System searches vector database for relevant content
3. If confidence ≥ 0.3: Uses retrieved context
4. If confidence < 0.3: Calls ChatGPT API for enrichment
5. Agents process and refine the answer
6. User provides feedback for continuous improvement

---

## 🔄 Reflection Mechanism

The Reflection Agent evaluates every answer on 4 dimensions:

1. **Accuracy** - Mathematical correctness
2. **Clarity** - Easy to understand explanations
3. **Completeness** - Full coverage of the topic
4. **Pedagogy** - Effective teaching approach

---

## 💾 Adaptive Memory

The system learns from your feedback:

- **👍/👎 Ratings** - Helps improve future responses
- **Style Preferences** - Concise, balanced, or detailed
- **Tone Preferences** - Neutral, friendly, or academic
- **Comment History** - Stored locally for continuous improvement

All feedback is saved in `db/feedback.jsonl` and `db/memory.json`.

---

## 📁 Project Structure

```
.
├── math_rag_crewai_app.py    # Main application
├── requirements.txt          # Dependencies
├── README.md                 # This file
├── knowledge/                # Knowledge base
│   ├── math_knowledge.txt    # Core math concepts
│   └── limits.txt            # Calculus topics
├── db/                       # Feedback storage
│   ├── feedback.jsonl
│   └── memory.json
└── math_db/                  # Vector database
    └── chroma.sqlite3
```

---

## 💻 Installation

### Prerequisites

- Python 3.10+
- OpenAI API key ([get one here](https://platform.openai.com/api-keys))

### Install Dependencies

```bash
pip install -r requirements.txt
```

Main dependencies:
- `streamlit` - Web UI
- `crewai` - Multi-agent framework
- `langchain-community` & `langchain-openai` - RAG integration
- `chromadb` - Vector database
- `sentence-transformers` - Embeddings
- `openai` - ChatGPT API

---

## 🎯 Usage

### 1. Start the Application

```bash
streamlit run math_rag_crewai_app.py
```

### 2. Configure API Key

Open `http://localhost:8501` and enter your OpenAI API key in the sidebar.

### 3. Ask a Question

Type your mathematical question (e.g., "What is a derivative?") and click **🚀 Start Research**.

### 4. Provide Feedback

- Choose your preferred style and tone
- Click **👍 Helpful** or **👎 Not Helpful**
- Add optional comments
- System learns for next time

---

## 🗂️ Knowledge Base Management

### Adding New Documents

1. Add `.txt` files to the `knowledge/` folder
2. Delete the `math_db/` folder:
   ```powershell
   Remove-Item -Recurse -Force math_db
   ```
3. Restart the app to rebuild the database

### Current Topics Covered

- Calculus (derivatives, integrals, limits)
- Algebra (equations, polynomials)
- Probability & Statistics
- Number Theory
- Topology
- Graph Theory

---

## 📊 System Workflow

```
User Question
     ↓
RAG Retrieval (ChromaDB)
     ↓
Score ≥ 0.3?
     ├─ YES → Use RAG Context
     └─ NO  → ChatGPT Fallback
     ↓
CrewAI Agents Process
     ↓
1. RAG Agent: Retrieve context
2. Research Agent: Analyze & explain
3. Reflection Agent: Evaluate quality
4. Summary Agent: Polish final answer
     ↓
Display Answer
     ↓
User Feedback (👍/👎)
     ↓
Update Memory
```

---

## 🛡️ Responsible AI Principles

- **Privacy**: All data stored locally, no external uploads
- **Fairness**: Equal treatment for all queries
- **Explainability**: Agent workflow visible to users
- **Accountability**: Feedback loop for error correction

---

## 🐛 Troubleshooting

---

## 🐛 Troubleshooting

**App won't start:**
```bash
pip install -r requirements.txt
streamlit run math_rag_crewai_app.py
```

**ChromaDB error:**
```powershell
Remove-Item -Recurse -Force math_db
```

**OpenAI API error:**
- Verify your API key starts with `sk-`
- Check you have available credits
- Enter key in Streamlit sidebar

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/new-feature`
3. Commit changes: `git commit -m 'Add new feature'`
4. Push: `git push origin feature/new-feature`
5. Open a Pull Request

---

## 📞 Contact

**Developer**: [@bydursun](https://github.com/bydursun)  
**Repository**: [Mathematical-Research-Multi-Agent-System](https://github.com/bydursun/Mathematical-Research-Multi-Agent-System)  
**Issues**: [GitHub Issues](https://github.com/bydursun/Mathematical-Research-Multi-Agent-System/issues)

---

## 📜 License

MIT License - See [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **CrewAI** - Multi-agent orchestration
- **LangChain** - RAG framework
- **ChromaDB** - Vector database
- **HuggingFace** - Embeddings
- **OpenAI** - ChatGPT API
- **Streamlit** - Web interface

---

## 📚 References

1. Lewis, P., et al. (2020). "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks." NeurIPS.
2. Wang, X., et al. (2023). "Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection." arXiv.
3. [CrewAI Documentation](https://docs.crewai.com)
4. [LangChain Documentation](https://python.langchain.com)

---

**Built with by Abdullah Dursun**  
*Production-ready • Open Source • Continuously Improved*

---

**Status**: ✅ Complete and Working  
**Last Updated**: December 3, 2025  
**Version**: 1.0.0
