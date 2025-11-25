# Mathematical Research Multi-Agent System  
Team #2 - Mathematical Enquiries

![Status](https://img.shields.io/badge/status-active-green) ![Python](https://img.shields.io/badge/python-3.10%2B-blue) ![Streamlit](https://img.shields.io/badge/streamlit-app-red) ![License](https://img.shields.io/badge/license-educational-lightgrey)

A sophisticated multi-agent AI system using CrewAI that helps Research Analysts explore and understand mathematical concepts through intelligent collaboration between 5 specialized AI agents.

---

## 🚀 Quick Start

```bash
git clone https://github.com/bydursun/Mathematical-Research-Multi-Agent-System.git
cd <Mathematical-Research-Multi-Agent-System>

# Optional: create & activate virtual environment (Windows PowerShell)
python -m venv .venv
./.venv/Scripts/Activate.ps1

# Install dependencies
pip install -r requirements.txt

# Add your OpenAI key (do NOT commit .env)
echo OPENAI_API_KEY=sk-xxx > .env

# Run the application
streamlit run math_rag_crewai_app.py
```

Open http://localhost:8502 (or 8501) in your browser.

---

## 📋 Features

✅ **5 AI Agents** in hierarchical structure (Manager + 4 Workers)
✅ **RAG Implementation** with ChromaDB vector database
✅ **Reflection Mechanism** for quality assurance
✅ **2 Custom Tools** (RAG Search + Math Calculator)
✅ **Streamlit UI** for interactive research
✅ **Responsible AI** principles followed

---

## 🏗️ Architecture

```
User Question
     ↓
[MANAGER AGENT] - Coordinates workflow
     ↓
[RAG AGENT] - Retrieves knowledge from ChromaDB
     ↓
[RESEARCH AGENT] - Analyzes with context + calculations
     ↓
[REFLECTION AGENT] - Evaluates quality (4 metrics)
     ↓
[SUMMARIZATION AGENT] - Creates polished answer
     ↓
Final Answer
```

---

## 👥 The 5 Agents

1. **Manager Agent (Planner)** - Coordinates all workers, delegates tasks, synthesizes outputs
2. **RAG Agent (Worker)** - Searches knowledge base using vector similarity
3. **Research Agent (Worker)** - Analyzes concepts, provides explanations, performs calculations
4. **Reflection Agent (Worker)** - Evaluates accuracy, clarity, completeness, pedagogical value
5. **Summarization Agent (Worker)** - Creates clear, well-structured final answers

---

## 🛠️ Technologies Used

- **CrewAI** - Multi-agent orchestration framework
- **LangChain** - LLM integration and tooling
- **ChromaDB** - Vector database for RAG
- **HuggingFace** - Embeddings (all-MiniLM-L6-v2)
- **OpenAI** - GPT-3.5-turbo for agents
- **Streamlit** - Interactive web interface
- **Python 3.10** - Programming language

---

## 📁 Project Structure

```
.
├── math_rag_crewai_app.py          # Main application with 5 AI agents
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── .gitignore                      # Git ignore rules (excludes vector DBs)
├── knowledge/                      # Mathematical knowledge base
│   ├── math_knowledge.txt          # General mathematics concepts
│   └── limits.txt                  # Calculus limits knowledge
├── db/                             # ChromaDB vector storage (auto-generated)
│   ├── chroma.sqlite3
│   └── [collection data]
└── math_db/                        # Alternative ChromaDB storage (auto-generated)
    ├── chroma.sqlite3
    └── [collection data]
```

**Note:** The `db/` and `math_db/` directories are automatically generated on first run and excluded from version control.

---

## 💻 Installation

### Prerequisites
- Python 3.10+
- OpenAI API key (`OPENAI_API_KEY`)

### Dependencies
Managed via `requirements.txt`:

```bash
pip install -r requirements.txt
```

### Environment Variables
Create a `.env` file or set in shell:

```
OPENAI_API_KEY=your_openai_key_here
```

PowerShell examples:
```powershell
# Session only
$env:OPENAI_API_KEY="sk-..."
# Persist for future shells
setx OPENAI_API_KEY "sk-..."
```

---

## 🎯 Usage

1. **Start the application:**
   ```bash
   streamlit run math_rag_crewai_app.py
   ```

2. **Enter your OpenAI API key** in the sidebar

3. **Type your mathematical question** (e.g., "What is a derivative?")

4. **Click "Start Research"** and wait 1-2 minutes

5. **View the final answer** and agent workflow details

---

## 🧠 RAG Implementation

- **Vector Database:** ChromaDB with persistent storage
- **Embeddings:** HuggingFace all-MiniLM-L6-v2 (384 dimensions)
- **Retrieval:** Top-3 similarity search
- **Knowledge Base:** Covers calculus, algebra, probability, statistics, number theory, topology, graph theory

---

## 🔄 Reflection Mechanism

The Reflection Agent evaluates answers using 4 metrics:

1. **Accuracy** - Mathematical correctness
2. **Clarity** - Explanation understandability
3. **Completeness** - Topic coverage
4. **Pedagogical Value** - Teaching effectiveness

Feedback is used by the Summarization Agent to improve the final answer.

---

## 🗄️ Vector Database Management

The ChromaDB vector database is stored in `db/` and `math_db/` directories. These are auto-generated from the knowledge files and excluded from Git via `.gitignore`.

**Rebuild Database (if needed):**
```powershell
# Windows PowerShell
Remove-Item -Recurse -Force db, math_db
streamlit run math_rag_crewai_app.py  # Recreates DB on startup
```

**Adding New Knowledge:**
1. Add/edit files in `knowledge/` directory
2. Delete the database folders
3. Restart the application to regenerate embeddings

---

## 🛡️ Responsible AI Principles

- **Privacy:** Local storage, no permanent data retention, session-only API keys
- **Fairness:** Equal treatment of all queries regardless of complexity
- **Explainability:** Clear agent workflow visible to users
- **Responsible:** Hierarchical oversight and quality assurance through reflection

---

## 🔐 Security & Privacy

- Never commit your real `OPENAI_API_KEY`.
- Use `.env` (already ignored) for secrets.
- Remove and regenerate vector DBs if corruption occurs.
- All processing is ephemeral; no personal data stored.

---

## 📊 Performance Metrics

- **Retrieval Precision** - Relevance of retrieved documents
- **Context Relevance** - Quality of RAG context
- **Answer Completeness** - Coverage of the topic
- **Response Time** - End-to-end execution time

---

## 🎓 Educational Use Cases

- Understanding mathematical concepts
- Learning calculus, algebra, probability
- Getting detailed explanations with examples
- Exploring mathematical topics in depth
- Research assistance for students and analysts

---

## 🔧 Custom Tools

### 1. RAGSearchTool
- Searches ChromaDB vector database
- Uses similarity search (k=3)
- Returns relevant mathematical context

### 2. MathCalculatorTool
- Performs safe mathematical calculations
- Supports: +, -, *, /, ** (power), parentheses
- Restricted eval for security

---

## 📖 Knowledge Base

The system includes two knowledge files in the `knowledge/` directory:

- **math_knowledge.txt** - Core mathematical concepts covering calculus, algebra, probability, statistics, number theory, topology, and graph theory
- **limits.txt** - Detailed explanations of limits in calculus with examples and properties

These files are automatically indexed into ChromaDB vector database on first run.

---

## 🎬 Demo Workflow

**Example Research Question:** "What is a derivative?"

1. **Manager Agent** coordinates the workflow
2. **RAG Agent** retrieves relevant calculus concepts from ChromaDB
3. **Research Agent** analyzes and explains derivatives with examples
4. **Reflection Agent** evaluates the answer quality (accuracy, clarity, completeness, pedagogical value)
5. **Summarization Agent** creates the polished final answer

Response time: 1-2 minutes for complete multi-agent processing

---

## 🏆 Project Deliverables Met

✅ Multi-agent architecture (5 agents max)
✅ Hierarchical model (Manager + Workers)
✅ LLM choices justified (GPT-3.5-turbo)
✅ Tools defined and justified (2 tools)
✅ Agent responsibilities documented
✅ Workflow diagrams provided
✅ Naive RAG with ChromaDB
✅ Performance measures identified
✅ Reflection capability with metrics
✅ Knowledge base architecture described
✅ UML diagrams included
✅ AI principles addressed
✅ Built with CrewAI
✅ Streamlit frontend
✅ Working prototype

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/<name>`
3. Install dependencies: `pip install -r requirements.txt`
4. Implement changes (keep style consistent)
5. Test locally: `streamlit run math_rag_crewai_app.py`
6. Open a Pull Request describing motivation & changes

Please open issues for bugs, enhancement ideas, documentation gaps, or performance discussions.

---

## 🧪 Suggested Future Improvements

- Add unit tests for tools (calculator edge cases)
- Enhanced retrieval evaluation metrics dashboard
- Support for additional mathematical domains (geometry, combinatorics)
- Optional local LLM integration for offline use
- Caching repeated retrieval calls
- Export answer summaries to PDF/Markdown automatically

---

## 🐛 Troubleshooting

**App won't start:**
```bash
streamlit run math_rag_crewai_app.py
```

**ChromaDB error:**
```powershell
# Delete both database folders and restart
Remove-Item -Recurse -Force db, math_db
streamlit run math_rag_crewai_app.py
```

**OpenAI API error:**
- Ensure your API key is valid and has available credits
- Check the key format starts with `sk-`
- Verify you've entered it in the Streamlit sidebar

**Slow response:**
- First query may take 1-2 minutes (normal for multi-agent processing)
- The Manager coordinates 4 worker agents sequentially
- Database initialization on first run adds extra time

---

## 📚 References

- [CrewAI Documentation](https://docs.crewai.com/)
- [LangChain Documentation](https://python.langchain.com/)
- [ChromaDB Documentation](https://docs.trychroma.com/)
- [Streamlit Documentation](https://docs.streamlit.io/)

---

## 👨‍💻 Development

**Framework:** CrewAI + LangChain
**Frontend:** Streamlit
**Database:** ChromaDB
**LLM:** OpenAI GPT-3.5-turbo
**Embeddings:** HuggingFace all-MiniLM-L6-v2

---

## 📝 License

Educational project for AI Systems Design course. If publishing publicly, add a proper license (e.g., MIT) in a new `LICENSE` file:

```
MIT License
Copyright (c) 2025 Team #2
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction...
```

Until then, usage is restricted to course educational purposes.

---

## 🙏 Acknowledgments

- Course: AI Systems Design
- Project: Multi-Agent Research and Summarization System
- Team #2: Mathematical Enquiries
- Framework: CrewAI for multi-agent orchestration
- Database: ChromaDB for vector storage




---

**Status:** ✅ Complete and Working

**Last Updated:** November 24, 2025

**Version:** 1.0.0
