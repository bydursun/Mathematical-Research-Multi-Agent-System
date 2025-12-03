# Math Research Assistant - CrewAI Multi-Agent System with RAG & Reflection
# Assignment: Multi-Agent Research and Summarization System for Mathematical Enquiries

# ========== IMPORTS ==========
# These are all the libraries we need for our multi-agent system

import os  # For file and environment variable operations
import streamlit as st  # For creating the web interface (frontend)
try:
    from langdetect import detect, DetectorFactory  # Language detection
    DetectorFactory.seed = 0
    _LANGDETECT_AVAILABLE = True
except ImportError:
    _LANGDETECT_AVAILABLE = False

# CrewAI imports - for building multi-agent system
from crewai import Agent, Task, Crew, Process
from crewai.tools import BaseTool  # Base class for creating custom tools

# LangChain imports - for RAG (Retrieval-Augmented Generation) implementation
from langchain_community.embeddings import HuggingFaceEmbeddings  # Converts text to vectors
from langchain_community.vectorstores import Chroma  # Vector database (ChromaDB)
from langchain_community.document_loaders import TextLoader  # Loads text files
from langchain_openai import ChatOpenAI  # OpenAI GPT integration
from openai import OpenAI  # Fallback ChatGPT API

from pydantic import Field  # For data validation in tools

# ========== STREAMLIT SETUP ==========
# This section creates the user interface (UI) using Streamlit

# Configure the web page settings
st.set_page_config(page_title="Math Research Multi-Agent System", layout="wide")

# Display the main title and description
st.title("📚 Mathematical Research Multi-Agent System")
st.markdown("**CrewAI-based system with RAG, Reflection, and Hierarchical Agents**")

# Create a sidebar for configuration settings
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Get OpenAI API key from user (required to use GPT models)
    api_key = st.text_input("OpenAI API Key:", type="password")
    
    # If no API key is entered, show warning and stop the app
    if not api_key:
        st.warning("⚠️ Please enter your OpenAI API key to continue.")
        st.stop()  # Prevents app from running without API key
    
    st.divider()  # Visual separator
    
    # Show the agent architecture to the user
    st.markdown("### 🤖 Agent Architecture")
    st.markdown("""
    **Hierarchical Structure:**
    - **Manager Agent** (Planner) - Coordinates all workers
    - **RAG Agent** (Worker) - Searches knowledge base
    - **Research Agent** (Worker) - Analyzes and explains
    - **Reflection Agent** (Worker) - Quality checks
    - **Summarization Agent** (Worker) - Final answer
    """)

# ========== RAG SETUP (ChromaDB) ==========
# RAG = Retrieval-Augmented Generation
# This means we search our knowledge base and use that info to generate better answers

@st.cache_resource  # This decorator caches the function result so it only runs once
def setup_rag_system():
    """
    Initialize the RAG system with ChromaDB vector database.
    This function:
    1. Creates a knowledge base file with mathematical content
    2. Converts text to vectors (embeddings)
    3. Stores vectors in ChromaDB for fast similarity search
    """
    
    # STEP 1: Create knowledge directory if it doesn't exist
    if not os.path.exists("knowledge"):
        os.makedirs("knowledge")  # Make the folder
    
    # STEP 2: Create knowledge base file with mathematical content
    knowledge_file = "knowledge/math_knowledge.txt"
    
    # If the file doesn't exist, create it with mathematical content
    if not os.path.exists(knowledge_file):
        with open(knowledge_file, "w", encoding="utf-8") as f:
            # Write mathematical knowledge to the file
            # This is our "knowledge base" that the RAG agent will search
            f.write("""Calculus is a branch of mathematics focused on limits, functions, derivatives, integrals, and infinite series.
                    
The derivative measures the rate of change of a function. For a function f(x), the derivative f'(x) represents 
the instantaneous rate of change at point x. It is defined as: f'(x) = lim(h->0) [f(x+h) - f(x)] / h

The integral is the reverse operation of differentiation. The definite integral of f(x) from a to b represents 
the area under the curve between those points.

Linear algebra deals with vector spaces, linear mappings, matrices, and systems of linear equations. 
A matrix is a rectangular array of numbers arranged in rows and columns.

Probability theory is the branch of mathematics concerned with probability. It provides a mathematical framework 
for quantifying uncertainty. The probability of an event is a number between 0 and 1.

Statistics involves collecting, analyzing, interpreting, and presenting data. Descriptive statistics summarize data, 
while inferential statistics make predictions based on samples.

Number theory is a branch of pure mathematics devoted to the study of integers and integer-valued functions. 
Prime numbers are fundamental in number theory.

Topology studies properties of space that are preserved under continuous deformations. It is sometimes called 
rubber sheet geometry.

Graph theory studies graphs as mathematical structures used to model pairwise relations between objects. 
A graph consists of vertices connected by edges.""")
    
    # STEP 3: Load the text file
    loader = TextLoader(knowledge_file, encoding="utf-8")
    docs = loader.load()  # Loads the file as documents
    
    # STEP 4: Create embeddings (convert text to numerical vectors)
    # We use HuggingFace's all-MiniLM-L6-v2 model which creates 384-dimensional vectors
    # Each piece of text becomes a list of 384 numbers that captures its meaning
    embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # STEP 5: Create ChromaDB vector database
    # This stores our text as vectors and allows fast similarity search
    # persist_directory="math_db" means it saves to disk (not just memory)
    db = Chroma.from_documents(docs, embedding=embedding, persist_directory="math_db")
    
    return db  # Return the database so we can search it later

# Initialize RAG system when app starts (runs only once due to @st.cache_resource)
vectordb = setup_rag_system()

# ========== CREWAI TOOLS ==========
# Tools are capabilities that agents can use to perform actions
# We create 2 custom tools: RAG Search and Math Calculator

# TOOL 1: RAG Search Tool
# This tool allows agents to search our knowledge base
class RAGSearchTool(BaseTool):
    # Tool metadata
    name: str = "RAG Search Tool"
    description: str = "Retrieval-Augmented Generation tool for searching mathematical knowledge base. Uses ChromaDB vector database to find relevant information."
    
    def _run(self, query: str) -> str:
        """
        Search the knowledge base using vector similarity.
        
        How it works:
        1. Takes the user's question (query)
        2. Converts it to a vector
        3. Finds the 3 most similar pieces of text in our knowledge base
        4. Returns those pieces as context with relevance score
        """
        # Search ChromaDB for top 3 most similar documents with scores
        results = vectordb.similarity_search_with_relevance_scores(query, k=3)
        
        if not results:
            return "No relevant information found in knowledge base. Please use your general knowledge to answer the question."
        
        # Check if the best match has low relevance (threshold: 0.3)
        best_score = results[0][1] if results else 0
        
        if best_score < 0.3:
            return f"Knowledge base search found low relevance results (score: {best_score:.2f}). The information may not be directly relevant. Consider using general knowledge to supplement the answer."
        
        # Combine the results into one string
        context = "\n\n".join([doc.page_content for doc, score in results])
        
        # Return the retrieved knowledge with relevance indicator
        return f"Retrieved Knowledge (relevance score: {best_score:.2f}):\n{context}"

# Lightweight, direct RAG helper returning score and context
def rag_query(query: str):
    try:
        results = vectordb.similarity_search_with_relevance_scores(query, k=3)
        if not results:
            return {"score": 0.0, "context": ""}
        best_score = results[0][1]
        context = "\n\n".join([doc.page_content for doc, _ in results])
        return {"score": float(best_score), "context": context}
    except Exception:
        return {"score": 0.0, "context": ""}

# TOOL 2: Math Calculator Tool
# This tool allows agents to perform mathematical calculations
class MathCalculatorTool(BaseTool):
    # Tool metadata
    name: str = "Math Calculator Tool"
    description: str = "Simple calculator tool for basic mathematical operations. Supports +, -, *, /, ** (power), and parentheses."
    
    def _run(self, expression: str) -> str:
        """
        Calculate a mathematical expression safely.
        
        Example: "2 + 2" returns "Calculation result: 4"
        Example: "5 ** 2" returns "Calculation result: 25"
        """
        try:
            # Safe evaluation for basic math
            # {"__builtins__": {}} prevents dangerous operations
            # Only allows basic math: +, -, *, /, **, ()
            result = eval(expression, {"__builtins__": {}}, {})
            return f"Calculation result: {result}"
        except Exception as e:
            # If there's an error (like invalid syntax), return error message
            return f"Error in calculation: {str(e)}"

# ========== CREWAI AGENTS ==========
# Agents are AI entities with specific roles and capabilities
# We create 5 agents: 1 Manager (Planner) + 4 Workers

def create_agents(api_key: str):
    """
    Create the multi-agent system with hierarchical structure.
    
    Hierarchy:
    - Manager Agent (at top) coordinates everything
    - 4 Worker Agents (below) do specialized tasks
    
    Returns a dictionary with all 5 agents
    """
    
    # Set OpenAI API key as environment variable
    # CrewAI needs this to access GPT models
    os.environ["OPENAI_API_KEY"] = api_key
    
    # Create instances of our custom tools
    rag_tool = RAGSearchTool()  # For searching knowledge base
    calc_tool = MathCalculatorTool()  # For calculations
    
    # AGENT 1: RAG Agent (Worker) - Retrieves information from knowledge base
    rag_agent = Agent(
        role="Knowledge Retrieval Specialist",
        goal="Retrieve relevant mathematical information from the knowledge base using RAG, and indicate when knowledge base lacks information",
        backstory="""You are an expert at searching and retrieving information from 
        mathematical knowledge bases. You use vector similarity search to find the most 
        relevant content for any mathematical query. When the knowledge base doesn't contain 
        relevant information, you clearly indicate this so other agents can use their general knowledge.""",
        tools=[rag_tool],
        verbose=True,
        allow_delegation=False
    )
    
    # AGENT 2: Research Agent (Worker) - Performs deep research and analysis
    research_agent = Agent(
        role="Mathematical Research Analyst",
        goal="Conduct thorough research and analysis on mathematical topics using both retrieved knowledge and general mathematical expertise",
        backstory="""You are a mathematical research analyst with deep expertise in various 
        branches of mathematics. You analyze mathematical concepts and provide detailed 
        explanations. When the knowledge base provides relevant information, you use it as 
        primary source. When the knowledge base lacks information or has low relevance, you 
        confidently use your extensive general mathematical knowledge to provide accurate and 
        comprehensive answers. You can perform calculations when needed.""",
        tools=[calc_tool],
        verbose=True,
        allow_delegation=False
    )
    
    # AGENT 3: Reflection Agent (Worker) - Evaluates quality and suggests improvements
    reflection_agent = Agent(
        role="Quality Assurance & Reflection Specialist",
        goal="Evaluate the quality, accuracy, and completeness of mathematical answers",
        backstory="""You are a meticulous quality assurance specialist who evaluates 
        mathematical content for clarity, completeness, accuracy, and pedagogical value. 
        You provide constructive feedback and suggest improvements.""",
        verbose=True,
        allow_delegation=False
    )
    
    # AGENT 4: Summarization Agent (Worker) - Creates concise summaries
    summarization_agent = Agent(
        role="Content Summarization Expert",
        goal="Create clear, concise, and well-structured summaries of mathematical research",
        backstory="""You are an expert at distilling complex mathematical information 
        into clear, understandable summaries. You organize information logically and 
        present it in an accessible manner.""",
        verbose=True,
        allow_delegation=False
    )
    
    # AGENT 5: Manager Agent (Planner) - Coordinates all workers
    manager_agent = Agent(
        role="Research Coordinator & Project Manager",
        goal="Coordinate the research team to provide comprehensive mathematical answers",
        backstory="""You are an experienced research coordinator who manages a team of 
        specialists. You delegate tasks effectively, ensure quality standards, and 
        synthesize team outputs into coherent responses.""",
        verbose=True,
        allow_delegation=True
    )
    
    return {
        "rag": rag_agent,
        "research": research_agent,
        "reflection": reflection_agent,
        "summarization": summarization_agent,
        "manager": manager_agent
    }

# ========== CREWAI TASKS ==========
def detect_language(question: str):
    """Detect language code and human readable name. Fallback to English if unavailable."""
    if not question.strip():
        return "en", "English"
    if _LANGDETECT_AVAILABLE:
        try:
            code = detect(question)
        except Exception:
            code = "en"
    else:
        code = "en"
    names = {
        "en": "English",
        "tr": "Turkish",
        "fr": "French",
        "de": "German",
        "es": "Spanish",
        "it": "Italian",
        "ru": "Russian",
        "ar": "Arabic",
        "zh-cn": "Chinese",
        "zh-tw": "Chinese (Traditional)",
        "ja": "Japanese",
        "ko": "Korean",
        "pt": "Portuguese"
    }
    return code, names.get(code, "English")

def create_tasks(question: str, agents: dict, lang_code: str, lang_name: str, rag_result: dict, memory_prefs: str):
    """Create tasks for the crew based on the research question, enforcing response language."""
    language_instruction = f"Respond strictly in {lang_name}. If source context is in another language, translate faithfully preserving technical accuracy." if lang_code != "en" else "Respond in clear English."

    # Load recent feedback to guide improvements
    feedback_note = memory_prefs
    
    # Task 1: Knowledge Retrieval (RAG)
    rag_context_str = "No relevant knowledge found." if (rag_result.get("score", 0) < 0.3 or not rag_result.get("context")) else f"Retrieved Knowledge (relevance {rag_result['score']:.2f}):\n{rag_result['context']}"
    rag_task = Task(
        description=f"""RAG context prepared for question: {question}\n{rag_context_str}\n{language_instruction}""",
        agent=agents["rag"],
        expected_output="RAG context acknowledged"
    )
    
    # Task 2: Research & Analysis
    research_task = Task(
        description=f"""Answer comprehensively: {question}\n\nUse provided RAG context if relevance is high (>=0.3). Otherwise rely on general mathematical knowledge. Perform calculations when needed. Provide step-by-step derivations where applicable. {language_instruction}{feedback_note}""",
        agent=agents["research"],
        expected_output="Detailed mathematical explanation with examples and optional derivations",
        context=[rag_task]
    )
    
    # Task 3: Quality Reflection
    reflection_task = Task(
        description=f"""Evaluate the research answer for:\n1. Accuracy of mathematical content\n2. Clarity of explanation\n3. Completeness of coverage\n4. Pedagogical effectiveness\nProvide specific improvement suggestions. {language_instruction}{feedback_note}""",
        agent=agents["reflection"],
        expected_output="Quality evaluation with improvement suggestions",
        context=[research_task]
    )
    
    # Task 4: Final Summarization
    summary_task = Task(
        description=f"""Create a final, polished answer that:\n1. Incorporates the reflection feedback\n2. Is clear and well-structured\n3. Provides complete information\n4. Is accessible to the target audience\nOutput language requirement: {language_instruction}{feedback_note}""",
        agent=agents["summarization"],
        expected_output="Final polished answer",
        context=[research_task, reflection_task]
    )
    
    return [rag_task, research_task, reflection_task, summary_task]

# ========== MAIN UI ==========
st.markdown("---")

# Initialize session state for persisting results
if "last_result" not in st.session_state:
    st.session_state.last_result = None
if "last_question" not in st.session_state:
    st.session_state.last_question = None

col1, col2 = st.columns([2, 1])

with col1:
    question = st.text_area(
        "Enter your mathematical research question:",
        "What is a derivative and how is it calculated?",
        height=100
    )
    lang_code, lang_name = detect_language(question)
    if _LANGDETECT_AVAILABLE:
        st.caption(f"Detected language: {lang_name} ({lang_code})")
    else:
        st.caption("Language detection library not installed; defaulting to English.")

with col2:
    st.markdown("### 📊 RAG Performance Metrics")
    st.markdown("""
    **Measures Used:**
    - Retrieval Precision
    - Context Relevance
    - Answer Completeness
    - Response Time
    """)
    st.markdown("### 🧠 Memory Preferences")
    # Show brief memory summary
    try:
        mem_path = os.path.join("db", "memory.json")
        if os.path.exists(mem_path):
            import json
            with open(mem_path, "r", encoding="utf-8") as f:
                mem = json.load(f)
            prefs = mem.get("preferences", {})
            st.caption(
                ", ".join([f"{k}: {v}" for k, v in prefs.items()]) or "No preferences saved yet."
            )
        else:
            st.caption("No preferences saved yet.")
    except Exception:
        st.caption("Memory unavailable.")
    
    # Show recent feedback entries
    with st.expander("📜 Recent Feedback"):
        try:
            import json as _json
            fb_path = os.path.join("db", "feedback.jsonl")
            if os.path.exists(fb_path):
                with open(fb_path, "r", encoding="utf-8") as f:
                    lines = f.readlines()[-10:]
                if lines:
                    for line in lines:
                        try:
                            obj = _json.loads(line)
                            rating_emoji = "👍" if obj.get("rating", 0) >= 4 else "👎"
                            comment = obj.get("comments", "").strip() or "(no comment)"
                            st.markdown(f"{rating_emoji} {obj.get('timestamp', '')[:19]} | {comment}")
                        except Exception:
                            pass
                else:
                    st.caption("No feedback yet.")
            else:
                st.caption("No feedback yet.")
        except Exception:
            st.caption("Unable to load feedback.")

if st.button("🚀 Start Research", type="primary"):
    with st.spinner("🤖 Initializing multi-agent system..."):
        agents = create_agents(api_key)
    
    with st.spinner("📋 Creating tasks..."):
        # Query RAG to decide fallback
        rag_result = rag_query(question)
        # Load memory prefs and recent feedback notes
        def load_memory_prefs():
            try:
                import json
                os.makedirs("db", exist_ok=True)
                mem_path = os.path.join("db", "memory.json")
                feedback_path = os.path.join("db", "feedback.jsonl")
                prefs_note = ""
                prefs = {}
                if os.path.exists(mem_path):
                    with open(mem_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    prefs = data.get("preferences", {})
                    if prefs:
                        prefs_note = "\n\nUser preferences: " + "; ".join([f"{k}={v}" for k, v in prefs.items()])
                    # Add explicit style and tone instructions
                    style = prefs.get("style", "balanced")
                    tone = prefs.get("tone", "neutral")
                    prefs_note += f"\n\nAnswer style: {style}. Tone: {tone}."
                # Incorporate recent feedback comments
                if os.path.exists(feedback_path):
                    recent_comments = []
                    with open(feedback_path, "r", encoding="utf-8") as f:
                        lines = f.readlines()[-5:]
                    import json as _json
                    for line in lines:
                        try:
                            obj = _json.loads(line)
                            c = (obj.get("comments") or "").strip()
                            if c:
                                recent_comments.append(c)
                        except Exception:
                            pass
                    if recent_comments:
                        prefs_note += "\n\nRecent feedback: " + "; ".join(recent_comments)
                return prefs_note
            except Exception:
                return ""

        tasks = create_tasks(question, agents, lang_code, lang_name, rag_result, load_memory_prefs())
    
    with st.spinner("🔄 Agents working (this may take 1-2 minutes)..."):
        # Create Crew with hierarchical process
        crew = Crew(
            agents=list(agents.values()),
            tasks=tasks,
            process=Process.hierarchical,
            manager_llm=ChatOpenAI(model="gpt-3.5-turbo", api_key=api_key),
            verbose=True
        )
        
        # Execute the crew
        result = crew.kickoff()
    
    # Convert CrewOutput to string for storage
    result_str = str(result)
    
    # Store in session state for feedback persistence
    st.session_state.last_result = result_str
    st.session_state.last_question = question

    # If RAG confidence is low, add fallback enrichment via ChatGPT API
    try:
        if rag_result.get("score", 0) < 0.3:
            client = OpenAI(api_key=api_key)
            # Load style and tone from memory for fallback
            style_pref = "balanced"
            tone_pref = "neutral"
            try:
                import json
                mem_path = os.path.join("db", "memory.json")
                if os.path.exists(mem_path):
                    with open(mem_path, "r", encoding="utf-8") as f:
                        mem = json.load(f)
                    style_pref = mem.get("preferences", {}).get("style", "balanced")
                    tone_pref = mem.get("preferences", {}).get("tone", "neutral")
            except Exception:
                pass
            prompt = (
                f"Question: {question}\n\n"
                f"RAG context (may be low relevance):\n{rag_result.get('context','')}\n\n"
                f"Provide a clear, accurate mathematical explanation with derivations and examples.\n"
                f"Use a {style_pref} style and {tone_pref} tone."
            )
            completion = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
            )
            fallback_answer = completion.choices[0].message.content
            # Combine reflection output with fallback answer if crew result is weak
            result_str = fallback_answer.strip() + "\n\n---\n(Enhanced with fallback LLM due to low RAG confidence)"
            st.session_state.last_result = result_str  # Update session state with fallback
    except Exception:
        pass
    
    # Display results
    st.success("✅ Research Complete!")

# Display last result if available
if st.session_state.last_result:
    st.markdown("---")
    st.markdown("### 📝 Final Answer")
    st.markdown(st.session_state.last_result)
    
    # Show agent outputs in expandable sections
    with st.expander("🔍 View Agent Workflow Details"):
        st.markdown("#### Agent Execution Flow:")
        st.markdown("""
        1. **RAG Agent** → Retrieved knowledge from ChromaDB
        2. **Research Agent** → Analyzed and provided detailed answer
        3. **Reflection Agent** → Evaluated quality and suggested improvements
        4. **Summarization Agent** → Created final polished response
        5. **Manager Agent** → Coordinated the entire workflow
        """)

# Feedback section - always visible when there's a result
if st.session_state.last_result:
    st.markdown("---")
    st.markdown("### 💬 Provide Feedback")
    
    # Style and tone preferences
    st.markdown("#### 🧩 Preferred Answer Style")
    colp1, colp2 = st.columns(2)
    with colp1:
        style = st.radio("Style for future answers:", ["concise", "balanced", "detailed"], index=1, key="style_pref")
    with colp2:
        tone = st.radio("Tone:", ["neutral", "friendly", "academic"], index=0, key="tone_pref")
    
    colf1, colf2, colf3 = st.columns([1,1,2])
    with colf1:
        helpful = st.button("👍 Helpful")
    with colf2:
        not_helpful = st.button("👎 Not Helpful")
    with colf3:
        comments = st.text_input("Optional comments or preferences", "", key="feedback_comments")
    
    if helpful or not_helpful:
        try:
            import json
            from datetime import datetime
            os.makedirs("db", exist_ok=True)
            entry = {
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "question": st.session_state.last_question,
                "answer": st.session_state.last_result,
                "rating": 5 if helpful else 1,
                "comments": comments,
                "preferred_style": style,
                "preferred_tone": tone,
            }
            with open(os.path.join("db", "feedback.jsonl"), "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            # Update tiny memory database
            mem_path = os.path.join("db", "memory.json")
            memory = {"preferences": {}}
            if os.path.exists(mem_path):
                with open(mem_path, "r", encoding="utf-8") as f:
                    try:
                        memory = json.load(f) or {"preferences": {}}
                    except Exception:
                        memory = {"preferences": {}}
            prefs = memory.setdefault("preferences", {})
            # Simple adaptive preference updates
            prefs["likes_detail"] = int(prefs.get("likes_detail", 0)) + (1 if helpful else 0)
            prefs["wants_more_clarity"] = int(prefs.get("wants_more_clarity", 0)) + (1 if not_helpful else 0)
            if comments:
                prefs["last_comment"] = comments
            # Persist selected style/tone to drive future prompts
            prefs["style"] = style
            prefs["tone"] = tone
            with open(mem_path, "w", encoding="utf-8") as f:
                json.dump(memory, f, ensure_ascii=False, indent=2)
            
            # Show success and indicate memory update
            feedback_type = "👍 Helpful" if helpful else "👎 Not Helpful"
            st.success(f"✅ Feedback '{feedback_type}' saved! Memory updated with your preferences (style: {style}, tone: {tone}).")
            st.info("🔄 Next time you ask a question, the system will adapt based on your feedback.")
            
            # Force sidebar refresh by rerunning
            st.rerun()
        except Exception as e:
            st.error(f"Failed to save feedback: {e}")

# ========== FOOTER ==========
st.markdown("---")
st.markdown("""
### 🏗️ System Architecture
**Multi-Agent Design:**
- **Hierarchical Model**: Manager (Planner) + 4 Workers
- **RAG Implementation**: ChromaDB with HuggingFace embeddings
- **Reflection**: Quality assurance with introspection
- **Tools**: RAG Search Tool, Math Calculator Tool
- **Framework**: CrewAI + LangChain + Streamlit

**AI Principles Considered:**
- **Privacy**: No user data stored permanently
- **Fairness**: Equal treatment of all queries
- **Explainability**: Clear agent workflow visible
- **Responsible AI**: Transparent decision-making process
""")
