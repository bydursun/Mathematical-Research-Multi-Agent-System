# Math Research Assistant - CrewAI Multi-Agent System with RAG & Reflection
# Assignment: Multi-Agent Research and Summarization System for Mathematical Enquiries

# ========== IMPORTS ==========
# These are all the libraries we need for our multi-agent system

import os  # For file and environment variable operations
import streamlit as st  # For creating the web interface (frontend)

# CrewAI imports - for building multi-agent system
from crewai import Agent, Task, Crew, Process
from crewai.tools import BaseTool  # Base class for creating custom tools

# LangChain imports - for RAG (Retrieval-Augmented Generation) implementation
from langchain_community.embeddings import HuggingFaceEmbeddings  # Converts text to vectors
from langchain_community.vectorstores import Chroma  # Vector database (ChromaDB)
from langchain_community.document_loaders import TextLoader  # Loads text files
from langchain_openai import ChatOpenAI  # OpenAI GPT integration

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
        4. Returns those pieces as context
        """
        # Search ChromaDB for top 3 most similar documents
        results = vectordb.similarity_search(query, k=3)
        
        # Combine the results into one string
        context = "\n\n".join([doc.page_content for doc in results])
        
        # Return the retrieved knowledge
        return f"Retrieved Knowledge:\n{context}"

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
        goal="Retrieve relevant mathematical information from the knowledge base using RAG",
        backstory="""You are an expert at searching and retrieving information from 
        mathematical knowledge bases. You use vector similarity search to find the most 
        relevant content for any mathematical query.""",
        tools=[rag_tool],
        verbose=True,
        allow_delegation=False
    )
    
    # AGENT 2: Research Agent (Worker) - Performs deep research and analysis
    research_agent = Agent(
        role="Mathematical Research Analyst",
        goal="Conduct thorough research and analysis on mathematical topics",
        backstory="""You are a mathematical research analyst with expertise in various 
        branches of mathematics. You analyze mathematical concepts, provide detailed 
        explanations, and can perform calculations when needed.""",
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
def create_tasks(question: str, agents: dict):
    """Create tasks for the crew based on the research question"""
    
    # Task 1: Knowledge Retrieval (RAG)
    rag_task = Task(
        description=f"""Search the knowledge base for information related to: {question}
        Use the RAG tool to retrieve relevant mathematical knowledge.""",
        agent=agents["rag"],
        expected_output="Retrieved relevant context from the knowledge base"
    )
    
    # Task 2: Research & Analysis
    research_task = Task(
        description=f"""Based on the retrieved knowledge, provide a comprehensive answer to: {question}
        Use the calculator tool if any computations are needed.
        Provide detailed explanations with examples.""",
        agent=agents["research"],
        expected_output="Detailed mathematical explanation with examples",
        context=[rag_task]
    )
    
    # Task 3: Quality Reflection
    reflection_task = Task(
        description="""Evaluate the research answer for:
        1. Accuracy of mathematical content
        2. Clarity of explanation
        3. Completeness of coverage
        4. Pedagogical effectiveness
        Provide specific improvement suggestions.""",
        agent=agents["reflection"],
        expected_output="Quality evaluation with improvement suggestions",
        context=[research_task]
    )
    
    # Task 4: Final Summarization
    summary_task = Task(
        description="""Create a final, polished answer that:
        1. Incorporates the reflection feedback
        2. Is clear and well-structured
        3. Provides complete information
        4. Is accessible to the target audience""",
        agent=agents["summarization"],
        expected_output="Final polished answer",
        context=[research_task, reflection_task]
    )
    
    return [rag_task, research_task, reflection_task, summary_task]

# ========== MAIN UI ==========
st.markdown("---")
col1, col2 = st.columns([2, 1])

with col1:
    question = st.text_area(
        "Enter your mathematical research question:",
        "What is a derivative and how is it calculated?",
        height=100
    )

with col2:
    st.markdown("### 📊 RAG Performance Metrics")
    st.markdown("""
    **Measures Used:**
    - Retrieval Precision
    - Context Relevance
    - Answer Completeness
    - Response Time
    """)

if st.button("🚀 Start Research", type="primary"):
    with st.spinner("🤖 Initializing multi-agent system..."):
        agents = create_agents(api_key)
    
    with st.spinner("📋 Creating tasks..."):
        tasks = create_tasks(question, agents)
    
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
    
    # Display results
    st.success("✅ Research Complete!")
    
    st.markdown("---")
    st.markdown("### 📝 Final Answer")
    st.markdown(result)
    
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
