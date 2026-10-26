"""CrewAI Agent and Task definitions for the Agentic RAG Coder."""

import os
from pathlib import Path
from typing import Any, Dict, Optional
try:
    from dotenv import load_dotenv
    load_dotenv(override=True)
except ImportError:
    pass

from src.tools.vector_query_tool import VectorStoreQueryTool
from src.tools.web_search_tool import WebSearchTool
from src.tools.code_exec_tool import CodeExecutionTool, FileWriteTool


def load_yaml_config(file_path: str) -> Dict[str, Any]:
    """Load configuration dictionary from YAML file."""
    path = Path(file_path)
    if not path.exists():
        return {}
    try:
        import yaml
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except ImportError:
        return {}


def get_llm(provider: Optional[str] = None):
    """
    Instantiate LLM for CrewAI agents based on configuration.
    Supports Groq, OpenAI, and Anthropic.
    """
    from crewai import LLM

    provider = (provider or os.getenv("DEFAULT_LLM_PROVIDER", "groq")).lower().strip()

    if provider == "groq":
        api_key = os.getenv("GROQ_API_KEY", "").strip()
        model_name = os.getenv("GROQ_MODEL_NAME", "groq/llama-3.3-70b-versatile").strip()
        if not model_name.startswith("groq/"):
            model_name = f"groq/{model_name}"
        return LLM(model=model_name, api_key=api_key, temperature=0.1)

    elif provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        model_name = os.getenv("OPENAI_MODEL_NAME", "gpt-4o-mini").strip()
        return LLM(model=model_name, api_key=api_key, temperature=0.1)

    elif provider == "anthropic":
        api_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
        model_name = os.getenv("ANTHROPIC_MODEL_NAME", "claude-3-5-sonnet-20241022").strip()
        return LLM(model=model_name, api_key=api_key, temperature=0.1)

    else:
        # Fallback to Groq or OpenAI if available
        if os.getenv("GROQ_API_KEY"):
            return LLM(model="groq/llama-3.3-70b-versatile", api_key=os.getenv("GROQ_API_KEY"), temperature=0.1)
        return LLM(model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"), temperature=0.1)


class CrewFactory:
    """Factory to construct agents, tasks, and crews from config files."""

    def __init__(
        self,
        agents_config_path: str = "./config/agents.yaml",
        tasks_config_path: str = "./config/tasks.yaml",
        settings_config_path: str = "./config/settings.yaml",
    ):
        self.agents_config = load_yaml_config(agents_config_path)
        self.tasks_config = load_yaml_config(tasks_config_path)
        self.settings = load_yaml_config(settings_config_path)
        self.llm = get_llm()

    def build_indexer_agent(self):
        """Build the Indexer Agent (Local Knowledge Curator)."""
        from crewai import Agent

        cfg = self.agents_config.get("indexer_agent", {})
        return Agent(
            role=cfg.get("role", "Local Knowledge Curator"),
            goal=cfg.get("goal", "Scan, chunk, and index local files into ChromaDB."),
            backstory=cfg.get("backstory", "An expert in organizing codebases and indexing documentation."),
            llm=self.llm,
            verbose=True,
            allow_delegation=False,
        )

    def build_retriever_agent(self):
        """Build the Retriever Agent (Context Retrieval Specialist)."""
        from crewai import Agent

        cfg = self.agents_config.get("retriever_agent", {})
        top_k = int(self.settings.get("vector_store", {}).get("top_k", 4))
        threshold = float(self.settings.get("vector_store", {}).get("similarity_threshold", 0.65))

        query_tool = VectorStoreQueryTool(
            top_k=top_k,
            similarity_threshold=threshold,
        )

        return Agent(
            role=cfg.get("role", "Context Retrieval Specialist"),
            goal=cfg.get("goal", "Retrieve relevant chunks from ChromaDB for coding queries."),
            backstory=cfg.get("backstory", "An expert search engineer ranking semantic code context."),
            tools=[query_tool],
            llm=self.llm,
            verbose=True,
            allow_delegation=False,
        )

    def build_researcher_agent(self):
        """Build the Researcher Agent (External Knowledge Fetcher)."""
        from crewai import Agent

        cfg = self.agents_config.get("researcher_agent", {})
        search_tool = WebSearchTool(max_results=3)

        return Agent(
            role=cfg.get("role", "External Knowledge Fetcher"),
            goal=cfg.get("goal", "Search external web APIs for updated docs and APIs when local context is absent."),
            backstory=cfg.get("backstory", "A researcher uncovering official developer documentation."),
            tools=[search_tool],
            llm=self.llm,
            verbose=True,
            allow_delegation=False,
        )

    def build_coding_agent(self):
        """Build the Coding Agent (Senior Software Engineer)."""
        from crewai import Agent

        cfg = self.agents_config.get("coding_agent", {})
        code_tool = CodeExecutionTool()
        file_tool = FileWriteTool()

        return Agent(
            role=cfg.get("role", "Senior Software Engineer"),
            goal=cfg.get("goal", "Write, explain, and debug code grounded in retrieved context with citations."),
            backstory=cfg.get("backstory", "A senior architect delivering clean code and citing references."),
            tools=[code_tool, file_tool],
            llm=self.llm,
            verbose=True,
            allow_delegation=False,
        )

    def build_manager_agent(self):
        """Build the Orchestrator / Manager Agent (Task Router)."""
        from crewai import Agent

        cfg = self.agents_config.get("manager_agent", {})
        return Agent(
            role=cfg.get("role", "Task Router & Technical Project Lead"),
            goal=cfg.get("goal", "Route tasks between local RAG retrieval and web search, then coordinate synthesis."),
            backstory=cfg.get("backstory", "An engineering director coordinating specialist agents."),
            llm=self.llm,
            verbose=True,
            allow_delegation=True,
        )

    def create_coding_crew(self, user_query: str, session_context: str = ""):
        """
        Create a hierarchical Crew with Manager Agent delegating to Retriever,
        Researcher, and Coding agents.
        """
        from crewai import Crew, Process, Task

        retriever = self.build_retriever_agent()
        researcher = self.build_researcher_agent()
        coder = self.build_coding_agent()
        manager = self.build_manager_agent()

        full_query = user_query
        if session_context:
            full_query = f"{user_query}\n\n[Session Context]\n{session_context}"

        # 1. Retrieval Task
        t_retrieve_cfg = self.tasks_config.get("retrieve_task", {})
        t_retrieve = Task(
            description=t_retrieve_cfg.get(
                "description",
                "Query local ChromaDB for user query: '{user_query}'."
            ).format(user_query=full_query),
            expected_output=t_retrieve_cfg.get(
                "expected_output",
                "Retrieved code chunks, source paths, and relevance summary."
            ),
            agent=retriever,
        )

        # 2. Research Task
        t_research_cfg = self.tasks_config.get("research_task", {})
        t_research = Task(
            description=t_research_cfg.get(
                "description",
                "Search external web for '{user_query}' if local context is insufficient."
            ).format(user_query=full_query),
            expected_output=t_research_cfg.get(
                "expected_output",
                "External documentation findings or confirmation that local context was sufficient."
            ),
            agent=researcher,
        )

        # 3. Coding Task
        t_coding_cfg = self.tasks_config.get("coding_task", {})
        t_coding = Task(
            description=t_coding_cfg.get(
                "description",
                "Synthesize code solution with citations for '{user_query}'."
            ).format(user_query=full_query),
            expected_output=t_coding_cfg.get(
                "expected_output",
                "Markdown solution with explanation, code blocks, and Sources & Citations section."
            ),
            agent=coder,
        )

        memory_enabled = os.getenv("MEMORY_ENABLED", "false").lower() == "true"

        crew = Crew(
            agents=[retriever, researcher, coder],
            tasks=[t_retrieve, t_research, t_coding],
            process=Process.hierarchical,
            manager_agent=manager,
            verbose=True,
            memory=memory_enabled,
        )

        return crew
