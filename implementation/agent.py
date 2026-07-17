import re
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
import subprocess
import os
from typing import Dict, Any, Optional
from src.llm_client import BaseLLMClient
from src.prompt_manager import PromptManager
from src.config_loader import ConfigLoader

class OptMinerAgent:
    """
    Opt-Miner Agent that runs the multi-turn information seeking loop.
    Supports dynamic Arxiv queries, solver verification via Gurobi, and
    LLM prompt formatting.
    """
    def __init__(self, llm_client: Optional[BaseLLMClient] = None, 
                 prompt_manager: Optional[PromptManager] = None, 
                 logger: Optional[Any] = None,
                 max_turns: int = 3):
        self.llm_client = llm_client
        self.prompt_manager = prompt_manager
        self.logger = logger
        self.max_turns = max_turns
        
        # Load API configs
        config_loader = ConfigLoader()
        try:
            api_config = config_loader.load_config("apis")
            arxiv_config = api_config.get("arxiv_api", {})
            self.arxiv_url = arxiv_config.get("url", "http://export.arxiv.org/api/query")
            self.max_search_results = arxiv_config.get("max_results", 10)
        except Exception:
            self.arxiv_url = "http://export.arxiv.org/api/query"
            self.max_search_results = 10

    def arxiv_search(self, query: str) -> str:
        """
        Retrieves papers from Arxiv based on a search query.
        Returns a formatted string of the results.
        """
        try:
            url = f"{self.arxiv_url}?search_query=all:{urllib.parse.quote(query)}&start=0&max_results={self.max_search_results}"
            if self.logger:
                self.logger.info(f"Querying Arxiv URL: {url}")
            response = urllib.request.urlopen(url)
            xml_data = response.read()
            root = ET.fromstring(xml_data)
            
            results = []
            ns = {'atom': 'http://www.w3.org/2005/Atom'}
            for entry in root.findall('atom:entry', ns):
                title = entry.find('atom:title', ns).text.strip()
                summary = entry.find('atom:summary', ns).text.strip()
                results.append(f"Title: {title}\nSummary: {summary}")
                
            if not results:
                return "No results found."
            return "\n\n---\n\n".join(results)
        except Exception as e:
            if self.logger:
                self.logger.error(f"Arxiv search query failed: {e}")
            return f"Search failed: {str(e)}"

    def execute_python_code(self, code: str) -> str:
        """
        Executes python code using Gurobi and returns stdout/stderr logs.
        """
        file_path = "temp_gurobi_solver.py"
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(code)
            
        try:
            result = subprocess.run(["python", file_path], capture_output=True, text=True, timeout=30)
            output = result.stdout
            if result.stderr:
                output += "\nErrors:\n" + result.stderr
            return output
        except subprocess.TimeoutExpired:
            return "Execution timeout (exceeded 30 seconds)."
        except Exception as e:
            return f"Execution failed: {str(e)}"
        finally:
            if os.path.exists(file_path):
                os.remove(file_path)

    def run_inference_loop(self, problem_description: str, max_turns: Optional[int] = None) -> str:
        """
        Runs the multi-turn agentic information-seeking research loop.
        """
        turns = max_turns or self.max_turns
        
        # 1. Format the initial prompt using prompt_manager
        if self.prompt_manager:
            system_prompt = self.prompt_manager.get_prompt(
                "Prompt_research", 
                problem_description=problem_description, 
                max_turns=turns
            )
        else:
            system_prompt = f"Goal: Solve optimization problem: {problem_description}."

        context = f"Problem: {problem_description}\n"
        
        if self.logger:
            self.logger.info(f"Starting agent inference loop for problem: {problem_description[:100]}...")
            
        for turn in range(turns):
            if not self.llm_client:
                raise ValueError("OptMinerAgent requires a valid LLM client.")
            
            if self.logger:
                self.logger.info(f"Querying LLM, turn {turn+1}/{turns}...")
            response = self.llm_client.generate(prompt=context, system_prompt=system_prompt)
            context += f"\nAgent:\n{response}\n"
            
            # Check for search tags
            search_match = re.search(r"<search>(.*?)</search>", response, re.DOTALL)
            if search_match:
                query = search_match.group(1).strip()
                if self.logger:
                    self.logger.info(f"Executing Arxiv search query: '{query}'")
                search_result = self.arxiv_search(query)
                context += f"\n<result>\n{search_result}\n</result>\n"
                continue
                
            # Check for python code tags
            python_match = re.search(r"<python>(.*?)</python>", response, re.DOTALL)
            if python_match:
                code = python_match.group(1).strip()
                if self.logger:
                    self.logger.info("Executing Python Gurobi code block...")
                exec_result = self.execute_python_code(code)
                context += f"\n<result>\n{exec_result}\n</result>\n"
                
                # Check if execution succeeded (Optimal solution found)
                if "Optimal solution found" in exec_result or "Errors:" not in exec_result:
                    if self.logger:
                        self.logger.info("Optimal solution found or code executed successfully. Ending loop.")
                    break
                    
        return context
