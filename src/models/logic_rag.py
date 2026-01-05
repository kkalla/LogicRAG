import json
import logging
import pdb
import time
from typing import List, Dict, Tuple, Any
from src.models.base_rag import BaseRAG
from src.utils.utils import get_response_with_retry, fix_json_response
from colorama import Fore, Style, init

# Initialize colorama
init()

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)

class LogicRAG(BaseRAG):
    
    def __init__(self, corpus_path: str = None, cache_dir: str = "./cache", filter_repeats: bool = False):
        """Initialize the LogicRAG system."""
        super().__init__(corpus_path, cache_dir)
        self.max_rounds = 3  # Default max rounds for iterative retrieval
        self.MODEL_NAME = "LogicRAG"
        self.filter_repeats = filter_repeats  # Option to filter repeated chunks across rounds
    
    def set_max_rounds(self, max_rounds: int):
        """Set the maximum number of retrieval rounds."""
        self.max_rounds = max_rounds
    
    def refine_summary_with_context(self, question: str, new_contexts: List[str],
                                  current_summary: str = "") -> str:
        """
        Generate a new summary or refine an existing one based on newly retrieved contexts.

        Args:
            question: The original question
            new_contexts: Newly retrieved context chunks
            current_summary: Current information summary (if any)

        Returns:
            A concise summary of all relevant information so far
        """
        try:
            context_text = "\n".join(new_contexts)
            logger.info(f"[refine_summary_with_context] Processing {len(new_contexts)} contexts")

            if not current_summary:
                # Generate initial summary
                prompt = f"""Please create a concise summary of the following information as it relates to answering this question:

Question: {question}

Information:
{context_text}

Your summary should:
1. Include all relevant facts that might help answer the question
2. Exclude irrelevant information
3. Be clear and concise
4. Preserve specific details, dates, numbers, and names that may be relevant

Summary:"""
            else:
                # Refine existing summary with new information
                prompt = f"""Please refine the following information summary using newly retrieved information.

Question: {question}

Current summary:
{current_summary}

New information:
{context_text}

Your refined summary should:
1. Integrate new relevant facts with the existing summary
2. Remove redundancies
3. Remain concise while preserving all important information
4. Prioritize information that helps answer the question
5. Maintain specific details, dates, numbers, and names that may be relevant

Refined summary:"""

            logger.info(f"[refine_summary_with_context] Calling get_response_with_retry...")
            summary = get_response_with_retry(prompt)
            logger.info(f"[refine_summary_with_context] Got response, length={len(summary)}")
            return summary
            
        except Exception as e:
            logger.error(f"{Fore.RED}Error generating/refining summary: {e}{Style.RESET_ALL}")
            # If error occurs, concatenate current summary with new contexts as fallback
            if current_summary:
                return f"{current_summary}\n\nNew information:\n{context_text}"
            return context_text
    
    def warm_up_analysis(self, question: str, info_summary: str) -> Dict:
        """
        This is a warm-up analysis, which is used to analyze if the question can be answered with simple fact retrieval, without any dependency analysis.

        Args:
            question: The original question
            info_summary: Current information summary

        Returns:
            Dictionary with analysis results
        """
        try:
            prompt = f"""You are a JSON API. Respond ONLY with valid JSON, no other text.

Question: {question}

Available Information:
{info_summary}

Analyze and return JSON with this exact structure:
{{
  "can_answer": true or false,
  "missing_info": "what information is missing",
  "subquery": "search query for missing info",
  "current_understanding": "summary of what we know",
  "dependencies": ["dependency1", "dependency2"],
  "missing_reason": "why info is missing (max 20 words)"
}}

Example if can_answer=true:
{{
  "can_answer": true,
  "missing_info": "",
  "subquery": "",
  "current_understanding": "The PARA method is a note-taking system...",
  "dependencies": [],
  "missing_reason": ""
}}

Example if can_answer=false:
{{
  "can_answer": false,
  "missing_info": "Need definition of PARA",
  "subquery": "What is PARA method",
  "current_understanding": "Question asks about PARA but no info found",
  "dependencies": ["PARA definition", "PARA components"],
  "missing_reason": "Context lacks PARA details"
}}

Respond ONLY with the JSON object:"""

            response = get_response_with_retry(prompt)
            
            # Clean up response to ensure it's valid JSON
            response = response.strip()
            
            # Remove any markdown code block markers
            response = response.replace('```json', '').replace('```', '')
            
            # Parse the cleaned response using fix_json_response
            result = fix_json_response(response)
            if result is None:
                return {
                    "can_answer": True,
                    "missing_info": "",
                    "subquery": question,
                    "current_understanding": "Failed to parse reflection response.",
                    "dependencies": ["Information relevant to the question"],
                    "missing_reason": "Parse error occurred"
                }
            
            # Validate required fields
            required_fields = ["can_answer", "missing_info", "subquery", "current_understanding"]
            if not all(field in result for field in required_fields):
                logger.error(f"{Fore.RED}Missing required fields in response: {response}{Style.RESET_ALL}")
                raise ValueError("Missing required fields")
            
            # Add default values for new interpretability fields if missing
            if "dependencies" not in result:
                result["dependencies"] = ["Information relevant to the question"]
            if "missing_reason" not in result:
                result["missing_reason"] = "Additional context needed" if not result["can_answer"] else "No missing information"
            
            # Ensure boolean type for can_answer
            result["can_answer"] = bool(result["can_answer"])
            
            # Ensure non-empty subquery
            if not result["subquery"]:
                result["subquery"] = question
            
            return result
                
        except Exception as e:
            logger.error(f"{Fore.RED}Error in analyze_dependency_graph: {e}{Style.RESET_ALL}")
            return {
                "can_answer": True,
                "missing_info": "",
                "subquery": question,
                "current_understanding": f"Error during analysis: {str(e)}",
                "dependencies": ["Information relevant to the question"],
                "missing_reason": "Analysis error occurred"
            }

    def dependency_aware_rag(self, question: str, info_summary: str, dependencies: List[str], idx: int) -> str:
        """
        similar to "self.analyze_dependency_graph" that analyzes whether the current information summary is sufficient to answer the question,
        this function analyzes whether the current information summary is sufficient to answer the question with the decomposed dependencies as references.

        And the function will answer whether the question can be answered, and if not, it will update the current query with dependencies as references.

        Args:
            question: str
            info_summary: str
            dependencies: List[str]
            idx: int
        """

        try:
            prompt = f"""You are a JSON API. Respond ONLY with valid JSON, no other text.

Question: {question}

Available Information:
{info_summary}

Decomposed dependencies:
{dependencies}

Current dependency to be answered:
{dependencies[idx]}

Return JSON with this exact structure:
{{"can_answer": true or false, "current_understanding": "summary of what we know"}}

Example if can_answer=true:
{{"can_answer": true, "current_understanding": "Based on the information, PARA is..."}}

Example if can_answer=false:
{{"can_answer": false, "current_understanding": "Still need more information about..."}}

Respond ONLY with the JSON object:"""

            response = get_response_with_retry(prompt)
            result = fix_json_response(response)

            # Handle None case from fix_json_response failure
            if result is None:
                return {
                    "can_answer": True,
                    "current_understanding": "Failed to parse dependency analysis response."
                }

            return result
        except Exception as e:
            logger.error(f"{Fore.RED}Error in dependency_aware_rag: {e}{Style.RESET_ALL}")
            return {
                "can_answer": True,
                "current_understanding": f"Error during analysis: {str(e)}",
            }

    def generate_answer(self, question: str, info_summary: str) -> str:
        """Generate final answer based on the information summary."""
        try:
            prompt = f"""You must give ONLY the direct answer in the most concise way possible. DO NOT explain or provide any additional context.
If the answer is a simple yes/no, just say "Yes." or "No."
If the answer is a name, just give the name.
If the answer is a date, just give the date.
If the answer is a number, just give the number.
If the answer requires a brief phrase, make it as concise as possible.

Question: {question}

Information Summary:
{info_summary}

Remember: Be concise - give ONLY the essential answer, nothing more.
Ans: """
            
            return get_response_with_retry(prompt)
        except Exception as e:
            logger.error(f"{Fore.RED}Error generating answer: {e}{Style.RESET_ALL}")
            return ""

    def _sort_dependencies(self, dependencies: List[str], query) -> List[Tuple]:
        """
        given a list of dependencies and the original query,
        sort the dependencies in a topological order, that is solving a dependency A relies on the solution of the dependent dependency B,
        then B should be before A in the sorted string.

        Args:
            dependencies: List[str]
            query: str

            
        For example, if the question is "What is the mayor of the capital of France?",
        the input dependencies for this question are:
        - The capital of France
        - The mayor of this capital

        Then the output should be:
        - The capital of France
        - The mayor of this capital

        there are two steps to solve this problem:
        1. generate the dependency pairs that dependency A relies on dependency B
        2. use graph-based algorithm to sort the dependencies in a topological order

        For example, answering the question "What is the mayor of the capital of France?"
        the input dependencies are:
        - The capital of France
        - The mayor of this capital

        Then the dependency pairs are:
        - [(1, 0)]
        because the mayor of the capital of France relies on the capital of France

        Then the topological order is computed by the self._topological_sort function, which is a graph-based algorithm. The output is a list of indices of the dependencies in the topological order.
        In this case, the output is:
        [0, 1]

        The sorted dependencies are thus:
        - The capital of France
        - The mayor of this capital
        """


        # Step 1: generate the dependency pairs by prompting LLMs
        prompt = f"""You are a JSON API. Respond ONLY with valid JSON, no other text.

Question: {query}

Dependencies: {dependencies}

Output dependency pairs where dependency A relies on dependency B.
Format: [A, B] means A depends on B.

Return JSON with this exact structure:
{{"dependency_pairs": [[0, 1], [2, 0]]}}

Example with dependencies:
- "The capital of France"
- "The mayor of this capital"

The mayor depends on knowing the capital first, so:
{{"dependency_pairs": [[1, 0]]}}

If no dependencies exist:
{{"dependency_pairs": []}}

Respond ONLY with the JSON object:"""

        logger.info(f"[_sort_dependencies] Calling get_response_with_retry for dependency pairs...")
        response = get_response_with_retry(prompt)
        logger.info(f"[_sort_dependencies] Got response: {response[:100] if response else 'empty'}")
        result = fix_json_response(response)

        # Handle None case from fix_json_response failure
        if result is None:
            logger.error("Failed to parse dependency pairs response, using empty pairs")
            dependency_pairs = []
        else:
            dependency_pairs = result.get("dependency_pairs", [])

        # Step 2: use graph-based algorithm to sort the dependencies in a topological order
        sorted_dependencies = self._topological_sort(dependencies, dependency_pairs)
        return sorted_dependencies

    @staticmethod
    def _topological_sort(dependencies: List[str], dependencies_pairs: List[Tuple[int, int]]) -> List[str]:
        """
        Use graph-based algorithm to sort the dependencies in a topological order.
        Args:
            dependencies: List[str]
            dependencies_pairs: List[Tuple[int, int]]
        Returns:
            List[str]
        """
        graph = {dep: [] for dep in dependencies}
        
        for dependent_idx, dependency_idx in dependencies_pairs:
            if dependent_idx < len(dependencies) and dependency_idx < len(dependencies):
                dependent = dependencies[dependent_idx]
                dependency = dependencies[dependency_idx]
                graph[dependency].append(dependent)  # dependency -> dependent
        
        visited = set()
        stack = []
        
        def dfs(node):
            if node in visited:
                return
            visited.add(node)
            for neighbor in graph[node]:
                dfs(neighbor)
            stack.append(node)

        for node in graph:
            if node not in visited:
                dfs(node)
        
        return stack[::-1]

    def _retrieve_with_filter(self, query: str, retrieved_chunks_set: set) -> list:
        """
        Retrieve top_k unique chunks not in retrieved_chunks_set. If not enough unique chunks, return as many as possible.
        """
        all_results = self.retrieve(query)
        unique_results = []
        idx = self.top_k
        # If not enough unique in top_k, keep expanding
        while len(unique_results) < self.top_k and idx <= len(self.corpus):
            # Expand retrieval window
            all_results = self.retrieve(query) if idx == self.top_k else self._retrieve_top_n(query, idx)
            unique_results = [chunk for chunk in all_results if chunk not in retrieved_chunks_set]
            idx += self.top_k
        return unique_results[:self.top_k]

    def _retrieve_top_n(self, query: str, n: int) -> list:
        """Retrieve top-n results for a query (helper for filtering)."""
        # Temporarily override top_k
        old_top_k = self.top_k
        self.top_k = n
        results = self.retrieve(query)
        self.top_k = old_top_k
        return results

    def answer_question(self, question: str) -> Tuple[str, List[str], int]:

        info_summary = "" 
        round_count = 0
        current_query = question
        retrieval_history = []
        last_contexts = []  
        dependency_analysis_history = []  
        retrieved_chunks_set = set() if self.filter_repeats else None  # Track retrieved chunks if filtering
        
        print(f"\n\n{Fore.CYAN}{self.MODEL_NAME} answering: {question}{Style.RESET_ALL}\n\n")

        #===============================================
        #== Stage 1: warm up retrieval ==
        print(f"[DEBUG] Starting Stage 1: warm up retrieval...")
        if self.filter_repeats:
            new_contexts = self._retrieve_with_filter(question, retrieved_chunks_set)
            for chunk in new_contexts:
                retrieved_chunks_set.add(chunk)
        else:
            print(f"[DEBUG] Retrieving contexts for question...")
            new_contexts = self.retrieve(question)
            print(f"[DEBUG] Retrieved {len(new_contexts)} contexts")
        last_contexts = new_contexts
        print(f"[DEBUG] Refining summary with context...")
        info_summary = self.refine_summary_with_context(
            question,
            new_contexts,
            info_summary
        )
        print(f"[DEBUG] Summary refined. Starting warm_up_analysis...")

        analysis = self.warm_up_analysis(question, info_summary)
        print(f"[DEBUG] warm_up_analysis complete. can_answer={analysis.get('can_answer')}")

        if analysis["can_answer"]:
            # In this case, the question can be answered with simple fact retrieval, without any dependency analysis
            print(f"Warm-up analysis indicate the question can be answered with simple fact retrieval, without any dependency analysis.")
            answer = self.generate_answer(question, info_summary)
            # Reset dependency analysis history for simple questions
            self.last_dependency_analysis = []
            return answer, last_contexts, round_count
        else:
            logger.info(f"Warm-up analysis indicate the requirement of deeper reasoning-enhanced RAG. Now perform analysis with logical dependency graph.")
            logger.info(f"Dependencies: {', '.join(analysis.get('dependencies', []))}")
            print(f"[DEBUG] Starting Stage 2: dependency analysis...")

            # sort the dependencies, by first constructing the dependency graphs, then use topological sort to get the sorted dependencies
            print(f"[DEBUG] Calling _sort_dependencies...")
            sorted_dependencies = self._sort_dependencies(analysis["dependencies"], question)
            print(f"[DEBUG] _sort_dependencies returned: {sorted_dependencies}")
            dependency_analysis_history.append({"sorted_dependencies": sorted_dependencies})
            logger.info(f"Sorted dependencies: {sorted_dependencies}\n\n")
        #===============================================
        #== Stage 2: agentic iterative retrieval ==
        idx = 0 # used to track the current dependency index
        print(f"[DEBUG] Starting Stage 2 loop. max_rounds={self.max_rounds}, dependencies={len(sorted_dependencies)}")

        while round_count < self.max_rounds and idx < len(sorted_dependencies):
            round_count += 1
            print(f"[DEBUG] Round {round_count}: idx={idx}, max_rounds={self.max_rounds}")

            current_query = sorted_dependencies[idx]
            print(f"[DEBUG] Current query: {current_query}")
            if self.filter_repeats:
                new_contexts = self._retrieve_with_filter(current_query, retrieved_chunks_set)
                for chunk in new_contexts:
                    retrieved_chunks_set.add(chunk)
            else:
                print(f"[DEBUG] Retrieving for current query...")
                new_contexts = self.retrieve(current_query)
                print(f"[DEBUG] Got {len(new_contexts)} contexts")
            last_contexts = new_contexts  # Save current contexts


            # Generate or refine information summary with new contexts
            print(f"[DEBUG] Refining summary with new contexts...")
            info_summary = self.refine_summary_with_context(
                question,
                new_contexts,
                info_summary
            )
            print(f"[DEBUG] Summary refined")

            logger.info(f"Agentic retrieval at round {round_count}")
            logger.info(f"current query: {current_query}")

            print(f"[DEBUG] Calling dependency_aware_rag...")
            analysis = self.dependency_aware_rag(question, info_summary, sorted_dependencies, idx)
            print(f"[DEBUG] dependency_aware_rag returned: can_answer={analysis.get('can_answer')}")

            retrieval_history.append({
                "round": round_count,
                "query": current_query,
                "contexts": new_contexts,
            }) 

            dependency_analysis_history.append({
                "round": round_count,
                "query": current_query,
                "analysis": analysis
            })

            if analysis["can_answer"]:
                # Generate and return final answer
                print(f"[DEBUG] Generating final answer...")
                answer = self.generate_answer(question, info_summary)
                print(f"[DEBUG] Final answer: {answer}")
                # Store dependency analysis history for evaluation access
                self.last_dependency_analysis = dependency_analysis_history
                # We return the last retrieved contexts for evaluation purposes
                return answer, last_contexts, round_count
            else:
                idx += 1

        # If max rounds reached, generate best possible answer
        print(f"[DEBUG] Exiting loop. round_count={round_count}, idx={idx}, max_rounds={self.max_rounds}")
        logger.info(f"Reached maximum rounds ({self.max_rounds}). Generating final answer...")
        print(f"[DEBUG] Generating best possible answer...")
        answer = self.generate_answer(question, info_summary)
        print(f"[DEBUG] Final answer: {answer}")
        # Store dependency analysis history for evaluation access
        self.last_dependency_analysis = dependency_analysis_history
        return answer, last_contexts, round_count