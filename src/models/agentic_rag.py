import json
import logging
import time
from typing import List, Dict, Tuple, Any
from src.models.base_rag import BaseRAG
from src.utils.utils import get_response_with_retry, REFLECTION_PROMPT, fix_json_response
from colorama import Fore, Style, init

# Initialize colorama
init()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AgenticRAG(BaseRAG):
    """
    AgenticRAG implements an agentic approach to retrieval-augmented generation.
    It uses iterative reflection and retrieval to improve answer quality.
    """
    
    def __init__(self, corpus_path: str = None, cache_dir: str = "./cache"):
        """Initialize the AgenticRAG system."""
        super().__init__(corpus_path, cache_dir)
        self.max_rounds = 3  # Default max rounds for iterative retrieval
        self.MODEL_NAME = "AgenticRAG"
    
    def set_max_rounds(self, max_rounds: int):
        """Set the maximum number of retrieval rounds."""
        self.max_rounds = max_rounds
    
    def analyze_completeness(self, question: str, context: List[str]) -> Dict:
        """Analyze if the retrieved context is sufficient to answer the question."""
        try:
            context_text = "\n".join(context)
            prompt = f"""Question: {question}

Retrieved Context:
{context_text}

{REFLECTION_PROMPT}"""
            
            response = get_response_with_retry(prompt)
            
            # Clean up response to ensure it's valid JSON
            response = response.strip()
            
            # Remove any markdown code block markers
            response = response.replace('```json', '').replace('```', '')
            
            # Try to find JSON-like content within the response
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                response = json_match.group()
            
            # Parse the cleaned response using fix_json_response
            result = fix_json_response(response)
            if result is None:
                return {
                    "can_answer": True,
                    "missing_info": "",
                    "subquery": question,
                    "current_understanding": "Failed to parse reflection response."
                }
            
            # Validate required fields
            required_fields = ["can_answer", "missing_info", "subquery", "current_understanding"]
            if not all(field in result for field in required_fields):
                logger.error(f"{Fore.RED}Missing required fields in response: {response}{Style.RESET_ALL}")
                raise ValueError("Missing required fields")
            
            # Ensure boolean type for can_answer
            result["can_answer"] = bool(result["can_answer"])
            
            # Ensure non-empty subquery
            if not result["subquery"]:
                result["subquery"] = question
            
            return result
                
        except Exception as e:
            logger.error(f"{Fore.RED}Error in analyze_completeness: {e}{Style.RESET_ALL}")
            return {
                "can_answer": True,
                "missing_info": "",
                "subquery": question,
                "current_understanding": f"Error during analysis: {str(e)}"
            }

    def generate_answer(self, question: str, context: List[str], 
                       current_understanding: str = "") -> str:
        """Generate final answer based on all retrieved context."""
        try:
            context_text = "\n".join(context)
            current_understanding_text = f"\nCurrent Understanding: {current_understanding}" if current_understanding else ""
            
            prompt = f"""You must give ONLY the direct answer in the most concise way possible. DO NOT explain or provide any additional context.
If the answer is a simple yes/no, just say "Yes." or "No."
If the answer is a name, just give the name.
If the answer is a date, just give the date.
If the answer is a number, just give the number.
If the answer requires a brief phrase, make it as concise as possible.

Question: {question}{current_understanding_text}

Context:
{context_text}

Remember: Be as concise as vanilla RAG - give ONLY the essential answer, nothing more.
Ans: """
            
            return get_response_with_retry(prompt)
        except Exception as e:
            logger.error(f"{Fore.RED}Error generating answer: {e}{Style.RESET_ALL}")
            return ""

    def answer_question(self, question: str) -> Tuple[str, List[str], int]:
        """Answer question with iterative retrieval and reflection."""
        all_contexts = []
        round_count = 0
        current_query = question
        retrieval_history = []

        logger.info(f"\n\n{Fore.CYAN}{self.MODEL_NAME} answering: {question}{Style.RESET_ALL}\n\n")
        
        while round_count < self.max_rounds:
            round_count += 1
            logger.info(f"Retrieval round {round_count}")
            
            # Retrieve relevant contexts
            new_contexts = self.retrieve(current_query)
            all_contexts.extend(new_contexts)
            
            # Remove duplicates while preserving order
            seen = set()
            all_contexts = [x for x in all_contexts if not (x in seen or seen.add(x))]
            
            # Record retrieval history
            retrieval_history.append({
                "round": round_count,
                "query": current_query,
                "contexts": new_contexts
            })
            
            # Analyze completeness
            analysis = self.analyze_completeness(question, all_contexts)
            
            if analysis["can_answer"]:
                # Generate and return final answer
                answer = self.generate_answer(
                    question, 
                    all_contexts,
                    analysis["current_understanding"]
                )
                return answer, all_contexts, round_count
            
            # Update query for next round
            current_query = analysis["subquery"]
            logger.info(f"Generated subquery: {current_query}")
        
        # If max rounds reached, generate best possible answer
        answer = self.generate_answer(
            question,
            all_contexts,
            "Note: Maximum retrieval rounds reached. Providing best possible answer."
        )
        return answer, all_contexts, round_count 