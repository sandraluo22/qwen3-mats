"""
GSM8K Chain of Thought Analysis with Qwen3
This script loads GSM8K dataset, runs Qwen3 on problems, extracts chain of thought,
and performs resampling analysis.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import json
import re
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import os
from datetime import datetime


class Qwen3Inference:
    """Wrapper for Qwen3 model inference."""
    
    def __init__(self, model_name: str = "Qwen/Qwen3-4B-Thinking-2507"):
        """Initialize the model and tokenizer."""
        print(f"Loading model {model_name}...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None,
        )
        
        if self.device == "cpu":
            self.model = self.model.to(self.device)
        
        print("Model loaded successfully!")
    
    def generate(self, prompt: str, max_new_tokens: int = 512, temperature: float = 0.7) -> str:
        """Generate a response from the model."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        
        response = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        return response.strip()


class ChainOfThoughtExtractor:
    """Extract chain of thought sentences from model responses."""
    
    @staticmethod
    def extract_sentences(text: str) -> List[str]:
        """Extract sentences from the chain of thought."""
        # Remove thinking tags if present
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
        
        # Split by sentence endings, but preserve numbers with periods
        sentences = []
        current_sentence = ""
        
        # Simple sentence splitting - split on . ! ? followed by space or newline
        parts = re.split(r'([.!?]\s+)', text)
        
        for i, part in enumerate(parts):
            current_sentence += part
            if i % 2 == 1:  # This is a sentence ending
                sentence = current_sentence.strip()
                if sentence and len(sentence) > 3:  # Filter very short sentences
                    sentences.append(sentence)
                current_sentence = ""
        
        # Add remaining text if any
        if current_sentence.strip():
            sentences.append(current_sentence.strip())
        
        return sentences
    
    @staticmethod
    def extract_final_answer(text: str) -> Optional[str]:
        """Extract the final answer from the response."""
        # Look for patterns like "The answer is X" or "Answer: X" or just a number at the end
        patterns = [
            r'(?:the answer is|answer is|answer:|final answer:|answer:)\s*([^\n.]+)',
            r'(?:therefore|thus|so)\s+([^\n.]+)',
            r'\b\d+(?:\.\d+)?\b(?=\s*$)',  # Number at the end
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text.lower())
            if match:
                answer = match.group(1) if match.lastindex else match.group(0)
                # Extract just the number if possible
                numbers = re.findall(r'\d+(?:\.\d+)?', answer)
                if numbers:
                    return numbers[-1]
                return answer.strip()
        
        # If no pattern found, try to extract the last number
        numbers = re.findall(r'\d+(?:\.\d+)?', text)
        if numbers:
            return numbers[-1]
        
        return None


class GSM8KAnalyzer:
    """Main analyzer for GSM8K problems."""
    
    def __init__(self, model_name: str = "Qwen/Qwen3-4B-Thinking-2507"):
        self.llm = Qwen3Inference(model_name)
        self.extractor = ChainOfThoughtExtractor()
        self.results = []
    
    def load_gsm8k(self, num_problems: int = 50) -> List[Dict]:
        """Load GSM8K dataset and return specified number of problems."""
        print(f"Loading GSM8K dataset (taking {num_problems} problems)...")
        dataset = load_dataset("gsm8k", "main", split="test")
        
        problems = []
        for i, item in enumerate(dataset):
            if i >= num_problems:
                break
            problems.append({
                'id': i,
                'problem': item['question'],
                'correct_answer': item['answer'].split('####')[1].strip() if '####' in item['answer'] else item['answer'],
                'solution': item['answer'],
            })
        
        print(f"Loaded {len(problems)} problems from GSM8K")
        return problems
    
    def create_prompt(self, problem: str, partial_cot: Optional[List[str]] = None) -> str:
        """Create a prompt for the model."""
        base_prompt = f"Solve the following math problem step by step.\n\nProblem: {problem}\n\nSolution:"
        
        if partial_cot:
            cot_text = " ".join(partial_cot)
            base_prompt = f"Solve the following math problem step by step. Use at most 15 sentences.\n\nProblem: {problem}\n\nSolution: {cot_text}"
        
        return base_prompt
    
    def resample_until_stable(
        self, 
        problem: str, 
        partial_cot: List[str], 
        max_resamples: int = 10
    ) -> Tuple[List[str], str]:
        """
        Resample the question given partial chain of thought 10 times.
        Returns: (list of all responses, most common answer)
        """
        prompt = self.create_prompt(problem, partial_cot)
        responses = []
        answers = []
        
        for i in range(max_resamples):
            response = self.llm.generate(prompt, max_new_tokens=256, temperature=0.7)
            responses.append(response)
            answer = self.extractor.extract_final_answer(response)
            if answer:
                answers.append(answer)
        
        # Return the most common answer
        if answers:
            answer_counts = defaultdict(int)
            for ans in answers:
                answer_counts[ans] += 1
            stabilized_answer = max(answer_counts.items(), key=lambda x: x[1])[0]
            return responses, stabilized_answer
        
        return responses, None
    
    def force_final_answer(self, problem: str, partial_cot: List[str]) -> str:
        """Force a final answer from partial chain of thought."""
        prompt = self.create_prompt(problem, partial_cot)
        # Add instruction to provide final answer
        prompt += "\n\nBased on the above reasoning, what is the final answer? Provide only the numerical answer."
        
        response = self.llm.generate(prompt, max_new_tokens=50, temperature=0.3)
        answer = self.extractor.extract_final_answer(response)
        return answer if answer else response.strip()
    
    def analyze_problem(self, problem_data: Dict) -> Dict:
        """Analyze a single problem with full chain of thought analysis."""
        print(f"\nAnalyzing problem {problem_data['id']}...")
        
        # Step 1: Get initial solution with full chain of thought
        initial_prompt = self.create_prompt(problem_data['problem'])
        full_response = self.llm.generate(initial_prompt, max_new_tokens=512, temperature=0.7)
        
        # Step 2: Extract chain of thought sentences
        cot_sentences = self.extractor.extract_sentences(full_response)
        
        print(f"  Extracted {len(cot_sentences)} chain of thought sentences")
        
        # Step 3: For each sentence, do resampling and forced answer
        sentence_analyses = []
        
        for i, sentence_idx in enumerate(range(len(cot_sentences))):
            print(f"  Processing sentence {sentence_idx + 1}/{len(cot_sentences)}...")
            
            # Get partial chain of thought up to this sentence
            partial_cot = cot_sentences[:sentence_idx + 1]
            
            # Resample until stable
            resample_responses, stabilized_answer = self.resample_until_stable(
                problem_data['problem'], 
                partial_cot,
                max_resamples=10
            )
            
            # Force final answer
            forced_answer = self.force_final_answer(
                problem_data['problem'],
                partial_cot
            )
            
            sentence_analyses.append({
                'sentence_index': sentence_idx,
                'sentence': cot_sentences[sentence_idx],
                'partial_cot': partial_cot,
                'resample_responses': resample_responses,
                'stabilized_answer': stabilized_answer,
                'forced_answer': forced_answer,
            })
        
        # Compile results
        result = {
            'problem_id': problem_data['id'],
            'problem': problem_data['problem'],
            'correct_answer': problem_data['correct_answer'],
            'full_solution': problem_data['solution'],
            'full_response': full_response,
            'chain_of_thought_sentences': cot_sentences,
            'sentence_analyses': sentence_analyses,
        }
        
        return result
    
    def run_analysis(self, num_problems: int = 20, output_file: str = "gsm8k_analysis_results.json"):
        """Run the full analysis pipeline."""
        print("=" * 60)
        print("GSM8K Chain of Thought Analysis")
        print("=" * 60)
        
        # Load problems
        problems = self.load_gsm8k(num_problems)
        
        # Analyze each problem
        for problem_data in problems:
            try:
                result = self.analyze_problem(problem_data)
                self.results.append(result)
                
                # Save intermediate results
                self.save_results(output_file)
                
            except Exception as e:
                print(f"Error analyzing problem {problem_data['id']}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # Final save
        self.save_results(output_file)
        print(f"\nAnalysis complete! Results saved to {output_file}")
        print(f"Analyzed {len(self.results)} problems")
    
    def save_results(self, output_file: str):
        """Save results to JSON file."""
        with open(output_file, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'num_problems': len(self.results),
                'results': self.results
            }, f, indent=2)


def main():
    """Main entry point."""
    analyzer = GSM8KAnalyzer()
    analyzer.run_analysis(num_problems=20)


if __name__ == "__main__":
    main()

