"""
Logic Questions Analysis with Qwen3
This script loads logic questions from logic_qs.json, runs Qwen3 on them,
extracts chain of thought, and performs logit analysis.
"""

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import re
from typing import List, Dict, Optional, Tuple
from collections import defaultdict
from datetime import datetime
import os


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
    
    def generate(self, prompt: str, max_new_tokens: int = 4096, temperature: float = 0.7) -> str:
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
        
        input_len = inputs['input_ids'].shape[1]
        response = self.tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
        
        # Clear GPU cache after generation
        del inputs, outputs
        if self.device == "cuda":
            torch.cuda.empty_cache()
        
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


class Qwen3LogitExtractor:
    """Wrapper for Qwen3 model with logit extraction capabilities."""
    
    def __init__(self, model_name: str = "Qwen/Qwen3-4B-Thinking-2507"):
        """Initialize the model and tokenizer."""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None,
        )
        
        if self.device == "cpu":
            self.model = self.model.to(self.device)
        
        # Find token IDs for "Yes" and "No"
        vocab = self.tokenizer.get_vocab()
        
        # Try to find Yes token
        yes_candidates = ["Yes", "yes", "YES", " Yes", " yes", "YES"]
        self.yes_token_id = None
        for candidate in yes_candidates:
            if candidate in vocab:
                self.yes_token_id = vocab[candidate]
                break
        
        if self.yes_token_id is None:
            encoded = self.tokenizer.encode("Yes", add_special_tokens=False)
            if len(encoded) > 0:
                self.yes_token_id = encoded[0]
        
        # Try to find No token
        no_candidates = ["No", "no", "NO", " No", " no", "NO"]
        self.no_token_id = None
        for candidate in no_candidates:
            if candidate in vocab:
                self.no_token_id = vocab[candidate]
                break
        
        if self.no_token_id is None:
            encoded = self.tokenizer.encode("No", add_special_tokens=False)
            if len(encoded) > 0:
                self.no_token_id = encoded[0]
    
    def get_logit_probabilities(self, prompt: str) -> Dict[str, float]:
        """
        Get logit probabilities for Yes vs No tokens.
        
        Returns:
            Dictionary with 'yes_logit', 'no_logit', 'yes_prob', 'no_prob'
        """
        # Tokenize the prompt
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_ids = inputs['input_ids']
        # Create attention mask (all ones for a single sequence)
        attention_mask = torch.ones_like(input_ids)
        
        # Forward pass to get logits (without generation)
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits  # Shape: [batch_size, seq_len, vocab_size]
        
        # Get logits for the last token position
        last_token_logits = logits[0, -1, :]  # Shape: [vocab_size]
        
        # Extract logits for Yes and No tokens (move to CPU immediately)
        yes_logit = last_token_logits[self.yes_token_id].item() if self.yes_token_id is not None else None
        no_logit = last_token_logits[self.no_token_id].item() if self.no_token_id is not None else None
        
        # Convert logits to probabilities using softmax (do on CPU to save GPU memory)
        last_token_logits_cpu = last_token_logits.cpu()
        del logits, outputs, last_token_logits  # Free GPU memory early
        
        if yes_logit is not None and no_logit is not None:
            # Softmax over just Yes and No tokens (on CPU)
            yes_no_logits = torch.tensor([no_logit, yes_logit])
            yes_no_probs = F.softmax(yes_no_logits, dim=0)
            no_prob = yes_no_probs[0].item()
            yes_prob = yes_no_probs[1].item()
        else:
            yes_prob = None
            no_prob = None
        
        # Also get the full vocabulary probability for Yes and No (for comparison) - on CPU
        full_probs = F.softmax(last_token_logits_cpu, dim=0)
        yes_prob_full = full_probs[self.yes_token_id].item() if self.yes_token_id is not None else None
        no_prob_full = full_probs[self.no_token_id].item() if self.no_token_id is not None else None
        
        # Clear GPU cache after logit extraction
        del inputs, input_ids, attention_mask, full_probs, last_token_logits_cpu
        if self.device == "cuda":
            torch.cuda.empty_cache()
        
        return {
            'yes_logit': yes_logit,
            'no_logit': no_logit,
            'yes_prob': yes_prob,  # Probability from Yes/No softmax
            'no_prob': no_prob,     # Probability from Yes/No softmax
            'yes_prob_full': yes_prob_full,  # Probability from full vocabulary softmax
            'no_prob_full': no_prob_full,    # Probability from full vocabulary softmax
        }


class LogicQuestionsAnalyzer:
    """Main analyzer for logic questions."""
    
    def __init__(self, model_name: str = "Qwen/Qwen3-4B-Thinking-2507"):
        self.llm = Qwen3Inference(model_name)
        self.extractor = ChainOfThoughtExtractor()
        self.logit_extractor = Qwen3LogitExtractor(model_name)
        self.results = []
    
    def load_logic_questions(self, json_file: str = "logic_qs.json") -> List[Dict]:
        """Load logic questions from JSON file."""
        print(f"Loading logic questions from {json_file}...")
        with open(json_file, 'r') as f:
            questions = json.load(f)
        print(f"Loaded {len(questions)} questions")
        return questions
    
    def create_prompt(self, question: str, partial_cot: Optional[List[str]] = None) -> str:
        """Create a prompt for the model."""
        base_prompt = f"{question}\n\nAnswer Yes or No:"
        
        if partial_cot:
            cot_text = " ".join(partial_cot)
            base_prompt = f"{question}\n\nReasoning: {cot_text}\n\nAnswer Yes or No:"
        
        return base_prompt
    
    def resample_logits(
        self, 
        question: str, 
        partial_cot: List[str], 
        num_resamples: int = 10
    ) -> List[Dict]:
        """
        Resample logits multiple times given partial chain of thought.
        Returns: list of logit dictionaries
        """
        prompt = self.create_prompt(question, partial_cot)
        # Add instruction to provide final answer
        prompt += "\n\nBased on the above reasoning, answer Yes or No:"
        
        logit_responses = []
        
        for i in range(num_resamples):
            logit_data = self.logit_extractor.get_logit_probabilities(prompt)
            logit_responses.append(logit_data)
            
            # Clear GPU cache after each resample
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        return logit_responses
    
    def get_logits_from_partial_cot(self, question: str, partial_cot: List[str]) -> Dict:
        """Force a final Yes/No answer from partial chain of thought, then extract logits."""
        prompt = self.create_prompt(question, partial_cot)
        # Add instruction to provide final answer
        prompt += "\n\nBased on the above reasoning, answer Yes or No:"
        
        logit_data = self.logit_extractor.get_logit_probabilities(prompt)
        return logit_data
    
    def analyze_question(self, question_data: Dict) -> Dict:
        """Analyze a single question with full chain of thought analysis."""
        print(f"\nAnalyzing question {question_data['id']}...")
        
        # Step 1: Get initial solution with full chain of thought
        initial_prompt = self.create_prompt(question_data['question'])
        full_response = self.llm.generate(initial_prompt, max_new_tokens=4096, temperature=0.7)
        
        # Step 2: Extract chain of thought sentences
        cot_sentences = self.extractor.extract_sentences(full_response)
        
        print(f"  Extracted {len(cot_sentences)} chain of thought sentences")
        
        # Step 3: For each sentence, force Yes/No and resample 10 times
        sentence_analyses = []
        
        for i, sentence_idx in enumerate(range(len(cot_sentences))):
            print(f"  Processing sentence {sentence_idx + 1}/{len(cot_sentences)}...")
            
            # Get partial chain of thought up to this sentence
            partial_cot = cot_sentences[:sentence_idx + 1]
            
            # Force Yes/No answer and extract logits (once)
            forced_logits = self.get_logits_from_partial_cot(
                question_data['question'],
                partial_cot
            )
            
            # Resample 10 times
            resampled_logits = self.resample_logits(
                question_data['question'], 
                partial_cot,
                num_resamples=10
            )
            
            # Clear GPU cache after processing each sentence
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            sentence_analyses.append({
                'sentence_index': sentence_idx,
                'sentence': cot_sentences[sentence_idx],
                'partial_cot': partial_cot,
                'forced_logits': forced_logits,
                'resampled_logits': resampled_logits,
            })
        
        # Determine predicted answer based on final forced logits
        if sentence_analyses:
            final_forced_logits = sentence_analyses[-1]['forced_logits']
            if final_forced_logits.get('yes_prob') is not None and final_forced_logits.get('no_prob') is not None:
                if final_forced_logits['yes_prob'] > final_forced_logits['no_prob']:
                    predicted_answer = "Yes"
                else:
                    predicted_answer = "No"
            else:
                predicted_answer = None
        else:
            predicted_answer = None
        
        # Check if prediction matches correct answer
        is_correct = (predicted_answer == question_data['answer']) if predicted_answer else None
        
        result = {
            'question_id': question_data['id'],
            'question': question_data['question'],
            'correct_answer': question_data['answer'],
            'predicted_answer': predicted_answer,
            'is_correct': is_correct,
            'full_response': full_response,
            'chain_of_thought_sentences': cot_sentences,
            'sentence_analyses': sentence_analyses,
        }
        
        final_prob = sentence_analyses[-1]['forced_logits'].get('yes_prob', 0) if sentence_analyses else 0
        print(f"  Correct: {question_data['answer']}, Predicted: {predicted_answer}, "
              f"Yes prob: {final_prob:.4f}")
        
        return result
    
    def run_analysis(self, json_file: str = "logic_qs.json", output_file: str = "logic_qs_analysis_results.json"):
        """Run the full analysis pipeline."""
        print("=" * 60)
        print("Logic Questions Chain of Thought Analysis")
        print("=" * 60)
        
        # Load questions
        questions = self.load_logic_questions(json_file)
        
        # Analyze first 30 questions
        for question_data in questions[:30]:
            try:
                result = self.analyze_question(question_data)
                self.results.append(result)
                
                # Save intermediate results
                self.save_results(output_file)
                
            except Exception as e:
                print(f"Error analyzing question {question_data['id']}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # Final save
        self.save_results(output_file)
        
        # Print summary
        correct_count = sum(1 for r in self.results if r.get('is_correct') == True)
        total_count = len(self.results)
        accuracy = (correct_count / total_count * 100) if total_count > 0 else 0
        
        print(f"\nAnalysis complete! Results saved to {output_file}")
        print(f"Analyzed {total_count} questions")
        print(f"Accuracy: {correct_count}/{total_count} ({accuracy:.2f}%)")
    
    def save_results(self, output_file: str):
        """Save results to JSON file."""
        with open(output_file, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'num_questions': len(self.results),
                'results': self.results
            }, f, indent=2)


def main():
    """Main entry point."""
    analyzer = LogicQuestionsAnalyzer()
    analyzer.run_analysis()


if __name__ == "__main__":
    main()
