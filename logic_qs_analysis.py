"""
Logic Questions Analysis with Qwen3
This script loads logic questions from logic_qs.json, runs Qwen3 on them,
and extracts logit probabilities for "Yes" vs "No" from model internals.
"""

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
from typing import List, Dict, Optional
from datetime import datetime
import os


class Qwen3LogitExtractor:
    """Wrapper for Qwen3 model with logit extraction capabilities."""
    
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
        
        # Find token IDs for "Yes" and "No"
        # Try different variations to find the correct token IDs
        vocab = self.tokenizer.get_vocab()
        
        # Try to find Yes token
        yes_candidates = ["Yes", "yes", "YES", " Yes", " yes", "YES"]
        self.yes_token_id = None
        for candidate in yes_candidates:
            if candidate in vocab:
                self.yes_token_id = vocab[candidate]
                break
        
        # If not found, try encoding
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
        
        # If not found, try encoding
        if self.no_token_id is None:
            encoded = self.tokenizer.encode("No", add_special_tokens=False)
            if len(encoded) > 0:
                self.no_token_id = encoded[0]
        
        print(f"Token IDs - Yes: {self.yes_token_id}, No: {self.no_token_id}")
        
        # Verify tokens can be decoded back
        if self.yes_token_id is not None:
            decoded_yes = self.tokenizer.decode([self.yes_token_id])
            print(f"  Yes token decodes to: '{decoded_yes}'")
        if self.no_token_id is not None:
            decoded_no = self.tokenizer.decode([self.no_token_id])
            print(f"  No token decodes to: '{decoded_no}'")
        print("Model loaded successfully!")
    
    def get_logit_probabilities(self, prompt: str) -> Dict[str, float]:
        """
        Get logit probabilities for Yes vs No tokens.
        
        Returns:
            Dictionary with 'yes_logit', 'no_logit', 'yes_prob', 'no_prob', and 'generated_text'
        """
        # Tokenize the prompt
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_ids = inputs['input_ids']
        
        # Forward pass to get logits (without generation)
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids)
            logits = outputs.logits  # Shape: [batch_size, seq_len, vocab_size]
        
        # Get logits for the last token position
        last_token_logits = logits[0, -1, :]  # Shape: [vocab_size]
        
        # Extract logits for Yes and No tokens
        yes_logit = last_token_logits[self.yes_token_id].item() if self.yes_token_id is not None else None
        no_logit = last_token_logits[self.no_token_id].item() if self.no_token_id is not None else None
        
        # Convert logits to probabilities using softmax
        # We'll compute probabilities for just Yes/No tokens, and also full vocabulary
        if yes_logit is not None and no_logit is not None:
            # Softmax over just Yes and No tokens
            yes_no_logits = torch.tensor([no_logit, yes_logit])
            yes_no_probs = F.softmax(yes_no_logits, dim=0)
            no_prob = yes_no_probs[0].item()
            yes_prob = yes_no_probs[1].item()
        else:
            yes_prob = None
            no_prob = None
        
        # Also get the full vocabulary probability for Yes and No (for comparison)
        full_probs = F.softmax(last_token_logits, dim=0)
        yes_prob_full = full_probs[self.yes_token_id].item() if self.yes_token_id is not None else None
        no_prob_full = full_probs[self.no_token_id].item() if self.no_token_id is not None else None
        
        # Generate the actual response for reference
        with torch.no_grad():
            generated = self.model.generate(
                input_ids=input_ids,
                max_new_tokens=50,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        
        generated_text = self.tokenizer.decode(
            generated[0][input_ids.shape[1]:], 
            skip_special_tokens=True
        ).strip()
        
        return {
            'yes_logit': yes_logit,
            'no_logit': no_logit,
            'yes_prob': yes_prob,  # Probability from Yes/No softmax
            'no_prob': no_prob,     # Probability from Yes/No softmax
            'yes_prob_full': yes_prob_full,  # Probability from full vocabulary softmax
            'no_prob_full': no_prob_full,    # Probability from full vocabulary softmax
            'generated_text': generated_text,
        }


class LogicQuestionsAnalyzer:
    """Main analyzer for logic questions."""
    
    def __init__(self, model_name: str = "Qwen/Qwen3-4B-Thinking-2507"):
        self.extractor = Qwen3LogitExtractor(model_name)
        self.results = []
    
    def load_logic_questions(self, json_file: str = "logic_qs.json") -> List[Dict]:
        """Load logic questions from JSON file."""
        print(f"Loading logic questions from {json_file}...")
        with open(json_file, 'r') as f:
            questions = json.load(f)
        print(f"Loaded {len(questions)} questions")
        return questions
    
    def create_prompt(self, question: str) -> str:
        """Create a prompt for the model."""
        # Format the question with a clear instruction to answer Yes or No
        prompt = f"{question}\n\nAnswer Yes or No:"
        return prompt
    
    def analyze_question(self, question_data: Dict) -> Dict:
        """Analyze a single question and extract logit probabilities."""
        print(f"\nAnalyzing question {question_data['id']}...")
        
        # Create prompt
        prompt = self.create_prompt(question_data['question'])
        
        # Get logit probabilities
        logit_data = self.extractor.get_logit_probabilities(prompt)
        
        # Determine predicted answer based on probabilities
        if logit_data['yes_prob'] is not None and logit_data['no_prob'] is not None:
            if logit_data['yes_prob'] > logit_data['no_prob']:
                predicted_answer = "Yes"
            else:
                predicted_answer = "No"
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
            'yes_logit': logit_data['yes_logit'],
            'no_logit': logit_data['no_logit'],
            'yes_prob': logit_data['yes_prob'],
            'no_prob': logit_data['no_prob'],
            'yes_prob_full': logit_data['yes_prob_full'],
            'no_prob_full': logit_data['no_prob_full'],
            'generated_text': logit_data['generated_text'],
        }
        
        print(f"  Correct: {question_data['answer']}, Predicted: {predicted_answer}, "
              f"Yes prob: {logit_data['yes_prob']:.4f}, No prob: {logit_data['no_prob']:.4f}")
        
        return result
    
    def run_analysis(self, json_file: str = "logic_qs.json", output_file: str = "logic_qs_analysis_results.json"):
        """Run the full analysis pipeline."""
        print("=" * 60)
        print("Logic Questions Logit Analysis")
        print("=" * 60)
        
        # Load questions
        questions = self.load_logic_questions(json_file)
        
        # Analyze each question
        for question_data in questions:
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

