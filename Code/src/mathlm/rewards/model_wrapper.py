import torch
import torch.nn as nn
from typing import List, Optional, Union
from mathlm.rewards import RewardCalculator

class MathRewardModel(nn.Module):
    """
    Wraps RewardCalculator to be compatible with TRL's PPOTrainer.
    Expects input_ids to contain [prompt + response].
    """
    def __init__(self, reward_calc: RewardCalculator, tokenizer):
        super().__init__()
        self.reward_calc = reward_calc
        self.tokenizer = tokenizer
        # TRL v0.25.1 compatibility
        self.base_model_prefix = "reward_calc"
        self.config = nn.Module() # Dummy config
        self.config.is_encoder_decoder = False
    
    def forward(
        self, 
        input_ids: torch.LongTensor, 
        attention_mask: Optional[torch.LongTensor] = None, 
        **kwargs
    ):
        # Decode full sequences
        texts = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)
        
        rewards = []
        for text in texts:
            # We need to extract the question and response.
            # Assuming the prompt ends with "Answer:" or similar, and the rest is the response.
            # However, RewardCalculator.evaluate(question, response_text) needs the original question.
            # Parsing the question back from the prompt is brittle.
            
            # BETTER APPROACH:
            # Since we are using this in a specific training loop, maybe we can assume
            # the dataset provided the prompt?
            # But PPOTrainer calls this with generated text.
            
            # For now, let's try to split by a known separator if possible.
            # If we look at `mathlm.prompts`, the default template usually ends with "Answer:".
            
            parts = text.split("Answer:")
            if len(parts) >= 2:
                # Everything before the last "Answer:" is the prompt (roughly)
                # Everything after is the response.
                # This is a heuristic.
                question_part = "Answer:".join(parts[:-1]).strip()
                response_part = parts[-1].strip()
            else:
                # Fallback
                question_part = text
                response_part = ""
                
            # RewardCalculator expects the GSM8KExample object or just the question text?
            # Let's check RewardCalculator.evaluate signature.
            # It takes (question: GSM8KExample | str, response_text: str)
            
            breakdown = self.reward_calc.evaluate(question_part, response_part)
            rewards.append(breakdown.total)
            
        return torch.tensor(rewards, dtype=torch.float32, device=input_ids.device)
