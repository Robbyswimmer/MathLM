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
        # TRL expects base_model_prefix to point to a callable "backbone" module.
        # We point it to 'self' so that self.forward() is called.
        self.model = self 
        self.base_model_prefix = "model"
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
            parts = text.split("Answer:")
            if len(parts) >= 2:
                question_part = "Answer:".join(parts[:-1]).strip()
                response_part = parts[-1].strip()
            else:
                question_part = text
                response_part = ""
            
            breakdown = self.reward_calc.evaluate(question_part, response_part)
            rewards.append(breakdown.total)
            
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=input_ids.device)
        
        # TRL expects output.hidden_states[-1] to be passed to score()
        # We wrap our rewards in a dummy object.
        # We return the rewards tensor as the "hidden state".
        class DummyOutput:
            def __init__(self, hidden_states):
                self.hidden_states = hidden_states
                
        return DummyOutput(hidden_states=[rewards_tensor])

    def score(self, hidden_states):
        # TRL calls this with output.hidden_states[-1]
        # In our case, that is the rewards_tensor we returned in forward()
        return hidden_states
