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
        # We use a property to return 'self' to avoid RecursionError in .to()
        # (properties are not registered as child modules)
        self.base_model_prefix = "model"
        
        self.config = nn.Module() # Dummy config
        self.config.is_encoder_decoder = False

    @property
    def model(self):
        return self
    
    def forward(
        self, 
        input_ids: torch.LongTensor, 
        attention_mask: Optional[torch.LongTensor] = None, 
        **kwargs
    ):
        # Decode full sequences with special tokens to find the separator
        texts = self.tokenizer.batch_decode(input_ids, skip_special_tokens=False)
        
        rewards = []
        for i, text in enumerate(texts):
            # Try multiple potential separators for Gemma and other models
            separators = [
                "<start_of_turn>model",
                "<start_of_turn> model",  # sometimes tokenizer adds a space
                "<|assistant|>",          # other models
                "Answer:",                # fallback
            ]
            
            question_part = text
            response_part = ""
            
            for sep in separators:
                if sep in text:
                    parts = text.split(sep)
                    # Take the last part as response (in case of multiple turns, though we usually have 1)
                    # Actually, for PPO, we usually have [Prompt] [Response]. 
                    # The prompt ends with the separator.
                    question_part = sep.join(parts[:-1]).strip()
                    response_part = parts[-1].strip()
                    break
            
            # Fallback: if no separator found, check for "Problem:" structure
            if response_part == "" and "Problem:" in text:
                 parts = text.rsplit("Problem:", 1)
                 if len(parts) > 1:
                     question_part = parts[0] + "Problem:" + parts[1].split("\n", 1)[0]
                     response_part = parts[1].split("\n", 1)[-1].strip()
            
            # Clean up special tokens
            clean_q = question_part.replace("<bos>", "").replace("<eos>", "").replace("<pad>", "").replace("<start_of_turn>", "").replace("<end_of_turn>", "").strip()
            clean_r = response_part.replace("<eos>", "").replace("<pad>", "").replace("<end_of_turn>", "").strip()

            # DEBUG: Print full text for the first example of the batch to diagnose splitting
            if i == 0 and len(rewards) == 0: # Print only once per batch
                import sys
                print(f"\n[REWARD DEBUG] Full text (repr): {repr(text)}", file=sys.stderr, flush=True)
                print(f"[REWARD DEBUG] Separator used: {sep if response_part else 'NONE'}", file=sys.stderr, flush=True)
                print(f"[REWARD DEBUG] Question (clean): {clean_q[:50]}...", file=sys.stderr, flush=True)
                print(f"[REWARD DEBUG] Response (clean): {clean_r[:100]}...", file=sys.stderr, flush=True)

            breakdown = self.reward_calc.evaluate(clean_q, clean_r)
            rewards.append(breakdown.total)
            
        # TRL expects a tensor of shape [batch_size, seq_len] (or [batch_size, seq_len, 1])
        # We place the reward at the last token position.
        batch_size = input_ids.shape[0]
        seq_len = input_ids.shape[1]
        rewards_tensor = torch.zeros(batch_size, seq_len, dtype=torch.float32, device=input_ids.device)
        
        # Assign rewards to the last token of each sequence
        for i, r in enumerate(rewards):
            rewards_tensor[i, -1] = r
        
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
