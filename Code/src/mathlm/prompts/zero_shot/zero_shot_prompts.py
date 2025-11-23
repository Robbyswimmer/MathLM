'''Zero-shot prompt templates aligned with our reward model.'''

ZERO_SHOT_PROMPT = '''
Solve the following math problem step-by-step.

Requirements:
1. Show your reasoning clearly by explaining each step.
2. Write lightweight Python code to compute the solution (no imports allowed).
3. Provide the final numerical answer at the end.

Format your response as follows:
<step-by-step reasoning - describe the problem and your approach>
<lightweight Python code - implement the solution>

```python
# Python code here to solve the problem
# Use basic arithmetic operations only
# Calculate and print the final answer
result = 3 + 5  # example calculation
print(result)
```
<final answer - the numerical result>

Problem: {problem}
'''

# For testing: a more explicit version emphasizing reward components
EXPLICIT_ZERO_SHOT_PROMPT = '''
Solve this math problem by following these steps:

1. Read and understand the problem
2. Explain your reasoning (describe the problem and how you plan to solve it)
3. Write Python code to calculate the answer (no imports, basic arithmetic only)
4. State the final numerical answer clearly

Your reasoning should explain the problem-solving approach before diving into calculations.

Python code should:
- Use only basic arithmetic operations (+, -, *, /, **)
- NOT include any imports or external libraries
- Calculate the final answer and print it

Format your response as follows:
<step-by-step reasoning - describe the problem and your approach>
<lightweight Python code - implement the solution>

```python
# Python code here to solve the problem
# Use basic arithmetic operations only
# Calculate and print the final answer
result = 3 + 5  # example calculation
print(result)
```

<final answer - the numerical result>

Now, solve the following problem: {problem}
'''

# Minimal version for comparison
MINIMAL_ZERO_SHOT_PROMPT = '''
Solve this math problem, show your reasoning, write Python code to calculate the answer (no imports), and state the final numerical result.

Problem: {problem}
'''

def get_zero_shot_prompt(template:str = 'default', problem:str = '') -> str:
    '''Returns the zero-shot prompt based on the specified template.'''
    if template == 'default':
        return ZERO_SHOT_PROMPT.format(problem=problem)
    elif template == 'explicit':
        return EXPLICIT_ZERO_SHOT_PROMPT.format(problem=problem)
    elif template == 'minimal':
        return MINIMAL_ZERO_SHOT_PROMPT.format(problem=problem)
    else:
        raise ValueError(f"Unknown template: {template}")