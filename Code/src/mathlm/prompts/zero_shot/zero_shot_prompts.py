'''Zero-shot prompt templates aligned with our reward model.'''

ZERO_SHOT_PROMPT = '''
Solve the following math problem step-by-step. Show your reasoning clearly and provide the final numerical answer.

Problem: {problem}

Solution:
'''

FEW_SHOT_EXAMPLES = """
Problem: A store sells 5 apples for $1 each and 3 oranges for $2 each. How much does it cost to buy all of them?

<step-by-step reasoning - describe the problem and your approach>
We need to calculate the total cost.
1. Cost of apples: 5 apples * $1/apple = $5
2. Cost of oranges: 3 oranges * $2/orange = $6
3. Total cost: $5 + $6 = $11

<lightweight Python code - implement the solution>
```python
# Calculate cost of apples
apples = 5 * 1
# Calculate cost of oranges
oranges = 3 * 2
# Total cost
total = apples + oranges
print(total)
```

<final answer - the numerical result>
11

Problem: If a train travels at 60 mph for 2 hours, how far does it go?

<step-by-step reasoning - describe the problem and your approach>
Distance is speed multiplied by time.
Speed = 60 mph
Time = 2 hours
Distance = 60 * 2 = 120 miles.

<lightweight Python code - implement the solution>
```python
speed = 60
time = 2
distance = speed * time
print(distance)
```

<final answer - the numerical result>
120
"""

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