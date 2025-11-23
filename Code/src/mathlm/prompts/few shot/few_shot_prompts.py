'''Few shot prompt templates aligned with our reward model.'''

ONE_SHOT_PROMPT = '''Solve the following math problem step-by-step. Show your reasoning, write lightweight Python code to compute the solution (no imports allowed), and provide the final numerical answer at the end.

Requirements:
1. Show your reasoning clearly by explaining each step.
2. Write lightweigth Python code to compute the solution (no imports allowed).
3. Provide the final numerical answer at the end.

Format your response as follows:
<step-by-step reasoning - describe the problem and your approach>
<lightweight Python code - implement the solution>

```python
# Python code here to solve the problem
# Use basic arithmetic operations only
# Calculate and print the final answer
result = <calculation>
print(result)
```
<final answer - the numerical result>

Example:

Problem: Jane has 3 apples and buys 5 more. How many apples does she have?

Solution: Jane starts with 3 apples and buys 5 more apples. To find the total number of apples, I need to add these two quantities together.

```python
# calculate the total number of apples
initial_apples = 3
bought_apples = 5
total_apples = initial_apples + bought_apples
print(total_apples)
``` 

The final answer is: 8

Now, solve the following problem:
Problem: {problem}'''

TWO_SHOT_PROMPT = '''Solve the following math problem step by step. Show your reasoning, write Python code to compute the solution (no imports allowed), and state the final numerical answer.

Requirements:
1. Show your reasoning clearly by explaining each step.
2. Write lightweigth Python code to compute the solution (no imports allowed).
3. Provide the final numerical answer at the end.

Format your response as follows:
<step-by-step reasoning - describe the problem and your approach>
<lightweight Python code - implement the solution>

```python
# Python code here to solve the problem
# Use basic arithmetic operations only
# Calculate and print the final answer
result = <calculation>
print(result)
```
<final answer - the numerical result>

Example 1:

Problem: Jane has 3 apples and buys 5 more. How many apples does she have?

Solution: Jane starts with 3 apples and buys 5 more apples. To find the total number of apples, I need to add these two quantities together.

```python
# Calculate total apples
initial_apples = 3
bought_apples = 5
total_apples = initial_apples + bought_apples
print(total_apples)
```

The final answer is: 8

Example 2:

Problem: If a rectangle has sides 4 and 6, what is its area?

Solution: The area of a rectangle is calculated by multiplying its length by its width. In this case, one side is 4 units and the other side is 6 units.

```python
# calculate rectangle area = length * width
length = 4
width = 6
area = length * width
print(area)
```

The final answer is: 24

Now solve this problem:

Problem: {problem}'''

THREE_SHOT_PROMPT = '''Solve the following math problem step by step. Show your reasoning, write Python code to compute the solution (no imports allowed), and state the final numerical answer.

Requirements:
1. Show your reasoning clearly by explaining each step.
2. Write lightweigth Python code to compute the solution (no imports allowed).
3. Provide the final numerical answer at the end.

Format your response as follows:
<step-by-step reasoning - describe the problem and your approach>
<lightweight Python code - implement the solution>

```python
# Python code here to solve the problem
# Use basic arithmetic operations only
# Calculate and print the final answer
result = <calculation>
print(result)
```
<final answer - the numerical result>

Example 1:

Problem: Jane has 3 apples and buys 5 more. How many apples does she have?

Solution: Jane starts with 3 apples and buys 5 more apples. To find the total number of apples, I need to add these two quantities together.

```python
# Calculate total apples
initial_apples = 3
bought_apples = 5
total_apples = initial_apples + bought_apples
print(total_apples)
```

The final answer is: 8

Example 2:

Problem: If a rectangle has sides 4 and 6, what is its area?

Solution: The area of a rectangle is calculated by multiplying its length by its width. In this case, one side is 4 units and the other side is 6 units.

```python
# calculate rectangle area = length * width
length = 4
width = 6
area = length * width
print(area)
```

The final answer is: 24

Example 3:

Problem: A shop sells pencils at $2 each. How much for 7 pencils?

Solution: Each pencil costs $2 and we need to buy 7 pencils. To find the total cost, I multiply the price per pencil by the number of pencils.

```python
# calculate total cost
price_per_pencil = 2
number_of_pencils = 7
total_cost = price_per_pencil * number_of_pencils
print(total_cost)
```

The final answer is: 14

Now solve this problem:

Problem: {problem}'''

# More concise few-shot prompts for comparison
CONCISE_ONE_SHOT_PROMPT = '''Solve this math problem with reasoning, Python code (no imports), and a final answer.

Example:
Q: Jane has 3 apples and buys 5 more. How many apples does she have?
A: Jane starts with 3 apples and gets 5 more, so I need to add them.
```python
total = 3 + 5
print(total)
```
Answer: 8

Q: {problem}
A:'''

CONCISE_TWO_SHOT_PROMPT = '''Solve this math problem with reasoning, Python code (no imports), and a final answer.

Example 1:
Q: Jane has 3 apples and buys 5 more. How many apples does she have?
A: Jane starts with 3 apples and gets 5 more, so I need to add them.
```python
total = 3 + 5
print(total)
```
Answer: 8

Example 2:
Q: If a rectangle has sides 4 and 6, what is its area?
A: Area of a rectangle is length times width.
```python
area = 4 * 6
print(area)
```
Answer: 24

Q: {problem}
A:'''

CONCISE_THREE_SHOT_PROMPT = '''Solve this math problem with reasoning, Python code (no imports), and a final answer.

Example 1:
Q: Jane has 3 apples and buys 5 more. How many apples does she have?
A: Jane starts with 3 apples and gets 5 more, so I need to add them.
```python
total = 3 + 5
print(total)
```
Answer: 8

Example 2:
Q: If a rectangle has sides 4 and 6, what is its area?
A: Area of a rectangle is length times width.
```python
area = 4 * 6
print(area)
```
Answer: 24

Example 3:
Q: A shop sells pencils at $2 each. How much for 7 pencils?
A: Each pencil costs $2, so for 7 pencils I multiply the price by quantity.
```python
cost = 2 * 7
print(cost)
```
Answer: 14

Q: {problem}
A:'''

def get_few_shot_prompt(shot_count:int = 3, type:str = 'detailed', problem:str = '') -> str:
    '''Returns the few-shot prompt based on the specified shot count and type.'''
    if shot_count < 1 or shot_count > 3:
        raise ValueError("shot_count must be between 1 and 3.")
    
    # Map shot counts to corresponding prompts
    prompts = {
        1: ONE_SHOT_PROMPT if type == 'detailed' else CONCISE_ONE_SHOT_PROMPT,
        2: TWO_SHOT_PROMPT if type == 'detailed' else CONCISE_TWO_SHOT_PROMPT,
        3: THREE_SHOT_PROMPT if type == 'detailed' else CONCISE_THREE_SHOT_PROMPT,
    }

    return prompts[shot_count].format(problem=problem)
