# POLICY EVALUATION

## AIM:
To develop a Python program to evaluate the given policy.

## PROBLEM STATEMENT:
The bandit slippery walk problem is a reinforcement learning problem in which an agent must learn to navigate a 7-state environment in order to reach a goal state. The environment is slippery, so the agent has a chance of moving in the opposite direction of the action it takes.

### States
The environment has 7 states:

* Two Terminal States: G: The goal state & H: A hole state.
* Five Transition states / Non-terminal States including S: The starting state.

### Actions
The agent can take two actions:

* R: Move right.
* L: Move left.

### Transition Probabilities
The transition probabilities for each action are as follows:

* 50% chance that the agent moves in the intended direction.
* 33.33% chance that the agent stays in its current state.
* 16.66% chance that the agent moves in the opposite direction.

For example, if the agent is in state S and takes the "R" action, then there is a 50% chance that it will move to state 4, a 33.33% chance that it will stay in state S, and a 16.66% chance that it will move to state 2.

### Rewards
The agent receives a reward of +1 for reaching the goal state (G). The agent receives a reward of 0 for all other states.

### Graphical Representation
![311135365-37d36c42-be52-455a-bcb3-155566bd7638](https://github.com/sivabalan28/rl-policy-evaluation/assets/113497347/986a9d25-0be0-435a-8c22-613347033a41)

## POLICY EVALUATION FUNCTION

### Formula
![311135399-e79e44f8-4ccd-4205-bf44-e2b10fd48a73](https://github.com/sivabalan28/rl-policy-evaluation/assets/113497347/d82cd793-4835-4e36-bf01-395214bc2b46)


## Program:
```
DEVELOPED BY:SIVABALAN S
REGISTER NO.:212222240100
```
## Policy Evaluation:
```python
def policy_evaluation(pi, P, gamma=1.0, theta=1e-10):
    prev_V = np.zeros(len(P), dtype=np.float64)
    while True:
        V = np.zeros(len(P), dtype=np.float64)
        for s in range(len(P)):
            for prob, next_state, reward, done in P[s][pi(s)]:
                V[s] += prob * (reward + gamma * prev_V[next_state] * (not done))
        if np.max(np.abs(prev_V - V)) < theta:
            break
        prev_V = V.copy()
    return V
```

## Code to evaluate the first policy:
```python
pi_1 = lambda s: {
    0:LEFT, 1:LEFT, 2:LEFT, 3:LEFT, 4:LEFT, 5:LEFT, 6:LEFT
}[s]
print_policy(pi_1, P, action_symbols=('<', '>'), n_cols=7)
print('Reaches goal {:.2f}%. Obtains an average undiscounted return of {:.4f}.'.format(
    probability_success(env, pi_1, goal_state=goal_state)*100,
    mean_return(env, pi_1)))
V1 = policy_evaluation(pi_1, P)
print_state_value_function(V1, P, n_cols=7, prec=5)
V1
```

## Code to evaluate the second policy:
```python
pi_2 = lambda s: {
    0:RIGHT, 1:RIGHT, 2:LEFT, 3:RIGHT, 4:RIGHT, 5:LEFT, 6:RIGHT
}[s]
print_policy(pi_2, P, action_symbols=('<', '>'), n_cols=7)
     

print('Reaches goal {:.2f}%. Obtains an average undiscounted return of {:.4f}.'.format(
    probability_success(env, pi_2, goal_state=goal_state)*100,
    mean_return(env, pi_2)))

V2 = policy_evaluation(pi_2, P)
print_state_value_function(V2, P, n_cols=7, prec=5)
V2
```
## COMPARISON:

```python
V1>=V2

print('SIVABALAN S')
print('212222240100')
if(np.sum(V1>=V2)==7):
  print("The first policy is the better policy")
elif(np.sum(V2>=V1)==7):
  print("The second policy is the better policy")
else:
  print("Both policies have their merits.")
```

## OUTPUT:
### Policy 1:
![Screenshot 2024-03-12 124807](https://github.com/sivabalan28/rl-policy-evaluation/assets/113497347/73fd0730-6c2d-4a3d-be0a-9471646324be)

### Policy 2:
![Screenshot 2024-03-12 124937](https://github.com/sivabalan28/rl-policy-evaluation/assets/113497347/4b4d4dc2-1951-4227-94b7-6afa2a93d397)

### Conclusion:
![Screenshot 2024-03-12 125031](https://github.com/sivabalan28/rl-policy-evaluation/assets/113497347/b2777777-22b8-43d0-9c02-65c1c6b00de8)\

## RESULT:
Thus, a Python program is developed to evaluate the given policy.
