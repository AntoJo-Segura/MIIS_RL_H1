
from numpy import argmax
from functools import reduce
from problem import *
from  iterative_policy_evaluation import *

def value_iteration(problem_input: problem):

    S = problem_input.state_space
    A = problem_input.actions
    r = problem_input.reward
    p = problem_input.transition_p
    gamma = problem_input.gamma
    V = [1]*len(S)#iterative_policy_evaluation(problem_input)
    threshold = 1e-5

    def sum_s_r(action, state):
        expected_reward = 0
        for i,sp in enumerate(S):  
            expected_reward += p(sp, action, state)*(r(s,action)+ gamma*V[i])
        return expected_reward

    increment = threshold
    loop = 0
    while increment >= 0:
        loop += 1
        for i,s in enumerate(S):
            v = V[i]
            V[i] = max(list(map(
                lambda a: sum_s_r(a,s), A
            )))
            test = abs(v - V[i])
            increment = max(increment, abs(v - V[i]))

    pi = list(map(lambda s:
            argmax(list(map(
                lambda a: sum_s_r(a,s),A
            ))), S
    ))
    return pi



print("start")

print(value_iteration(lazy_problem))
