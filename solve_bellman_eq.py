from problem import *
from iterative_policy_evaluation import *
import numpy as np

def solve_bellman_eq(matrix_problem):
    I = np.identity(len(matrix_problem.state_space))
    gamma = matrix_problem.gamma
    P = matrix_problem.transition_m
    r = matrix_problem.reward_space
    #cast to np array to allow element, matrix multiplication
    probability_matrix = np.linalg.inv(I - gamma * np.array(P) ) 

    return  np.matmul(probability_matrix, r)



def lazy_policy_f(state):
    return "low"

def lazy_p_p(action, state):
    if action == "low":
        return 1 
    else :
        return 0

lazy_policy = policy(lazy_policy_f,lazy_p_p)
mpro = matrix_problem(lazy_policy)

V_lazy = solve_bellman_eq(mpro)
print(V_lazy)
print("--------")

def aggre_policy_f( state):
    if state < 50:
        return "low"
    else:
        return "high"

def aggre_policy_p( action, state):
    if state < 50 and action == "low" or state >= 50 and action == "high" :
        return 1
    else:
        return 0


aggre_policy = policy(aggre_policy_f, aggre_policy_p)
aggre_problem = matrix_problem(aggre_policy)

V_agg = solve_bellman_eq(aggre_problem)
print(V_agg)


print("----")
print(V_lazy - V_agg) 


