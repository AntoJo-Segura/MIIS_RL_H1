from numpy import argmax
from problem import *
from  iterative_policy_evaluation import *

def policy_iteration(problem_input: problem):
    S = problem_input.state_space
    vec_policy = list(map(lambda x:
        problem_input.policy(x), 
        S
    ))
    A = problem_input.actions
    r = problem_input.reward
    p = problem_input.transition_p
    gamma = problem_input.gamma

    def initialization():
        return ["high"]*len(S) # ["high"]*(len(S)-50) +["low"]*50


    def policy_improvement(v_policy, V):

        def sum_s_r(action, state):
            expected_reward = 0
            for i,sp in enumerate(S):  
                expected_reward += p(sp, action, state)*(r(s,action)+ gamma*V[i])
            return expected_reward

        policy_stable = True
        for i,s in enumerate(S):
            old_action = v_policy[i]
            v_policy[i] = A[
                argmax(
                    list(map(
                        lambda x: sum_s_r(x ,s ), A
                    ))
                )
            ]
            if old_action != v_policy[i]:
                policy_stable = False
        
        if policy_stable :
            return v_policy, False
        else:
            new_policy = policy(problem_input.policy,problem_input.policy_p)
            new_policy.vec_to_f(v_policy)
            new_problem = problem(new_policy)
            V = iterative_policy_evaluation(new_problem)
            return v_policy, True

    init_v_pol = initialization()
    init_policy = policy(problem_input.policy,problem_input.policy_p)
    init_policy.vec_to_f(init_v_pol)
    new_problem = problem(init_policy)
    V = iterative_policy_evaluation(new_problem)
    pi,not_ended = policy_improvement(init_v_pol, V)

    loop = 0
    while not_ended:
        init_policy = policy(problem_input.policy,problem_input.policy_p)
        init_policy.vec_to_f(pi)
        new_problem = problem(init_policy)
        V = iterative_policy_evaluation(new_problem)
        pi, not_ended = policy_improvement(pi, V)
        loop += 1
    print(loop)
    return pi

# print("start")

# print(policy_iteration(lazy_problem))
