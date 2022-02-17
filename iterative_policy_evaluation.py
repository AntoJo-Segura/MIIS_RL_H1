def iterative_policy_evaluation(problem, max_iter = 100):
    increment = 0
    threshold = 1e-3
    gamma = problem.gamma
    reward = problem.reward
    S = problem.state_space
    transition_p = problem.transition_p
    actions = problem.actions
    policy = problem.policy_p
    V = [threshold]*len(S)
    v = 0
    increment = threshold +1 #just enter into loop
    loop = 0 
    while increment >= threshold:
        # print("start loop"+ str(loop))
        increment = 0
        for i,s in enumerate(S):#For each s in S
            v = V[i]

            sum_value = 0.0 #sum loop
            for a in actions: 
                sum_reward = 0.0
                for j,sp in enumerate(S):
                    sum_reward += transition_p(sp,a, s)*(reward(sp,a) +  gamma * V[j])
                sum_value += policy(a,s) * sum_reward

            V[i] = sum_value
            increment = max(increment, abs(v - V[i]))
            # print('    diff '+ str(abs(v - V[i])))
            
        # print('-----')
        # print('increment '+ str(increment))
        # print('value '+ str(v))
        # print('check condition: ' +str(increment)+ '>= '+ str(threshold))

        loop += 1
        if loop >= max_iter:
            print('loop '+ str(loop))
            break

    # print("end")
    return V


# from problem import *




# lazy_problem = problem(lazy_policy)



# V_lazy = iterative_policy_evaluation(lazy_problem)



# print(V_lazy[49])
# print(V_lazy[79])

# print("----")




# aggre_problem = problem(aggre_policy)

# V_agg = iterative_policy_evaluation(aggre_problem)

# print(V_agg[49])
# print(V_agg[79])

# print("----")
# import numpy as np
# print(np.array(V_lazy) - np.array(V_agg) )