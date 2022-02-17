


from iterative_policy_evaluation  import *
from problem import *
from solve_bellman_eq import *


def lazy_policy(state):
    return "low"
lazy_problem = problem(lazy_policy)

def aggre_policy( state):
    if state < 50:
        return "low"
    else:
        return "high"

aggre_problem = problem(aggre_policy)



# iterative_policy_evaluation(lazy_problem)
# iterative_policy_evaluation(aggre_problem)

# print(
#     iterative_policy_evaluation(lazy_problem)
# )

# print(
#     iterative_policy_evaluation(aggre_problem)
# )



lazy_problem_m = matrix_problem(lazy_policy)
aggre_problem_m = matrix_problem(aggre_policy)
solve_bellman_eq(lazy_problem_m) 
v_star1 = solve_bellman_eq(aggre_problem_m)
v_star2 = solve_bellman_eq(lazy_problem_m) 
print("----")
print(v_star1 - v_star2) # V* diff is not 0



with open('results.txt', 'w') as file:
    file.write('whatever')
