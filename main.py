
from problem import *
from iterative_policy_evaluation  import *
# from solve_bellman_eq import *
from policy_iteration import *
from value_iteration import *

from matplotlib.pyplot import *
from numpy import array

######  P1
V_lazy  =iterative_policy_evaluation(lazy_problem)
V_agg  =iterative_policy_evaluation(agg_problem)

dV = array(V_lazy) - array(V_agg)
print(dV)

plot([0]*len(lazy_problem.state_space))
plot(dV,'go-')
savefig('p1.png')
show()

######  P2
print("start")
pi_star = policy_iteration(lazy_problem)#any problem would work

policy_star = lazy_policy
policy_star.vec_to_f(pi_star)
policy_star.generate_p_f()
problem_star = problem(policy_star)
V_star = iterative_policy_evaluation(problem_star)

dV_lazy = array(V_star) - array(V_lazy)
print(dV_lazy)
plot([0]*len(lazy_problem.state_space))
plot(dV_lazy,'go-')
savefig('p2_lazy.png')
show()

dV_agg = array(V_star) - array(V_agg)
print(dV_agg)
plot([0]*len(lazy_problem.state_space))
plot(dV_agg,'go-')
savefig('p2_agg.png')
show()


plot(pi_star,'go-')
savefig('p2_star.png')
show()