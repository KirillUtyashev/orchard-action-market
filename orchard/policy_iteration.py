from main_pi import policy_iteration

"""
Call the policy iteration function (from main_pi.py).
Commences training.
"""

approach = "value"
name = "D-2_5_test"


policy_iteration(approach, name, 2, 5)


