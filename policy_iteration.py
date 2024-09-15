from main_pi import policy_iteration

"""
Call the policy iteration function (from main_pi.py).
Commences training.
"""

approach = "projection"
name = "AC-2_5"


policy_iteration(approach, name, 2, 5, skip_decen=True)


