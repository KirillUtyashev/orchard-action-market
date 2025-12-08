from typing import Dict
from tadd_helpers.env_functions import State

def get_reward_picker_1(state: State, acting_agent_idx: int) -> Dict[int, float]:
    """
    Picker gets 1.0. Others get 0.0.
    Sum = 1.0
    """
    num_agents = len(state._agents)
    rewards = {i: 0.0 for i in range(num_agents)}
    
    actor_pos = state.agent_position(acting_agent_idx)
    hit_apple = (state.apples[actor_pos[0], actor_pos[1]] > 0)
    
    if hit_apple:
        rewards[acting_agent_idx] = 1.0
        
    return rewards

def get_reward_other_1(state: State, acting_agent_idx: int) -> Dict[int, float]:
    """
    Picker gets 0.0. Others get 1/(N-1).
    Sum = 1.0
    """
    num_agents = len(state._agents)
    rewards = {i: 0.0 for i in range(num_agents)}
    
    actor_pos = state.agent_position(acting_agent_idx)
    hit_apple = (state.apples[actor_pos[0], actor_pos[1]] > 0)

    # Note: If N=1, no reward is issued because there are no others.
    if hit_apple and num_agents > 1:
        val = 1.0 / (num_agents - 1)
        for i in range(num_agents):
            if i != acting_agent_idx:
                rewards[i] = val
                
    return rewards

def get_reward_1_over_n(state: State, acting_agent_idx: int) -> Dict[int, float]:
    """
    Everyone gets 1/N.
    Sum = 1.0
    """
    num_agents = len(state._agents)
    rewards = {i: 0.0 for i in range(num_agents)}
    
    actor_pos = state.agent_position(acting_agent_idx)
    hit_apple = (state.apples[actor_pos[0], actor_pos[1]] > 0)
    
    if hit_apple:
        val = 1.0 / num_agents
        for i in range(num_agents):
            rewards[i] = val
            
    return rewards

def get_reward_minus1_2(state: State, acting_agent_idx: int) -> Dict[int, float]:
    """
    Picker gets -1.0. Others get 2/(N-1).
    Sum = -1 + 2 = 1.0
    """
    num_agents = len(state._agents)
    rewards = {i: 0.0 for i in range(num_agents)}
    
    actor_pos = state.agent_position(acting_agent_idx)
    hit_apple = (state.apples[actor_pos[0], actor_pos[1]] > 0)
    
    if hit_apple:
        rewards[acting_agent_idx] = -1.0
        if num_agents > 1:
            others_val = 2.0 / (num_agents - 1)
            for i in range(num_agents):
                if i != acting_agent_idx:
                    rewards[i] = others_val
                    
    return rewards