from models.simple_connected_multiple import CNetwork


def compare_policies(list_of_policies, num_agents, orchard_length):
    # in this case, list_of_policies contains the folder name with NN weights

    # step 1 -> initialize the plot

    for policy in list_of_policies:
        if "C-RANDOM" in policy:
            network = CNetwork(orchard_length, 0.0002, 0.99)
            agent_list = []
            for i in range(num_agents):





if __name__ == '__main__':
    pass
