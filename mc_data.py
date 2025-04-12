import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# from matplotlib.colors import Normalize
# from matplotlib.cm import ScalarMappable
# # mlines, mpatches are used for legend
# import matplotlib.lines as mlines
# import matplotlib.patches as mpatches
# import seaborn as sns
# sns.set_style('ticks')
import os
import csv
import math
from tqdm import tqdm
# from decimal import Decimal, getcontext
# from matplotlib.colors import LinearSegmentedColormap

import sys



from tqdm import tqdm
import os
import random
import csv
import math
from pprint import pprint
import json

import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# sns.set(style='whitegrid', palette='colorblind', context='talk')
# sns.set_style('ticks')

# from matplotlib.gridspec import GridSpec
# from mpl_toolkits.mplot3d import Axes3D
# from matplotlib.colors import Normalize
# from matplotlib.cm import ScalarMappable
# # mlines, mpatches are used for legend
# import matplotlib.lines as mlines
# import matplotlib.patches as mpatches


def is_bit_zero(x: int, i: int, K: int) -> bool:
    """
    Check if the i-th bit of x is 0 (from left to right).

    Parameters
    ----------
    x : int
        The binary number to test.
    i : int
        The position of the bit to test (from left to right, starting from 0).
    K : int
        The number of bits in the binary representation of x.
    Returns
    -------
    bool
        True if the i-th bit is 0, False otherwise.
    """
    num_str = bin(x)[2:]
    if len(num_str) < K: # Add leading zeros if necessary
        num_str = '0' * (K - len(num_str)) + num_str
    return num_str[i] == '0'


def generate_payoff_matrices(games: list[str], rs: list[float], N: int, K: int) -> np.ndarray:
    '''
    Generate the payoff matrices for each game.

    Parameters
    ----------
    games : list[str]
        The list of game types ('PD', 'SD' or 'SH') for each of the K games.
    rs : list[float]
        The list of r values for each of the K games.
    N : int
        The population size.
    K : int
        The number of payoff matrices to generate.

    Returns
    -------
    np.ndarray
        A Kx2x2 array of the payoff matrices of each game.
    '''
    valid_games = {'PD', 'SD', 'SH'}
    if not len(games) == K or not len(rs) == K:
        raise ValueError('Length of games and rs must be equal to K.')
    if not all(game in valid_games for game in games):
        raise ValueError('Invalid game name.')
    if not all(0 <= r <= 1 for r in rs):
        raise ValueError('Invalid r value.')

    payoff_matrices = np.zeros((K, 2, 2))

    for k in range(K):
        game = games[k]
        r = rs[k]

        if game == 'PD':
            payoff_matrices[k, :, :] = np.array([[1, -r], 
                                                 [1 + r, 0]])
        elif game == 'SD':
            payoff_matrices[k, :, :] = np.array([[1, 1 - (r - 1/N)], 
                                                 [1 + (r - 1/N), 0]])
        elif game == 'SH':
            payoff_matrices[k, :, :] = np.array([[1 + r * N / (N - 2), 0], 
                                                 [1, 1 - r * N / (N - 2)]])

    return payoff_matrices


def calculate_proportion(strategy_state: np.ndarray, K: int = 2) -> np.ndarray:
    '''
    Calculate the proportion of C and D strategies in the population.

    Parameters
    ----------
    strategy_state : np.ndarray
        A 1D array of binary numbers representing the number of each strategy in the population.
    K : int
        The number of games.

    Returns
    -------
    np.ndarray
        A 2D array where the first column is the proportion of C strategies and the second column is the proportion of D strategies.
	'''
    N = np.sum(strategy_state)
    proportion = np.zeros((K,2)) # proportion of C and D strategies

    for i in range(K):
        C_population = np.sum([x for j, x in enumerate(strategy_state) if is_bit_zero(j, i, K)])
        D_population = N - C_population
        proportion[i] = np.array([C_population, D_population]) / N


    return proportion


def calculate_payoff(strategy_state: np.ndarray, payoff_matrices: np.ndarray) -> np.ndarray:
	'''
	Calculate the payoff vector of each strategy.

	Parameters
	----------
	strategy_state : np.ndarray
		A 1D array of binary numbers representing the number of each strategy in the population.
	payoff_matrices : np.ndarray
		A Kx2x2 array of the payoff matrix of each game
	
	Returns
	-------
	np.ndarray
		A 1D array of the payoff of each strategy.
	'''
	K = len(payoff_matrices) # number of games
	m = 2**K # number of strategies
	N = np.sum(strategy_state) # population size
	payoff_vec = np.zeros([m, K]) # payoff vector of each strategy

	prop = calculate_proportion(strategy_state, K) # proportion of C and D strategies
	for i in range(K):
		payoff_C = np.sum(payoff_matrices[i][0] * (prop[i] - np.array([1/N, 0]))) # payoff of C strategy in game i
		payoff_D = np.sum(payoff_matrices[i][1] * (prop[i] - np.array([0, 1/N]))) # payoff of D strategy in game i

		for j in range(m):
			if is_bit_zero(j, i, K): # test if the strategy of j in game i is C
				payoff_vec[j][i] = payoff_C
			else:
				payoff_vec[j][i] = payoff_D
	
	return payoff_vec


def generate_states(N: int, m: int) -> list[tuple[int]]:
    """
    Generate all possible states for a given N and m.

    Parameters
    ----------
    N : int
        The population size.
    m : int
        The number of strategies.

    Returns
    -------
    list of tuples
        A list containing tuples, where each tuple represents a state.

    Examples
    --------
    >>> generate_states(2, 2)
    [(0, 2), (1, 1), (2, 0)]
    >>> generate_states(2, 3)
    [(0, 0, 2), (0, 1, 1), (0, 2, 0), (1, 0, 1), (1, 1, 0), (2, 0, 0)]
    """
    def recursive_generate(remaining_N, remaining_m):
        if remaining_m == 1:
            return [(remaining_N,)]

        states = []
        for n in range(remaining_N + 1):
            for sub_state in recursive_generate(remaining_N - n, remaining_m - 1):
                states.append((n,) + sub_state)
        return states

    return recursive_generate(N, m)


def dominance_relation_rule(strategy_i: int, strategy_j: int, payoff_vec: np.ndarray) -> bool:
	"""
	Check if strategy_i is dominated by strategy_j in the given state.

	Parameters
	----------
	strategy_i : int
		The index of the focal strategy to be checked.
	strategy_j : int
		The index of the reference strategy to be compared with.
	payoff_vec : np.ndarray
		The payoff vector for the state.

	Returns
	-------
	bool
		True if strategy_i is dominated by strategy_j, False otherwise.
	"""
	return all (payoff_vec[strategy_i].round(8) <= payoff_vec[strategy_j].round(8)) and any (payoff_vec[strategy_i].round(8) < payoff_vec[strategy_j].round(8))


def generate_trans_matrix(state_space: list, payoff_matrices: np.ndarray, state_to_index: dict) -> np.ndarray:
	"""
	Generate the transition matrix for a given state space and payoff matrices according to the given update rule.
	
	Parameters
	----------
	state_space : list of tuples
		The state space.
	payoff_matrices : np.ndarray
		The payoff matrices for all states.
	
	Returns
	-------
	np.ndarray
		The transition matrix.
	"""
	K = payoff_matrices.shape[0] # Number of games
	m = 2 ** K # Number of strategies
	N = sum(state_space[0]) # Population size
	num_states = len(state_space)
	trans_matrix = np.zeros((num_states, num_states), dtype=np.float64)

	for state in tqdm(state_space, desc="Processing", ncols=100):
		self_trans_prob = np.float64(1)

		# Compute the payoff vectors for the given state
		payoff_vec = calculate_payoff(state, payoff_matrices)
		
		for i in range(m):
			for j in range(m):
				if state[i] > 0 and state[j] > 0:
					if dominance_relation_rule(i, j, payoff_vec):
						trans_prob = np.float64(state[i]) * np.float64(state[j]) / np.float64(N * (N - 1))

						# Calculate the next state after the transition from i to j
						next_state = list(state)
						next_state[i] -= 1
						next_state[j] += 1
						
						trans_matrix[state_to_index[tuple(state)], state_to_index[tuple(next_state)]] += trans_prob
						self_trans_prob -= trans_prob
		
		trans_matrix[state_to_index[state], state_to_index[state]] = self_trans_prob

	return trans_matrix


def reachable_states(trans_matrix: np.ndarray, target_states: list) -> list:
    """
    Find all states that can reach the target states.
	
	Parameters
    ----------
	trans_matrix : np.ndarray
		The transition matrix.
	target_states : list
		The target states in index form.
	
	Returns
	-------
	list
		The list of reachable states.
	"""
    n = len(trans_matrix)
    visited = [False] * n
    stack = target_states.copy()

    while stack:
        current_state = stack.pop()
        for prev_state in range(n):
            if trans_matrix[prev_state][current_state] > 0 and not visited[prev_state]:
                visited[prev_state] = True
                stack.append(prev_state)

    # Return all visited states including the target states themselves
    return [i for i, v in enumerate(visited) if v]


def tarjan(adjacency_matrix: np.ndarray) -> list:
    """
    Find the strongly connected components of a directed graph using Tarjan's algorithm.
    
    Parameters
    ----------
    adjacency_matrix : list of list of int
		The adjacency matrix of the directed graph. The element at index i, j is the weight of the edge from node i to node j.
        
    Returns
    -------
    list of list of int
		A list of connected components, where each connected component is a list of nodes.
    """
    n = len(adjacency_matrix)
    index_counter = [0]
    stack = []
    lowlinks = {}
    index = {}
    result = []
    
    def strongconnect(node):
        index[node] = index_counter[0]
        lowlinks[node] = index_counter[0]
        index_counter[0] += 1
        stack.append(node)
        
        successors = [i for i, is_adjacent in enumerate(adjacency_matrix[node]) if is_adjacent > 0]
        for successor in successors:
            if successor not in lowlinks:
                strongconnect(successor)
                lowlinks[node] = min(lowlinks[node], lowlinks[successor])
            elif successor in stack:
                lowlinks[node] = min(lowlinks[node], index[successor])
                
        if lowlinks[node] == index[node]:
            connected_component = []
            
            while True:
                successor = stack.pop()
                connected_component.append(successor)
                if successor == node: break
            
            result.append(connected_component)
    
    for node in range(n):
        if node not in lowlinks:
            strongconnect(node)
            
    return result


def generate_oscillating_class(trans_matrix: np.ndarray, index_to_state: dict) -> tuple:
	"""
	Generate the oscillating class of a transition matrix.
	
	Parameters
	---------
	trans_matrix : np.ndarray
		The transition matrix.

	Returns
	-------
	oscillating_classes_index: list
		osciallating classes in index form
	oscillating_classes_state: list
		osciallating classes in state form
	"""
	# Find the strongly connected components of the transition matrix
	strongly_connected_components_index = tarjan(trans_matrix)
	
	oscillating_classes_index = strongly_connected_components_index.copy()

	# Iterate through all the strongly connected components
	for scc in strongly_connected_components_index:
		total_transition_prob = 0
		for state in scc:
			# Calculate the total transition probability of the state
			total_transition_prob = np.sum(trans_matrix[state, scc])
			if total_transition_prob.round(8) < 1: 
				# Remove the strongly connected component from the oscillatory classes
				oscillating_classes_index.remove(scc)
				break  # Prevent repeated removal
	

	oscillating_classes_state = []
	for scc in oscillating_classes_index:
		scc_states = []
		for state_index in scc:
			scc_states.append(index_to_state[state_index])
		oscillating_classes_state.append(scc_states)

	return oscillating_classes_index, oscillating_classes_state


def state_properties(trans_matrix: np.ndarray, oscillating_classes_index: list, index_to_state: dict) -> tuple:
	"""
	Generate the dictionaries according to the state properties.

	Parameters
	----------
	trans_matrix : np.ndarray
		The transition matrix.
	oscillating_classes_index : list
		The oscillating classes in index form.
	index_to_state : dict
		The mapping from index to state.

	Returns
	-------
	states_reaching_classes_index : dict
		The states that can reach the oscillating classes in index form.
	state_reaching_classes_state : dict
		The states that can reach the oscillating classes in state form.
	class_absorbing_states_index : dict
		The oscillating classes that the states can be absorbed to in index form.
	class_absorbing_states_state : dict
		The oscillating classes that the states can be absorbed to in state form.
	states_deterministic_index : dict
		class_absorbing_states_index that are deterministic in index form.
	states_deterministic_state : dict
		class_absorbing_states_index that are deterministic in state form.
	states_probabilistic_index : dict
		class_absorbing_states_index that are probabilistic (semi-deterministic) in index form.
	states_probabilistic_state : dict
		class_absorbing_states_index that are probabilistic (semi-deterministic) in state form.
	cyclic_class_basin_size_index: dict
		The size of the cyclic class basin in index form.
	cyclic_class_basin_size_state: dict
		The size of the cyclic class basin in state form.
	"""
	

	states_reaching_classes_index = {}
	for osc_class_index in oscillating_classes_index:
		states_reaching_classes_index[tuple(osc_class_index)] = reachable_states(trans_matrix, osc_class_index)
	

	state_reaching_classes_state = {}

	for osc_class, reaching_states in states_reaching_classes_index.items():
		osc_class_as_states = tuple(index_to_state[idx] for idx in osc_class)
		reaching_states_as_states = [index_to_state[idx] for idx in reaching_states]
		state_reaching_classes_state[osc_class_as_states] = reaching_states_as_states

	
	class_absorbing_states_index = {}

	for osc_class, reaching_states in states_reaching_classes_index.items():
		for state in reaching_states:
			if state in class_absorbing_states_index:
				class_absorbing_states_index[state].append(osc_class)
			else:
				class_absorbing_states_index[state] = [osc_class]

	class_absorbing_states_state = {}

	for state_idx, osc_classes in class_absorbing_states_index.items():
		state_as_state = index_to_state[state_idx]
		osc_classes_as_states = [tuple(index_to_state[idx] for idx in osc_class) for osc_class in osc_classes]
		class_absorbing_states_state[state_as_state] = osc_classes_as_states



	states_deterministic_index = {}
	states_probabilistic_index = {}

	for state, osc_classes in class_absorbing_states_index.items():
		if len(osc_classes) == 1:
			states_deterministic_index[state] = osc_classes
		else:
			states_probabilistic_index[state] = osc_classes


	
	states_deterministic_state = {}
	states_probabilistic_state = {}

	for state_idx, osc_classes in states_deterministic_index.items():
		state_as_state = index_to_state[state_idx]
		osc_classes_as_states_list = [tuple(index_to_state[idx] for idx in osc_class) for osc_class in osc_classes]
		states_deterministic_state[state_as_state] = osc_classes_as_states_list


	for state_idx, osc_classes in states_probabilistic_index.items():
		state_as_state = index_to_state[state_idx]
		osc_classes_as_states = [tuple(index_to_state[idx] for idx in osc_class) for osc_class in osc_classes]
		states_probabilistic_state[state_as_state] = osc_classes_as_states

	cyclic_class_basin_size_index = {}
	cyclic_class_basin_size_state = {}

	for cyclic_class, reaching_states in states_reaching_classes_index.items():
		cyclic_class_basin_size_index[cyclic_class] = len(reaching_states)
	
	for cyclic_class, reaching_states in state_reaching_classes_state.items():
		cyclic_class_basin_size_state[cyclic_class] = len(reaching_states)

	return cyclic_class_basin_size_index, cyclic_class_basin_size_state, states_reaching_classes_index, state_reaching_classes_state, class_absorbing_states_index, class_absorbing_states_state, states_deterministic_index, states_deterministic_state, states_probabilistic_index, states_probabilistic_state





def generate_JSON(K: int, N: int, trans_matrix: np.ndarray, trans_matrix_power: np.ndarray, state_space: list, index_to_state: dict, oscillating_classes_state: list, states_probabilistic_state: dict):
    """
    Generate JSON files for the given parameters.

    Parameters
    ----------
    games : list of str
        The list of games.
    rs : list of float
        The list of r values.
    K : int
        The number of games.
    N : int
        The population size.
    trans_matrix : np.ndarray
        The transition matrix.
    trans_matrix_power : np.ndarray
        The transition matrix to the power of infinity.
    state_space : list of tuples
        The state space.
    index_to_state : dict
        The mapping from index to state.
    oscillating_classes_state : list
        The oscillating classes in state form.
    states_probabilistic_state : dict
        The probabilistic states in state form.
    """

    # game_key = '+'.join(games)
    # r_key = '_'.join(str(np.float32(1-r).round(2)) for r in rs)
    # directory = f"{game_key}/{N}_{r_key}"
    directory = "."

    # Ensure the directory exists
    os.makedirs(directory, exist_ok=True)



    # Process nodes to create nodes.json
    states_prob_list = list(states_probabilistic_state.keys())
    nodes = []
    index_to_color = {}
    for i in range(len(trans_matrix)):
        state = index_to_state[i]
        label = str(state)
        # color = '#4595FF'  # color of deterministic states ['#88b7e0', '#a1d99c', '#ffd59a', '#ff9697']
        color = '#88b7e0'
        for osc_class in oscillating_classes_state:
            if state in osc_class:
                # color = '#ec4566'  # color of oscillating classes / absorbing states
                color = '#FF9697'
                break
            elif state in states_prob_list:
                # color = '#9467bd'  # color of non-deterministic states
                # color = '#21cc21'
                color = '#A1D99C'
                break
        in_degree_1 = np.sum(trans_matrix[:, i])
        in_degree_infty = np.sum(trans_matrix_power[:, i])

        index_to_color[i] = color
        nodes.append({
            "id": i,
            "label": label,
            "color": color,
            "size_1": 10 + in_degree_1 / len(trans_matrix) * 20,
            "size_infty": 10 + np.sqrt(in_degree_infty / len(trans_matrix)) * 50
        })

    # ! we comment this out because we directly return the nodes
    # # Save nodes to JSON file
    # with open(f"{directory}/nodes.json", 'w') as f:
    #     json.dump(nodes, f, indent=4)

    # ! we comment this out because we do not need edges_1
    # # Process the transition matrix to extract edges for edges_1.json
    # edges = []
    # for i in range(len(trans_matrix)):
    #     for j in range(len(trans_matrix[i])):
    #         weight = trans_matrix[i][j]
    #         if weight != 0:
    #             source_state = index_to_state[i]
    #             target_state = index_to_state[j]
    #             # color = '#000000'
    #             # for osc_class in oscillating_classes_state:
    #             #     if source_state in osc_class and target_state in osc_class:
    #             #         color = '#d62728'
    #             #         weight += 5
    #             #         break
    #             color = index_to_color[i]

    #             if source_state != target_state:
    #                 edges.append({
    #                     "source": str(i),
    #                     "target": str(j),
    #                     "weight": weight,
    #                     "color": color
    #                 })

    # ! we comment this out because we directly return the edges
    # # Save edges to JSON file
    # with open(f"{directory}/edges_1.json", 'w') as f:
    #     json.dump(edges, f, indent=4)

    # Repeat the process for trans_matrix_power to create edges_infty.json
    edges = []
    for i in range(len(trans_matrix_power)):
        for j in range(len(trans_matrix_power[i])):
            weight = trans_matrix_power[i][j]
            if weight != 0:
                source_state = index_to_state[i]
                target_state = index_to_state[j]
                # color = '#000000'
                # for osc_class in oscillating_classes_state:
                #     if source_state in osc_class and target_state in osc_class:
                #         color = '#d62728'
                #         weight += 10
                #         break
                color = index_to_color[i]

                if source_state != target_state:
                    edges.append({
                        "source": i,
                        "target": j,
                        "weight": weight,
                        "color": color
                    })
                    
    # ! we comment this out because we directly return the edges
    # with open(f"{directory}/edges_infty.json", 'w') as f:
    #     json.dump(edges, f, indent=4)

    # Instead of writing to files, return the data
    return nodes, edges


def function_all_in_one(N: int, K: int, games: list[str], rs: list[float]):
	"""
	Generate the CSV files for the given parameters.

	Parameters
	----------
	N : int
		The population size.
	K : int
		The number of games.
	games : list of str
		The list of games.
	rs : list of float
		The list of r values.
	"""

	# Generate the payoff matrices
	payoff_matrices = generate_payoff_matrices(games, rs, N, K)

	# Generate the state space
	state_space = generate_states(N, 2 ** K)

	# Generate the mappings
	state_to_index = {state: idx for idx, state in enumerate(state_space)}
	index_to_state = {idx: state for idx, state in enumerate(state_space)}

	# Generate the transition matrix
	trans_matrix = generate_trans_matrix(state_space, payoff_matrices, state_to_index)
	
	# Generate the transition matrix to the power of infinity
	trans_matrix_power = np.linalg.matrix_power(trans_matrix, int(1e12))
	
	# Generate the oscillating classes
	oscillating_classes_index, oscillating_classes_state = generate_oscillating_class(trans_matrix, index_to_state)

	# Generate the state properties
	cyclic_class_basin_size_index, cyclic_class_basin_size_state, states_reaching_classes_index, state_reaching_classes_state, class_absorbing_states_index, class_absorbing_states_state, states_deterministic_index, states_deterministic_state, states_probabilistic_index, states_probabilistic_state = state_properties(trans_matrix, oscillating_classes_index, index_to_state)

	# Call generate_JSON and return the results directly
	nodes, edges_infty = generate_JSON(K, N, trans_matrix, trans_matrix_power, state_space, index_to_state, oscillating_classes_state, states_probabilistic_state)
	return nodes, edges_infty


# Add a new function to generate nodes and edges from payoff matrices
def generate_markov_visualization(K: int, payoff_matrices: list, N: int):
    """
    Generate nodes and edges for Markov chain visualization directly without saving to files.
    
    Parameters
    ----------
    K : int
        Number of game contexts.
    payoff_matrices : list
        List of payoff matrices (R, S, T, P) for each context.
    N : int
        Population size.
        
    Returns
    -------
    tuple
        (nodes, edges) data for visualization.
    """
    # Create games and rs lists from payoff matrices
    games = []
    rs = []
    
    for i in range(K):
        R, S, T, P = payoff_matrices[i]
        # Determine game type based on payoff values
        if T > R > P > S:
            game_type = 'PD'  # Prisoner's Dilemma
        elif R > T > S > P:
            game_type = 'SH'  # Stag Hunt
        elif T > R > S > P:
            game_type = 'SD'  # Snowdrift
        else:
            game_type = 'OT'  # Other
        
        games.append(game_type)
        
        # Calculate r value based on game type and payoffs
        if game_type == 'PD':
            r = (P - S) / (T - S)
        elif game_type == 'SH':
            r = (R - T) / (R - P)
        elif game_type == 'SD':
            r = (R - T) / (R - S)
        else:
            r = 0.5
        
        rs.append(r)
    
    # Call function_all_in_one to generate nodes and edges
    return function_all_in_one(N, K, games, rs)


# For backwards compatibility when run as a script
if __name__ == "__main__":
    # Parse command line arguments when run directly
    import sys
    import json
    
    if len(sys.argv) >= 4:
        K = int(sys.argv[1])
        payoff_matrices = json.loads(sys.argv[2])
        N = int(sys.argv[3])
        
        nodes, edges = generate_markov_visualization(K, payoff_matrices, N)
        
        # Write to files for backwards compatibility
        with open('nodes.json', 'w') as f:
            json.dump(nodes, f)
        
        with open('edges_infty.json', 'w') as f:
            json.dump(edges, f)
