import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
import sys
import json

from mpl_toolkits.mplot3d.art3d import Poly3DCollection


from flask import Flask, request, jsonify, render_template, send_file
import subprocess
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
import base64
from io import BytesIO
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 使用非GUI后端


# # def main():
# K = int(sys.argv[1])
# lambda_values = sys.argv[2].split(' ') # split the string by space
# Lambda = np.array(lambda_values).astype(np.float64)
# M_matrix = sys.argv[3].split(' ')
# M_matrix = np.array(M_matrix).reshape(2**K, 2**K).astype(np.float64)
# x_0 = sys.argv[4].split(' ')
# x_0 = np.array(x_0).astype(np.float64)
# T = int(sys.argv[5])

# payoff_matrices = json.loads(sys.argv[6])
# payoff_matrices = np.array(payoff_matrices).astype(np.float64).reshape(K, 2, 2)
# print('payoff_matrices:', payoff_matrices)
# print('K:', K)
# print('Lambda:', Lambda)
# print('M_matrix:', M_matrix)
# print('x_0:', x_0)
# print('T:', T)
# print('111111111111')
# 	# N = 4000
	# K = 2
	# infty = True
	# Lambda = np.ones(2**K)
	# M_matrix = np.zeros((2**K, 2**K))
	# M_matrix = np.array([[-1+1/4, 1/4, 1/4, 1/4], 
	# 					  [1/4, -1+1/4, 1/4, 1/4], 
	# 					  [1/4, 1/4, -1+1/4, 1/4], 
	# 					  [1/4, 1/4, 1/4, -1+1/4]])
	# M_matrix = np.array([[0, 0, 0, 0], 
	# 					  [0, 0, 0, 0], 
	# 					  [0, 0, 0, 0],
	# 					  [1, 0, 0, 0]])
	# x_0 = np.ones(2**K) / 2**K
	# T = 10

	# games = ['PD', 'PD', 'PD']
	# rs = [1 - 0.3, 1 - 0.4, 1 - 0.5]
	# N = 4000
	# K = 3
	# infty = True
	# Lambda = np.ones(2**K)
	# M_matrix = np.zeros((2**K, 2**K))
	# # M_matrix[-1] = [0, 0, 0, 0, 0, 0, 0, 0]
	# x_0 = np.ones(2**K) / 2**K
	# T = 10



# if K == 2:
# 	colormap = ['#88b7e0', '#a1d99c', '#ffd59a', '#ff9697']
# else:
# 	colormap = ['#88b7e0', '#a1d99c', '#ffd59a', '#f3f3f3', '#e6f598', '#f4cae4', '#9cd9d5', '#ff9697']


def binary_to_CD(binary: int, K: int) -> str:
	'''
	Convert a binary string to a CD string.

	Parameters
	----------
	binary : int
		The binary string to convert.
	K : int
		The number of bits in the binary representation.
	Returns
	-------
	str
		The CD string.
	'''
	binary_str = bin(binary)[2:]
	if len(binary_str) < K:
		binary_str = '0' * (K - len(binary_str)) + binary_str
	return ''.join(['C' if bit == '0' else 'D' for bit in binary_str])

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


def calculate_payoff(x: np.ndarray, payoff_matrices: np.ndarray) -> np.ndarray:
	'''
	Calculate the payoff vector of each strategy.

	Parameters
	----------
	x : np.ndarray
		A 1D array of the proportion of each strategy.
	payoff_matrices : np.ndarray
		A Kx2x2 array of the payoff matrix of each game
	
	Returns
	-------
	np.ndarray
		A 1D array of the payoff of each strategy.
	'''
	K = len(payoff_matrices) # number of games
	m = 2**K # number of strategies

	payoff_vec = np.zeros([m, K]) # payoff vector of each strategy


	for i in range(K):
		prop_C = np.sum(x * np.array([is_bit_zero(j, i, K) for j in range(m)]))
		prop = np.array([prop_C, 1-prop_C])
		payoff_C = np.sum(payoff_matrices[i][0] * prop) # payoff of C strategy in game i
		payoff_D = np.sum(payoff_matrices[i][1] * prop) # payoff of D strategy in game i

		for j in range(m):
			if is_bit_zero(j, i, K): # test if the strategy of j in game i is C
				payoff_vec[j][i] = payoff_C
			else:
				payoff_vec[j][i] = payoff_D
	
	return payoff_vec


def phi(i, j, payoff_matrices, x):
	payoff_vec = calculate_payoff(x, payoff_matrices)
	if all (payoff_vec[i].round(8) <= payoff_vec[j].round(8)) and any (payoff_vec[i].round(8) < payoff_vec[j].round(8)):
		return 1
	else:
		return 0
	

def solve_ODE_async_euler(K: int,
						  payoff_matrices: np.ndarray,
						  Lambda: np.ndarray,
						  u: int,
						  v: int,
						  x_0: np.ndarray,
						  T: int,
						  dt: float):
	M = 2 ** K  # number of strategies
	N_steps = int(T / dt)  # total number of time steps
	


	def system(x):
		x = np.array(x)
		dx = np.zeros(M)
		Phi = np.zeros([M, M])
		for i in range(M):
			for j in range(M):
				if i != j:
					Phi[i, j] = phi(i, j, payoff_matrices, x)

		s = 1
		s_ = 2

		x_s_operand = [x, [s]]
		x_sprime_operand = [x, [s_]]
		lambda_operand_1 = [Lambda, [s_]]
		Phi_operand_1 = [Phi, [s_, s]]
		operands_1 = x_s_operand + x_sprime_operand + lambda_operand_1 + Phi_operand_1 + [[s]]
		term_1 = np.einsum(*operands_1, optimize=True)

		lambda_operand_2 = [Lambda, [s]]
		Phi_operand_2 = [Phi, [s, s_]]
		operands_2 = x_sprime_operand + x_s_operand + lambda_operand_2 + Phi_operand_2 + [[s]]
		term_2 = np.einsum(*operands_2, optimize=True)

		if K == 1:
			term_u = -u * x + u * np.array([x[0b1], x[0b0]])
			term_v = 0

		if K == 2:
			term_u = -2 * u * x + u * np.array([x[0b01]+x[0b10], 
									  		   x[0b00]+x[0b11],
											   x[0b00]+x[0b11],
											   x[0b01]+x[0b10]])

			term_v = v * np.array([x[0b01]+x[0b10],
						           -2*x[0b01],
								   -2*x[0b10],
								   x[0b01]+x[0b10]])
		
		elif K == 3:
			term_u = -3 * u * x + u * np.array([x[0b001]+x[0b010]+x[0b100], # 000
									            x[0b000]+x[0b011]+x[0b101], # 001
												x[0b011]+x[0b000]+x[0b110], # 010
												x[0b010]+x[0b001]+x[0b111], # 011
												x[0b101]+x[0b110]+x[0b000], # 100
									            x[0b100]+x[0b111]+x[0b001], # 101
												x[0b111]+x[0b100]+x[0b010], # 110
												x[0b110]+x[0b101]+x[0b011]]) # 111
			
			term_v = v * np.array([x[0b001]+x[0b010]+x[0b100],
						           -2*x[0b001] + 1/2*x[0b011] + 1/2*x[0b101],
								   -2*x[0b010] + 1/2*x[0b011] + 1/2*x[0b110],
								   -2*x[0b011] + 1/2*x[0b001] + 1/2*x[0b010],
								   -2*x[0b100] + 1/2*x[0b101] + 1/2*x[0b110],
								   -2*x[0b101] + 1/2*x[0b001] + 1/2*x[0b100],
								   -2*x[0b110] + 1/2*x[0b010] + 1/2*x[0b100],
								   x[0b011]+x[0b101]+x[0b110]])

		dx = term_1 - term_2 + term_u + term_v

		return dx
	
	# Initialize x and time arrays
	x = np.copy(x_0)
	solution_ODE = np.zeros((M, N_steps + 1))  # store all states at each time step
	solution_ODE[:, 0] = x_0
	solution_t = np.linspace(0, T, N_steps + 1)
	
	# Euler method iteration
	for step in range(1, N_steps + 1):
		# Get dx using the system's ODEs
		dx = system(x)
		# Update x using Euler's method
		x = x + dt * dx
		# Store the updated x
		solution_ODE[:, step] = x
	
	return solution_t, solution_ODE







def plot_simplex_v2(K, payoff_matrices_infty, Lambda, u, v, x_0_samples, T, dt=0.005, x_0_input=None):
    # Define the vertices
    x1 = np.array([1, 0, 0])
    x2 = np.array([-1/2, np.sqrt(3)/2, 0])
    x3 = np.array([-1/2, -np.sqrt(3)/2, 0])
    x4 = np.array([0, 0, 1])

    # List of vertices for the four faces of the simplex
    faces = [
        [x1, x2, x3],
        [x1, x2, x4],
        [x2, x3, x4],
        [x1, x3, x4]
    ]

    # Plot the simplex in 3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_box_aspect([1, 1, 1])  # Aspect ratio is 1:1:1

    # Add faces to plot
    for face in faces:
        # color_poly = 'cyan'
        color_poly = 'white'
        poly = Poly3DCollection([face], color=color_poly, alpha=0.05, edgecolor='k')
        ax.add_collection3d(poly)

    # Scatter plot the vertices
    vertices = np.array([x1, x2, x3, x4])
    # ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], color=colormap, s=50, alpha=1, zorder=-100)
    
    ax.text(*(x1+np.array([0, 0.025, -0.175])), s='CC', color='black', ha='center', fontsize=12)
    ax.text(*(x2+np.array([-0.125, 0.15, -0.025])), s='CD', color='black', ha='center', fontsize=12)
    ax.text(*(x3+np.array([-0.3, -0.2, 0.075])), s='DC', color='black', ha='center', fontsize=12)
    ax.text(*(x4+np.array([0, 0, 0.075])), s='DD', color='black', ha='center', fontsize=12)

    vertices = np.vstack([x1, x2, x3, x4])
    
    for x_0 in x_0_samples:
	    # compute the 3D coordinates of each point at each time step
        solution_t, solution_ODE = solve_ODE_async_euler(K, payoff_matrices_infty, Lambda, u, v, x_0, T, dt)
        points = np.dot(solution_ODE.T, vertices)
        
        ax.plot(points[:, 0], points[:, 1], points[:, 2], color="b", linewidth=0.25, zorder=1000)

	    # # Mark the starting point with a hollow circle
        # ax.scatter(points[0, 0], points[0, 1], points[0, 2], color="blue", s=25, facecolors='none', edgecolors='blue', marker='o')
        
		# Mark the ending point with a filled circle
        ax.scatter(points[-1, 0], points[-1, 1], points[-1, 2], color="b", s=1.5, edgecolors="b", marker='o', zorder=5)
        
    if x_0_input is not None:
        solution_t, solution_ODE = solve_ODE_async_euler(K, payoff_matrices_infty, Lambda, u, v, x_0_input, T, dt)
        points = np.dot(solution_ODE.T, vertices)
        ax.plot(points[:, 0], points[:, 1], points[:, 2], color="r", linewidth=0.25, zorder=1000)
        ax.scatter(points[-1, 0], points[-1, 1], points[-1, 2], color="r", s=1.5, edgecolors="r", marker='o', zorder=5)


	    # # Add an arrow at the midpoint of the trajectory
        # midpoint = int(0.1 * len(points))
        # endpoint = int(0.15 * len(points))
        # arrow_start = points[midpoint]
        # arrow_end = points[endpoint] - arrow_start  # Direction vector
        # ax.quiver(
	    # 	arrow_start[0], arrow_start[1], arrow_start[2],
	    # 	arrow_end[0], arrow_end[1], arrow_end[2],
	    # 	color='blue', arrow_length_ratio=1.2, length=0.05, normalize=True
	    # )
        
	
    flag_1 = False
    flag_2 = False

    # equilibrium surface
    a1, b1, c1, d1 = payoff_matrices_infty[0].flatten()
    if a1 - b1 - c1 + d1 != 0:
        e1 = (d1 - b1) / (a1 - b1 - c1 + d1)
        if e1 > 0 and e1 < 1:
            flag_1 = True
    # equilibrium surface
    a2, b2, c2, d2 = payoff_matrices_infty[1].flatten()
    if a2 - b2 - c2 + d2 != 0:
        e2 = (d2 - b2) / (a2 - b2 - c2 + d2)
        if e2 > 0 and e2 < 1:
            flag_2 = True
        
    # if flag_1 and not flag_2:
    if flag_1:
        x1_lin = np.linspace(0, e1, 100)
        x3_lin = np.linspace(0, 1-e1, 100)
        x1_mesh, x3_mesh = np.meshgrid(x1_lin, x3_lin)
        x2_mesh = e1 - x1_mesh
        x4_mesh = 1 - e1 - x3_mesh
        X_mesh_1 = x1_mesh * x1[0] + x2_mesh * x2[0] + x3_mesh * x3[0] + x4_mesh * x4[0]
        Y_mesh_1 = x1_mesh * x1[1] + x2_mesh * x2[1] + x3_mesh * x3[1] + x4_mesh * x4[1]
        Z_mesh_1 = x1_mesh * x1[2] + x2_mesh * x2[2] + x3_mesh * x3[2] + x4_mesh * x4[2]
        ax.plot_surface(X_mesh_1, Y_mesh_1, Z_mesh_1, alpha=0.1, color='red', edgecolor='none')

    # if flag_2 and not flag_1:
    if flag_2:
        x1_lin = np.linspace(0, e2, 100)
        x2_lin = np.linspace(0, 1-e2, 100)
        x1_mesh, x2_mesh = np.meshgrid(x1_lin, x2_lin)
        x3_mesh = e2 - x1_mesh
        x4_mesh = 1 - e2 - x2_mesh
        X_mesh_2 = x1_mesh * x1[0] + x2_mesh * x2[0] + x3_mesh * x3[0] + x4_mesh * x4[0]
        Y_mesh_2 = x1_mesh * x1[1] + x2_mesh * x2[1] + x3_mesh * x3[1] + x4_mesh * x4[1]
        Z_mesh_2 = x1_mesh * x1[2] + x2_mesh * x2[2] + x3_mesh * x3[2] + x4_mesh * x4[2]
        print(X_mesh_2.shape, Y_mesh_2.shape, Z_mesh_2.shape)
        ax.plot_surface(X_mesh_2, Y_mesh_2, Z_mesh_2, alpha=0.1, color='green', edgecolor='none')

    if flag_1 and flag_2:
        x1_lin = np.linspace(0, np.min([e1, e2]), 100)
        x2_lin = e1 - x1_lin
        x3_lin = e2 - x1_lin
        x4_lin = 1 - x1_lin - x2_lin - x3_lin
        
        X_lin = x1_lin * x1[0] + x2_lin * x2[0] + x3_lin * x3[0] + x4_lin * x4[0]
        Y_lin = x1_lin * x1[1] + x2_lin * x2[1] + x3_lin * x3[1] + x4_lin * x4[1]
        Z_lin = x1_lin * x1[2] + x2_lin * x2[2] + x3_lin * x3[2] + x4_lin * x4[2]
        ax.plot(X_lin, Y_lin, Z_lin, color='black', linewidth=0.5, linestyle='--', alpha=0.5)

    # Set axis limits for better visualization
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-0.5, 1])
    
    # rotate the plot
    ax.view_init(elev=20, azim=190)
    # ax.view_init(elev=20, azim=130)

    ax.axis('off')
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)

    # 将图像转换为 Base64 字符串
    image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
    return jsonify({'image': image_base64})





# 	solution_t, solution_ODE = solve_ODE_async_euler(K, payoff_matrices, Lambda, M_matrix, x_0, T, dt=0.005)


# 	sns.set_style('whitegrid')
# 	m = 2 ** K
# 	fig = plt.figure(figsize = (12, 8))
# 	fs = 25
# 	fig.patch.set_facecolor('white')
# 	ax = plt.subplot(111)



# 	xlim = plt.xlim()
# 	color_equilibria = '#000000'

# 	# for i in range(K):
# 	# 	game = games[i]
# 	# 	r = rs[i]
		
# 	# 	if game == 'SH':
# 	# 		plt.axhline(1 - r, color=color_equilibria, linestyle=(0, (5, 5)), alpha=1, linewidth=1, zorder=1000)
# 	# 		plt.text(xlim[1], 1 - r, fr'$x_{{{i+1}}}^{{*}}$', verticalalignment='center', horizontalalignment='left', fontsize=fs)
# 	# 	elif game == 'SD':
# 	# 		plt.axhline(1 - r, color=color_equilibria, linestyle='-', alpha=1, linewidth=1, zorder=1000)
# 	# 		plt.text(xlim[1], 1 - r, fr'$x_{{{i+1}}}^{{*}}$', verticalalignment='center', horizontalalignment='left', fontsize=fs)


# 	ax.stackplot(solution_t, solution_ODE, colors=colormap, labels=[binary_to_CD(i, K) for i in range(m)], alpha=0.8)

# 	ax.set_xlabel('Time step', fontsize = fs - 2)
# 	ax.set_ylabel('Frequency', fontsize = fs - 2)
# 	ax.set_ylim(-0.05, 1.05)
# 	ax.tick_params(axis = 'both', which = 'major', labelsize = fs - 4)



# 	plt.title('Solution of ODE', fontsize = fs)
# 	plt.xlabel('Time', fontsize = fs-2)
# 	plt.ylabel('Frequency', fontsize = fs-2)
# 	plt.grid(True)
# 	plt.legend(loc = 'upper right', fancybox = True, fontsize = fs - 4)

# 	# plt.savefig('static/ode.png')
# 	# plt.close()


# if __name__ == '__main__':
# 	main()