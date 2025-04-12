"""
Evolutionary Dynamics with Vector Payoffs - Flask Backend

This application serves as the backend for the Evolutionary Game Theory visualization tool.
It provides routes for:
1. Rendering the main HTML page
2. Generating Markov Chain visualization data
3. Running ODE simulations and returning results as images
"""

from flask import Flask, request, jsonify, render_template, send_file
from ode import solve_ODE_async_euler, binary_to_CD
# Import the function directly instead of using subprocess
from mc_data import generate_markov_visualization
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
import base64
from io import BytesIO
from PIL import Image
import numpy as np
import matplotlib
# from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

matplotlib.use('Agg')  # 使用非GUI后端


app = Flask(__name__)


@app.route('/')
def index():
    """Render the main index page with a link to the vectorpayoff page."""
    return render_template('main_index.html')


@app.route('/vectorpayoff')
def vectorpayoff():
    """Render the vector payoff visualization page."""
    return render_template('vectorpayoff.html')


@app.route('/generate', methods=['POST'])
def generate_json():
    """
    Generate JSON data for Markov Chain visualization.
    
    This endpoint receives game parameters and population size,
    calls functions to generate the Markov Chain data,
    and returns nodes and edges for visualization.
    """
    data = request.json
    K = int(data.get('K'))
    N = int(data.get('N'))

    payoff_matrices = []
    for i in range(1, K + 1):
        R = eval(data.get(f'R_{i}'))
        S = eval(data.get(f'S_{i}'))
        T = eval(data.get(f'T_{i}'))
        P = eval(data.get(f'P_{i}'))
        payoff_matrices.append((R, S, T, P))

    try:
        # Call the function directly instead of using subprocess
        nodes, edges = generate_markov_visualization(K, payoff_matrices, N)
        return jsonify({'nodes': nodes, 'edges': edges})
    except Exception as e:
        print("Error generating Markov visualization:", e)
        return jsonify({'error': f'Failed to generate visualization: {str(e)}'}), 500


@app.route('/submit', methods=['POST'])
def submit():
    """
    Run ODE simulation and return visualization.
    
    This endpoint processes form data for ODE simulation parameters,
    runs the simulation, generates a visualization (either simplex or stackplot),
    and returns the result as a base64-encoded image.
    """
    # 获取表单参数
    data = request.json
    print('Submit function is called')
    Z = data.get('Z')
    Z = int(Z)
    # lambda_values = data.get('lambda')
    # x0 = data.get('x0')
    # lambda_values = lambda_values.split(' '); lambda_values = np.array(lambda_values).astype(np.float32)
    # M_matrix = M_matrix.split(' '); M_matrix = np.array(M_matrix).reshape(2**Z, 2**Z).astype(np.float32)
    # x0 = x0.split(' '); x0 = np.array(x0).astype(np.float32)

    x0 = np.zeros(2**Z)
    lambda_values = np.zeros(2**Z)
    for i in range(2**Z):
        x0[i] = eval(data.get(f'x0_{i}'))
        lambda_values[i] = eval(data.get(f'lambda_{i}'))

    Time = eval(data.get('Time'))

    u = eval(data.get('u'))
    v = eval(data.get('v'))

    tag = str(data.get('tag'))

    # # 获取并拼接 M 参数
    # M_values = []
    # for i in range(2 ** Z):
    #     M_value = data.get(f'M_{i}')
    #     if M_value is not None:
    #         M_values.append(M_value)

	# 确保这些字段存在
    # print("Received values:", Z, lambda_values, M_matrix, x0, Time)

    payoff_matrices = []
    for i in range(1, Z + 1):
        R = eval(data.get(f'RR_{i}'))
        S = eval(data.get(f'SS_{i}'))
        T = eval(data.get(f'TT_{i}'))
        P = eval(data.get(f'PP_{i}'))
        payoff_matrices.append((R, S, T, P))

    
    

    K = Z
    payoff_matrices = np.array(payoff_matrices).reshape(Z, 2, 2).astype(np.float32)

    
    print('Payoff matrices:', payoff_matrices)
    print('Z:', Z)
    print('Lambda:', lambda_values)
    # print('M_matrix:', M_matrix)
    print('x_0:', x0)
    print('Time:', Time)
    
    
    x_0_samples = np.array([
                            # [0.5, 0.5, 0, 0],
                            # [0.5, 0, 0.5, 0],
                            # [0.5, 0, 0, 0.5],
                            # [0, 0.5, 0.5, 0],
                            # [0, 0.5, 0, 0.5],
                            # [0, 0, 0.5, 0.5],
                            
                            [1/3, 1/3, 1/3, 0],
                            [1/3, 1/3, 0, 1/3],
                            [1/3, 0, 1/3, 1/3],
                            [0, 1/3, 1/3, 1/3],
                            
                            [0.25, 0.25, 0.25, 0.25],
                            # [0.7, 0.1, 0.1, 0.1],
                            # [0.1, 0.7, 0.1, 0.1],
                            # [0.1, 0.1, 0.7, 0.1],
                            # [0.1, 0.1, 0.1, 0.7]
                            ])
    
    if tag == 'simplex':
        return generate_simplex_visualization(K, payoff_matrices, lambda_values, u, v, x0, Time, x_0_samples)

    if tag == 'stackplot':
        return generate_stackplot_visualization(K, payoff_matrices, lambda_values, u, v, x0, Time)


def generate_simplex_visualization(K, payoff_matrices, lambda_values, u, v, x_0_input, Time, x_0_samples):
    """Generate a 3D simplex visualization for 2-context games."""
    dt = 0.01
    factor = 1
    
    # Define the vertices of the simplex
    x1 = np.array([1, 0, 0]) * factor
    x2 = np.array([-1/2, np.sqrt(3)/2, 0]) * factor
    x3 = np.array([-1/2, -np.sqrt(3)/2, 0]) * factor
    x4 = np.array([0, 0, 1]) * factor

    # List of vertices for the four faces of the simplex
    faces = [
        [x1, x2, x3],
        [x1, x2, x4],
        [x2, x3, x4],
        [x1, x3, x4]
    ]

    # Plot the simplex in 3D
    fig = plt.figure(dpi=1000)
    ax = fig.add_subplot(111, projection='3d')
    ax.set_box_aspect([1, 1, 1])  # Aspect ratio is 1:1:1

    # Add faces to plot
    for face in faces:
        color_poly = 'white'
        poly = Poly3DCollection([face], color=color_poly, alpha=0.05, edgecolor='k')
        ax.add_collection3d(poly)

    # Add vertex labels
    ax.text(*(x1+np.array([0, 0.025, -0.175]) * factor), s='CC', color='black', ha='center', fontsize=12)
    ax.text(*(x2+np.array([-0.175, -0.1, 0]) * factor), s='CD', color='black', ha='center', fontsize=12)
    ax.text(*(x3+np.array([-0.2, -0.1, 0]) * factor), s='DC', color='black', ha='center', fontsize=12)
    ax.text(*(x4+np.array([0, 0, 0.05]) * factor), s='DD', color='black', ha='center', fontsize=12)

    vertices = np.vstack([x1, x2, x3, x4])
    
    # Plot trajectories for predefined initial states
    for x_0 in x_0_samples:
        # Compute the 3D coordinates of each point at each time step
        solution_t, solution_ODE = solve_ODE_async_euler(K, payoff_matrices, lambda_values, u, v, x_0, Time, dt)
        points = np.dot(solution_ODE.T, vertices)

        # Use a consistent color for all predefined trajectories
        color = plt.get_cmap('YlGnBu')(0.75)
        
        # Plot trajectory
        ax.plot(points[:, 0], points[:, 1], points[:, 2], color=color, linewidth=0.25, zorder=1000)
        
        # Mark the ending point
        ax.scatter(points[-1, 0], points[-1, 1], points[-1, 2], color=color, s=1.5, edgecolors=color, marker='o', zorder=5)

        # Add an arrow at the midpoint of the trajectory
        midpoint = int(0.05 * len(points))
        endpoint = int(0.075 * len(points))
        arrow_start = points[midpoint]
        arrow_end = points[endpoint] - arrow_start  # Direction vector
        ax.quiver(
            arrow_start[0], arrow_start[1], arrow_start[2],
            arrow_end[0], arrow_end[1], arrow_end[2],
            color=color, arrow_length_ratio=0.75, length=0.04, normalize=True, linewidths=0.6, capstyle='butt', zorder=10000
        )
    
    # Plot trajectory for custom initial state
    if x_0_input is not None:
        solution_t, solution_ODE = solve_ODE_async_euler(K, payoff_matrices, lambda_values, u, v, x_0_input, Time, dt)
        points = np.dot(solution_ODE.T, vertices)
        ax.plot(points[:, 0], points[:, 1], points[:, 2], color="#D62728", linewidth=0.25, zorder=2000)
        ax.scatter(points[-1, 0], points[-1, 1], points[-1, 2], color="#D62728", s=1.5, edgecolors="#D62728", marker='o', zorder=5000)

        # Add an arrow at the midpoint of the trajectory
        midpoint = int(0.05 * len(points))
        endpoint = int(0.075 * len(points))
        arrow_start = points[midpoint]
        arrow_end = points[endpoint] - arrow_start  # Direction vector
        ax.quiver(
            arrow_start[0], arrow_start[1], arrow_start[2],
            arrow_end[0], arrow_end[1], arrow_end[2],
            color="#D62728", arrow_length_ratio=0.75, length=0.04, normalize=True, linewidths=0.6, capstyle='butt', zorder=10000
        )
    
    # Calculate and plot equilibrium surfaces
    flag_1 = False
    flag_2 = False

    # Equilibrium surface for context 1
    a1, b1, c1, d1 = payoff_matrices[0].flatten()
    if a1 - b1 - c1 + d1 != 0:
        e1 = (d1 - b1) / (a1 - b1 - c1 + d1)
        if 0 < e1 < 1:
            flag_1 = True
    
    # Equilibrium surface for context 2
    a2, b2, c2, d2 = payoff_matrices[1].flatten()
    if a2 - b2 - c2 + d2 != 0:
        e2 = (d2 - b2) / (a2 - b2 - c2 + d2)
        if 0 < e2 < 1:
            flag_2 = True
    
    # Plot equilibrium surface for context 1
    if flag_1:
        x1_lin = np.linspace(0, e1, 100)
        x3_lin = np.linspace(0, 1-e1, 100)
        x1_mesh, x3_mesh = np.meshgrid(x1_lin, x3_lin)
        x2_mesh = e1 - x1_mesh
        x4_mesh = 1 - e1 - x3_mesh
        X_mesh_1 = x1_mesh * x1[0] + x2_mesh * x2[0] + x3_mesh * x3[0] + x4_mesh * x4[0]
        Y_mesh_1 = x1_mesh * x1[1] + x2_mesh * x2[1] + x3_mesh * x3[1] + x4_mesh * x4[1]
        Z_mesh_1 = x1_mesh * x1[2] + x2_mesh * x2[2] + x3_mesh * x3[2] + x4_mesh * x4[2]
        ax.plot_surface(X_mesh_1, Y_mesh_1, Z_mesh_1, alpha=0.1, color='red', edgecolor='none', 
                      label=r'Equilibrium surface of context 1: $x_{\text{CC}} + x_{\text{CD}} = x_1^{\ast}$')

    # Plot equilibrium surface for context 2
    if flag_2:
        x1_lin = np.linspace(0, e2, 100)
        x2_lin = np.linspace(0, 1-e2, 100)
        x1_mesh, x2_mesh = np.meshgrid(x1_lin, x2_lin)
        x3_mesh = e2 - x1_mesh
        x4_mesh = 1 - e2 - x2_mesh
        X_mesh_2 = x1_mesh * x1[0] + x2_mesh * x2[0] + x3_mesh * x3[0] + x4_mesh * x4[0]
        Y_mesh_2 = x1_mesh * x1[1] + x2_mesh * x2[1] + x3_mesh * x3[1] + x4_mesh * x4[1]
        Z_mesh_2 = x1_mesh * x1[2] + x2_mesh * x2[2] + x3_mesh * x3[2] + x4_mesh * x4[2]
        ax.plot_surface(X_mesh_2, Y_mesh_2, Z_mesh_2, alpha=0.1, color='green', edgecolor='none', 
                      label=r'Equilibrium surface of context 2: $x_{\text{CC}} + x_{\text{DC}} = x_2^{\ast}$')

    # Plot intersection of equilibrium surfaces
    if flag_1 and flag_2:
        if e1 <= 0.5 and e2 <= 0.5:
            x1_lin = np.linspace(0, np.min([e1, e2]), 100)
            x2_lin = e1 - x1_lin
            x3_lin = e2 - x1_lin
            x4_lin = 1 - x1_lin - x2_lin - x3_lin
        elif e1 <= 0.5 and e2 > 0.5:
            x2_lin = np.linspace(0, np.min([e1, 1-e2]), 100)
            x1_lin = e1 - x2_lin
            x3_lin = e2 - x1_lin
            x4_lin = 1 - x1_lin - x2_lin - x3_lin
        elif e1 > 0.5 and e2 <= 0.5:
            x3_lin = np.linspace(0, np.min([1-e1, e2]), 100)
            x1_lin = e2 - x3_lin
            x2_lin = e1 - x1_lin
            x4_lin = 1 - x1_lin - x2_lin - x3_lin
        else:
            x4_lin = np.linspace(0, np.min([1-e1, 1-e2]), 100)
            x3_lin = 1-e1 - x4_lin
            x1_lin = e2 - x3_lin
            x2_lin = e1 - x1_lin
        
        X_lin = x1_lin * x1[0] + x2_lin * x2[0] + x3_lin * x3[0] + x4_lin * x4[0]
        Y_lin = x1_lin * x1[1] + x2_lin * x2[1] + x3_lin * x3[1] + x4_lin * x4[1]
        Z_lin = x1_lin * x1[2] + x2_lin * x2[2] + x3_lin * x3[2] + x4_lin * x4[2]
        ax.plot(X_lin, Y_lin, Z_lin, color='black', linewidth=0.5, linestyle='--', alpha=0.5, 
              label='Intersection of the two equilibrium surfaces')

    # Create dummy points for the legend
    blue_line = ax.plot([], [], [], color=plt.get_cmap('YlGnBu')(0.75), label='Trajectory from predefined initial state', linewidth=0.5)
    blue_final_state = ax.scatter([], [], [], color=plt.get_cmap('YlGnBu')(0.75), s=1.5, edgecolors=plt.get_cmap('YlGnBu')(0.75), marker='o', label='Final state from predefined initial state', zorder=5)
    red_line = ax.plot([], [], [], color="#D62728", label='Trajectory from custom initial state', linewidth=0.5)
    red_final_state = ax.scatter([], [], [], color="#D62728", s=1.5, edgecolors="#D62728", marker='o', label='Final state from custom initial state', zorder=5000)
    
    # Add the legend
    ax.legend(loc='upper right', fontsize=4)

    # Set axis limits and view orientation
    ax.set_xlim(np.array([-1, 1]) * factor)
    ax.set_ylim(np.array([-1, 1]) * factor)
    ax.set_zlim(np.array([-0.5, 1]) * factor)
    ax.view_init(elev=20, azim=260)
    ax.axis('off')
    
    # Save figure to buffer
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)

    # Crop the image for better visualization
    image = Image.open(buffer)
    width, height = image.size
    cropped_image = image.crop((0.3 * width, 0.1 * height, 0.8 * width, 0.8 * height))

    # Reset buffer and save cropped image
    buffer.seek(0)
    buffer.truncate()
    cropped_image.save(buffer, format='PNG')
    buffer.seek(0)

    # Encode image to base64 and return
    image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
    return jsonify({'image': image_base64})


def generate_stackplot_visualization(K, payoff_matrices, lambda_values, u, v, x0, Time):
    """Generate a stackplot visualization showing strategy frequencies over time."""
    # Run ODE simulation
    solution_t, solution_ODE = solve_ODE_async_euler(K, payoff_matrices, lambda_values, u, v, x0, Time, dt=0.005)
    
    # Set up plot
    sns.set_style('whitegrid')
    m = 2 ** K
    fig = plt.figure(figsize=(12, 8))
    fs = 25
    fig.patch.set_facecolor('white')
    ax = plt.subplot(111)

    # Define colors based on number of contexts
    if K == 2:
        colormap = ['#88b7e0', '#a1d99c', '#ffd59a', '#ff9697']
    else:
        colormap = ['#88b7e0', '#a1d99c', '#ffd59a', '#f6f8b1', '#ffc6ff', '#bdb2ff', '#9bdad6', '#ff9697']

    # Create stackplot
    ax.stackplot(solution_t, solution_ODE, colors=colormap, 
                labels=[binary_to_CD(i, K) for i in range(m)], alpha=0.8)

    # Configure axes and labels
    ax.set_xlabel('Time step', fontsize=fs - 2)
    ax.set_ylabel('Frequency', fontsize=fs - 2)
    ax.set_ylim(-0.05, 1.05)
    ax.tick_params(axis='both', which='major', labelsize=fs - 4)

    plt.title('Solution of ODE', fontsize=fs)
    plt.xlabel('Time', fontsize=fs-2)
    plt.ylabel('Frequency', fontsize=fs-2)
    plt.grid(True)
    plt.legend(loc='upper right', fancybox=True, fontsize=fs - 4)

    # Save to buffer and encode as base64
    buffer = BytesIO()
    plt.savefig(buffer, format='png', bbox_inches='tight', dpi=1000)
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
    
    return jsonify({'image': image_base64})


@app.route('/vectorpayoff/generate', methods=['POST'])
def vectorpayoff_generate_json():
    """
    Generate JSON data for Markov Chain visualization.
    
    This endpoint receives game parameters and population size,
    calls functions to generate the Markov Chain data,
    and returns nodes and edges for visualization.
    """
    data = request.json
    K = int(data.get('K'))
    N = int(data.get('N'))

    payoff_matrices = []
    for i in range(1, K + 1):
        R = eval(data.get(f'R_{i}'))
        S = eval(data.get(f'S_{i}'))
        T = eval(data.get(f'T_{i}'))
        P = eval(data.get(f'P_{i}'))
        payoff_matrices.append((R, S, T, P))

    try:
        # Call the function directly instead of using subprocess
        nodes, edges = generate_markov_visualization(K, payoff_matrices, N)
        return jsonify({'nodes': nodes, 'edges': edges})
    except Exception as e:
        print("Error generating Markov visualization:", e)
        return jsonify({'error': f'Failed to generate visualization: {str(e)}'}), 500


@app.route('/vectorpayoff/submit', methods=['POST'])
def vectorpayoff_submit():
    """
    Run ODE simulation and return visualization.
    
    This endpoint processes form data for ODE simulation parameters,
    runs the simulation, generates a visualization (either simplex or stackplot),
    and returns the result as a base64-encoded image.
    """
    # 获取表单参数
    data = request.json
    print('Submit function is called')
    Z = data.get('Z')
    Z = int(Z)
    # lambda_values = data.get('lambda')
    # x0 = data.get('x0')
    # lambda_values = lambda_values.split(' '); lambda_values = np.array(lambda_values).astype(np.float32)
    # M_matrix = M_matrix.split(' '); M_matrix = np.array(M_matrix).reshape(2**Z, 2**Z).astype(np.float32)
    # x0 = x0.split(' '); x0 = np.array(x0).astype(np.float32)

    x0 = np.zeros(2**Z)
    lambda_values = np.zeros(2**Z)
    for i in range(2**Z):
        x0[i] = eval(data.get(f'x0_{i}'))
        lambda_values[i] = eval(data.get(f'lambda_{i}'))

    Time = eval(data.get('Time'))

    u = eval(data.get('u'))
    v = eval(data.get('v'))

    tag = str(data.get('tag'))

    # # 获取并拼接 M 参数
    # M_values = []
    # for i in range(2 ** Z):
    #     M_value = data.get(f'M_{i}')
    #     if M_value is not None:
    #         M_values.append(M_value)

	# 确保这些字段存在
    # print("Received values:", Z, lambda_values, M_matrix, x0, Time)

    payoff_matrices = []
    for i in range(1, Z + 1):
        R = eval(data.get(f'RR_{i}'))
        S = eval(data.get(f'SS_{i}'))
        T = eval(data.get(f'TT_{i}'))
        P = eval(data.get(f'PP_{i}'))
        payoff_matrices.append((R, S, T, P))

    
    

    K = Z
    payoff_matrices = np.array(payoff_matrices).reshape(Z, 2, 2).astype(np.float32)

    
    print('Payoff matrices:', payoff_matrices)
    print('Z:', Z)
    print('Lambda:', lambda_values)
    # print('M_matrix:', M_matrix)
    print('x_0:', x0)
    print('Time:', Time)
    
    
    x_0_samples = np.array([
                            # [0.5, 0.5, 0, 0],
                            # [0.5, 0, 0.5, 0],
                            # [0.5, 0, 0, 0.5],
                            # [0, 0.5, 0.5, 0],
                            # [0, 0.5, 0, 0.5],
                            # [0, 0, 0.5, 0.5],
                            
                            [1/3, 1/3, 1/3, 0],
                            [1/3, 1/3, 0, 1/3],
                            [1/3, 0, 1/3, 1/3],
                            [0, 1/3, 1/3, 1/3],
                            
                            [0.25, 0.25, 0.25, 0.25],
                            # [0.7, 0.1, 0.1, 0.1],
                            # [0.1, 0.7, 0.1, 0.1],
                            # [0.1, 0.1, 0.7, 0.1],
                            # [0.1, 0.1, 0.1, 0.7]
                            ])
    
    if tag == 'simplex':
        return generate_simplex_visualization(K, payoff_matrices, lambda_values, u, v, x0, Time, x_0_samples)

    if tag == 'stackplot':
        return generate_stackplot_visualization(K, payoff_matrices, lambda_values, u, v, x0, Time)


if __name__ == '__main__':
    app.run(debug=True)