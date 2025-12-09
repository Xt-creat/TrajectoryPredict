'''
PyTorch visualization script for Social LSTM results

Visualizes predicted trajectories vs true trajectories
'''

import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import argparse


def plot_trajectories(true_trajs, pred_trajs, obs_length, save_path=None, show_plot=True):
    '''
    Function that plots the true trajectories and the trajectories predicted by the model
    
    params:
    true_trajs : numpy matrix with points of the true trajectories [traj_length, maxNumPeds, 5]
    pred_trajs : numpy matrix with points of the predicted trajectories [traj_length, maxNumPeds, 5]
    obs_length : Length of observed trajectory
    save_path : Path to save the figure (optional)
    show_plot : Whether to display the plot
    '''
    traj_length, maxNumPeds, _ = true_trajs.shape

    # Initialize figure
    plt.figure(figsize=(12, 10))
    ax = plt.gca()
    
    # Try to load background image if exists
    plot_dir = os.path.join(os.path.dirname(__file__), 'plot')
    bg_path = os.path.join(plot_dir, 'plot.png')
    if os.path.exists(bg_path):
        try:
            im = plt.imread(bg_path)
            ax.imshow(im, extent=[0, 1, 0, 1], alpha=0.3)
        except:
            pass

    traj_data = {}
    # For each frame/each point in all trajectories
    for i in range(traj_length):
        pred_pos = pred_trajs[i, :]
        true_pos = true_trajs[i, :]

        # For each pedestrian
        for j in range(maxNumPeds):
            if true_pos[j, 0] == 0:
                # Not a ped
                continue
            elif pred_pos[j, 0] == 0:
                # Not a ped
                continue
            else:
                # If he is a ped
                if true_pos[j, 1] > 1 or true_pos[j, 1] < 0:
                    continue
                elif true_pos[j, 2] > 1 or true_pos[j, 2] < 0:
                    continue

                if (j not in traj_data) and i < obs_length:
                    traj_data[j] = [[], [], []]  # [true_traj, pred_traj, timesteps]

                if j in traj_data:
                    traj_data[j][0].append(true_pos[j, 1:3])  # true position
                    traj_data[j][1].append(pred_pos[j, 1:3])  # pred position
                    traj_data[j][2].append(i)  # timestep

    # Plot trajectories
    colors = plt.cm.tab20(np.linspace(0, 1, len(traj_data)))
    for idx, (j, traj_info) in enumerate(traj_data.items()):
        true_traj_ped = traj_info[0]  # List of [x,y] elements
        pred_traj_ped = traj_info[1]
        timesteps = traj_info[2]
        
        # Convert to numpy arrays
        true_traj_ped = np.array(true_traj_ped)
        pred_traj_ped = np.array(pred_traj_ped)
        
        # Normalize coordinates if needed (from [-1, 1] to [0, 1])
        if true_traj_ped.min() < 0:
            true_traj_ped = (true_traj_ped + 1) / 2
            pred_traj_ped = (pred_traj_ped + 1) / 2
        
        # Split into observed and predicted parts
        obs_mask = np.array(timesteps) < obs_length
        pred_mask = np.array(timesteps) >= obs_length
        
        # Plot observed trajectory (true)
        if np.any(obs_mask):
            obs_true = true_traj_ped[obs_mask]
            ax.plot(obs_true[:, 0], obs_true[:, 1], 
                   color=colors[idx], linestyle='-', linewidth=2, 
                   marker='o', markersize=4, label=f'Ped {j} (Observed)' if idx < 10 else '')
        
        # Plot predicted trajectory (true)
        if np.any(pred_mask):
            pred_true = true_traj_ped[pred_mask]
            ax.plot(pred_true[:, 0], pred_true[:, 1], 
                   color=colors[idx], linestyle='-', linewidth=2, 
                   marker='s', markersize=4, alpha=0.7)
        
        # Plot predicted trajectory (predicted)
        if np.any(pred_mask):
            pred_pred = pred_traj_ped[pred_mask]
            ax.plot(pred_pred[:, 0], pred_pred[:, 1], 
                   color=colors[idx], linestyle='--', linewidth=2, 
                   marker='x', markersize=5, alpha=0.8, label=f'Ped {j} (Predicted)' if idx < 10 else '')
        
        # Mark start and end points
        if len(true_traj_ped) > 0:
            ax.scatter(true_traj_ped[0, 0], true_traj_ped[0, 1], 
                      color=colors[idx], s=100, marker='*', zorder=5, edgecolors='black')
        if len(pred_traj_ped) > 0:
            ax.scatter(pred_traj_ped[-1, 0], pred_traj_ped[-1, 1], 
                      color=colors[idx], s=80, marker='D', zorder=5, edgecolors='black', alpha=0.7)

    # Add vertical line to separate observed and predicted
    if len(traj_data) > 0:
        # Find the last observed point
        last_obs_timestep = obs_length - 1
        ax.axvline(x=0.5 if ax.get_xlim()[0] < 0.5 else ax.get_xlim()[1], 
                  color='red', linestyle=':', linewidth=2, alpha=0.5, label='Observed/Predicted Boundary')

    ax.set_xlabel('X coordinate', fontsize=12)
    ax.set_ylabel('Y coordinate', fontsize=12)
    ax.set_title('Trajectory Prediction: True (solid) vs Predicted (dashed)', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax.set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()


def calculate_statistics(results, obs_length, maxNumPeds):
    '''
    Calculate statistics for all trajectories
    '''
    errors = []
    for true_traj, pred_traj, obs_len in results:
        error = np.zeros(len(true_traj) - obs_len)
        for i in range(obs_len, len(true_traj)):
            pred_pos = pred_traj[i, :]
            true_pos = true_traj[i, :]
            timestep_error = 0
            counter = 0
            for j in range(maxNumPeds):
                if true_pos[j, 0] == 0:
                    continue
                elif pred_pos[j, 0] == 0:
                    continue
                else:
                    if true_pos[j, 1] > 1 or true_pos[j, 1] < 0:
                        continue
                    elif true_pos[j, 2] > 1 or true_pos[j, 2] < 0:
                        continue
                    timestep_error += np.linalg.norm(true_pos[j, [1, 2]] - pred_pos[j, [1, 2]])
                    counter += 1
            if counter != 0:
                error[i - obs_len] = timestep_error / counter
        errors.append(np.mean(error))
    
    return {
        'mean_error': np.mean(errors),
        'std_error': np.std(errors),
        'min_error': np.min(errors),
        'max_error': np.max(errors),
        'median_error': np.median(errors)
    }


def main():
    parser = argparse.ArgumentParser(description='Visualize Social LSTM prediction results')
    parser.add_argument('--test_dataset', type=int, default=0,
                        help='Dataset index to visualize')
    parser.add_argument('--trajectory_idx', type=int, default=None,
                        help='Specific trajectory index to visualize (None for all)')
    parser.add_argument('--save_dir', type=str, default=None,
                        help='Directory to save figures (None to not save)')
    parser.add_argument('--show_plot', action='store_true', default=True,
                        help='Show plots interactively')
    parser.add_argument('--no_show', action='store_true',
                        help='Do not show plots (only save)')
    parser.add_argument('--max_trajectories', type=int, default=10,
                        help='Maximum number of trajectories to visualize')
    
    args = parser.parse_args()
    
    if args.no_show:
        args.show_plot = False
    
    # Get script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    save_directory = os.path.join(script_dir, 'save', str(args.test_dataset))
    results_path = os.path.join(save_directory, 'social_results.pkl')
    
    if not os.path.exists(results_path):
        print(f"Error: Results file not found at {results_path}")
        print("Please run social_sample.py first to generate results.")
        return
    
    # Load results
    print(f"Loading results from {results_path}")
    with open(results_path, 'rb') as f:
        results = pickle.load(f)
    
    print(f"Loaded {len(results)} trajectories")
    
    # Load config to get maxNumPeds
    config_path = os.path.join(save_directory, 'social_config.pkl')
    if os.path.exists(config_path):
        with open(config_path, 'rb') as f:
            config = pickle.load(f)
            maxNumPeds = config.maxNumPeds
            obs_length = results[0][2] if len(results) > 0 else 8
    else:
        maxNumPeds = 70
        obs_length = results[0][2] if len(results) > 0 else 8
    
    # Calculate statistics
    stats = calculate_statistics(results, obs_length, maxNumPeds)
    print("\n" + "="*50)
    print("Prediction Statistics:")
    print("="*50)
    print(f"Mean Error (ADE): {stats['mean_error']:.6f}")
    print(f"Std Error:        {stats['std_error']:.6f}")
    print(f"Min Error:         {stats['min_error']:.6f}")
    print(f"Max Error:         {stats['max_error']:.6f}")
    print(f"Median Error:      {stats['median_error']:.6f}")
    print("="*50 + "\n")
    
    # Create save directory if needed
    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
    
    # Visualize trajectories
    if args.trajectory_idx is not None:
        # Visualize specific trajectory
        if args.trajectory_idx >= len(results):
            print(f"Error: Trajectory index {args.trajectory_idx} out of range (0-{len(results)-1})")
            return
        
        true_traj, pred_traj, obs_len = results[args.trajectory_idx]
        save_path = os.path.join(args.save_dir, f'trajectory_{args.trajectory_idx}.png') if args.save_dir else None
        print(f"Visualizing trajectory {args.trajectory_idx}")
        plot_trajectories(true_traj, pred_traj, obs_len, save_path, args.show_plot)
    else:
        # Visualize multiple trajectories
        num_to_show = min(args.max_trajectories, len(results))
        print(f"Visualizing {num_to_show} trajectories...")
        
        for i in range(num_to_show):
            true_traj, pred_traj, obs_len = results[i]
            save_path = os.path.join(args.save_dir, f'trajectory_{i}.png') if args.save_dir else None
            print(f"Visualizing trajectory {i+1}/{num_to_show}")
            plot_trajectories(true_traj, pred_traj, obs_len, save_path, args.show_plot and i == 0)
            # Only show first plot interactively, save the rest


if __name__ == '__main__':
    main()

