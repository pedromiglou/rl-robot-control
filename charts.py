#!/usr/bin/env python3

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scipy.interpolate import interp1d


RESULTS_FOLDER = "./results/fetch_reach_cartesian_discrete"

plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.size"] = 15


def create_line_chart(x, y, xlabel, ylabel, avg_line = False, save_path=None):
    # Create a figure and axis
    plt.figure(figsize = (10,5))
    
    # Plot the line chart
    plt.plot(x, y)

    if avg_line:
        x_new = []
        y_new = []
        
        for i in range(10, len(y), 20):
            x_new.append(x[i])
            y_new.append(np.average(y[i-10:i+10]))
        
        cubic_interpolation_model = interp1d(x_new, y_new, kind = "cubic")
        y_new=cubic_interpolation_model(x_new)

        plt.plot(x_new, y_new, color="red")

    
    # Set the x-axis label
    plt.xlabel(xlabel)
    
    # Set the y-axis label
    plt.ylabel(ylabel)
    
    # Show or save the chart
    if save_path:
        plt.tight_layout()
        plt.savefig(save_path)
    
    else:
        plt.show()


# Specify the file path
file_path = f"{RESULTS_FOLDER}/progress.csv"

# Read the CSV file
df = pd.read_csv(file_path)

# Call the function with the DataFrame
create_line_chart(df["time/episodes"], df["train/loss"], "Episodes", "Loss", avg_line = True, save_path=f"{RESULTS_FOLDER}/loss.svg")

create_line_chart(df["time/episodes"], df["rollout/ep_rew_mean"], "Episodes", "Episode Reward Mean", avg_line = True, save_path=f"{RESULTS_FOLDER}/reward.svg")

create_line_chart(df["time/episodes"], df["rollout/exploration_rate"], "Episodes", "Exploration Rate", save_path=f"{RESULTS_FOLDER}/exploration_rate.svg")

create_line_chart(df["time/episodes"], df["rollout/success_rate"], "Episodes", "Success Rate", avg_line = True, save_path=f"{RESULTS_FOLDER}/success_rate.svg")
