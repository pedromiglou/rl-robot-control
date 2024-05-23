#!/usr/bin/env python3

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import pandas as pd


RESULTS_FOLDER = "./results/larcc_joints_continuous"
# RESULTS_FOLDER = "./results/larcc_joints_continuous/position_only"
# RESULTS_FOLDER = "./results/larcc_joints_continuous/orientation_only"

plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.size"] = 15


def create_line_chart(x, y, xlabel, ylabel, legend=None, save_path=None):
    # Create a figure and axis
    plt.figure(figsize = (10,5))
    
    for line in y:
        # Plot the line chart
        plt.plot(x, line)
    
    # Set the x-axis label
    plt.xlabel(xlabel)
    
    # Set the y-axis label
    plt.ylabel(ylabel)

    if legend:
        plt.legend(legend)
    
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

eval_df = df.iloc[:-1:2]
train_df = df.iloc[1::2]

# Call the function with the DataFrame
create_line_chart(train_df["time/episodes"], (eval_df["train/actor_loss"],), "Episodes", "Actor Loss", save_path=f"{RESULTS_FOLDER}/actor_loss.svg")

create_line_chart(train_df["time/episodes"], (eval_df["train/critic_loss"],), "Episodes", "Critic Loss", save_path=f"{RESULTS_FOLDER}/critic_loss.svg")

create_line_chart(train_df["time/episodes"], (train_df["rollout/ep_rew_mean"], eval_df["eval/mean_reward"]), "Episodes", "Episode Reward Mean (Max 50)", legend=["training mean reward", "validation mean reward"], save_path=f"{RESULTS_FOLDER}/reward.svg")

create_line_chart(train_df["time/episodes"], (eval_df["train/ent_coef"],), "Episodes", "Entropy Coefficient", save_path=f"{RESULTS_FOLDER}/entropy_coefficient.svg")

create_line_chart(train_df["time/episodes"], (train_df["rollout/success_rate"], eval_df["eval/success_rate"]), "Episodes", "Success Rate", legend=["training success rate", "validation success rate"], save_path=f"{RESULTS_FOLDER}/success_rate.svg")

try:
    f = open(f"{RESULTS_FOLDER}/joint_values.txt", "r")
    lines = [[float(n) for n in line.split(",")] for line in f.readlines()[:100]]
    f.close()

    lines = [[lines[x][y] for x in range(len(lines))] for y in range(len(lines[0]))]

    create_line_chart([t/25 for t in range(len(lines[0]))], lines, "Time", "Joint Velocities",
                    legend=["shoulder_pan_joint", "shoulder_lift_joint", "upperarm_roll_joint",
                            "elbow_flex_joint", "forearm_roll_joint", "wrist_flex_joint", "wrist_roll_joint"],
                            save_path=f"{RESULTS_FOLDER}/joint_values.svg")

except:
    pass
