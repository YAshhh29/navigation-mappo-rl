# Obstacle RL: David Silver Course Implementation

![Multi Agent Navigation Demo](images/marl1.gif)

## Project Description
The core RL task involves multiple agents (represented as colored circles and arrow-heads) navigating from their starting positions to their respective end goals while avoiding collisions with other agents and static obstacles. 

This project was built to demonstrate a deep, conceptual understanding of the foundational Reinforcement Learning algorithms taught in **David Silver's 2015 DeepMind RL Course**. By stripping away advanced modern techniques, the repository directly applies the core principles from his lectures to a continuous-space multi-agent environment.

*Note: The environment and original base code for this project are highly inspired by and adapted from the excellent "Neural Breakdown with AVB" YouTube channel and their `navigation-mappo-rl` repository.*

### Core Course Concepts Implemented:
1. **Value Function Approximation (Lecture 6):** An implementation of a Decentralized Critic Network using a Multi-Layer Perceptron (MLP) to estimate state-values for advantage bootstrapping.
2. **Policy Gradients (Lecture 7):**
    - **REINFORCE (Monte Carlo Policy Gradient):** A pure Monte Carlo approach that bootstraps using the full episode returns to update the actor network.
    - **Advantage Actor-Critic (A2C):** Uses Temporal Difference (TD) learning, bootstrapping the Value Function via a Critic Network to estimate advantages and train the Actor policy with less variance.

# Original Video Tutorial Reference

> **📺 Watch the original Environment Video**
> **[Training RL Agents to Navigate as a Team](https://youtu.be/Ji5VTbH7i08)**

## Getting Started

### Prerequisites

-   **Python 3.10+** (required)
-   **`uv`** (recommended) or `pip` for package management
-   To render movies and videos, install ffmpeg

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/avbiswas/navigation-mappo-rl.git
    ```

2.  **Install dependencies:**
    ```bash
    # Using uv
    uv sync
    ```

3.  **Train a new model:**
    You can train using either A2C (Actor-Critic) or REINFORCE (Monte Carlo Policy Gradient):
    ```bash
    uv run train_a2c.py model_id configs/config.yaml
    uv run train_reinforce.py model_id configs/config.yaml
    ```

    For example:
    ```bash
    uv run train_a2c.py model_1 configs/basic_env.yaml
    ```

    This will create a new model inside `models/model_1`. The latest model and the best all-time models are both saved.

4. **Run inference**
    ```bash
    uv run inference.py models/model_1
    ```

    By default, this will test on the environment config file where the model was originally trained. You can also test on a new config though.

    ```bash
    uv run inference.py models/model_1 configs/bottleneck.yaml
    ```

## Included Environments

A variety of pre-configured environments are provided for your experiments in the `configs/` directory.

1. Basic

   ![Basic Environment](images/basic.png)

2. Circle

   ![Circle Environment](images/circle.png)

3. Moving Environment

   ![Moving Environment](images/moving.png)

4. Hallway

   ![Hallway](images/hallway.png)

5. Bottleneck

   ![Bottleneck](images/bottleneck.png)

6. Four Crossing

   ![Four Crossing](images/fourcross.png)

## Creating Custom Environments

You can define your own environments by creating a new `.yaml` file in the `configs/` directory. The configuration schema is defined in `nav/config_models.py`.

Key components of a configuration file:
- **Boundary**: Defines the polygon vertices for the playable area.
- **Agents**: Specifies start/goal zones (rectangles), physical properties (radius, max speed), and sensor settings (FOV, range).
- **Obstacles**: Defines static or moving obstacles (rectangles or circles).

For reference, check `configs/basic_env.yaml` for a simple setup or `configs/moving_env.yaml` for dynamic obstacles.

## Training Configuration

The training configuration and hyperparameters are defined in `train_a2c.py` and `train_reinforce.py`.

Key settings include:
- **History Length**: `history_length = 4` (Number of past frames stacked).
- **Batch Size**: `batch_size=128` (Number of samples per update).
- **Learning Rate**: `learning_rate=5e-4`.
- **Inference Interval**: `inference_interval=5` (How often to run evaluation episodes).
- **Network Architecture**: The policy network uses an `ObservationEncoder` which processes LIDAR data and agent states, outputting a feature vector of size 384.

To modify these, edit the initialization in the respective training scripts.

## Rendering

Check out `nav/live_renderer.py` to see useful rendering settings.

## Environment Details

The environment specifications are defined in `nav/environment.py`.

### Observation Space (`Box`)
Each agent receives a composite observation consisting of:
1.  **State Vector**:
    -   Progress towards goal (normalized 0-1).
    -   Cosine of the angle between the agent's heading and the goal.
    -   Current speed ratio (current_speed / max_speed).
    -   Distance to goal.
    -   Goal vector (x, y).
2.  **LIDAR Readings**:
    -   A set of raycasts (default 60 rays).
    -   Each ray returns 3 channels: [Distance to Obstacle, Distance to Boundary, Distance to Agent].
    -   Stacked with `history_length` (default 4) frames to provide temporal context.

### Action Space (`Box(2,)`)
The agent controls its movement via a continuous 2D vector:
-   `[vx, vy]`: Velocity components in the X and Y directions.
-   Values are clipped to the range `[-1, 1]` and scaled by the agent's `max_speed`.
-   The action is applied in the Local Coordinate Space of the agent, where the origin is at the agent's center, and the Y-axis is the agent's goal vector.

### Reward Structure
The reward function incentivizes reaching the goal while avoiding collisions:
-   **Goal Reached**: `+10`
-   **Collision** (Obstacle, Boundary, or Agent): `-10`
-   **Progress Reward**: Scaled by speed and alignment with the goal direction (encourages moving efficiently towards the target).
-   **Time Penalty (Stay Alive)**: `-0.05` per step (encourages reaching the goal quickly).


## Network Architecture

The model architecture implements core concepts from David Silver's RL course.

### Shared Observation Encoder (`ObservationEncoder`)
Extracts features from the raw environment state (shared across both Actor and Critic in A2C).
- **Input**: Takes the agent's observation, which includes stacked history frames.
- **Structure**: 
    - **1D CNN**: Processes the LIDAR data.
    - **Agent Network (MLP)**: Processes scalar agent states (velocity, distance to goal, etc.).

### Decentralized Critic (`DecentralizedCriticNetwork` in A2C)
Reflects the Value Function Approximation theory (Lecture 6). The critic evaluates the value of a state for an individual agent.
- **Input**: State feature vector from `ObservationEncoder`.
- **Structure**: Multi-Layer Perceptron.
- **Output**: Scalar state-value estimate.

### Decentralized Actor (`DecentralizedActorNetwork`)
Implements the continuous-action Policy Network (Lecture 7).
- **Input**: Feature vector from the `ObservationEncoder`.
- **Output**: Probability distribution (Diagonal Gaussian) over the action space, from which the agent samples movement commands.
