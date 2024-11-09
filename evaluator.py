from torch import nn, optim

from critic import Critic
import minari

import torch
from torch.utils.data import TensorDataset, DataLoader
import TD3

def create_offline_dataset_from_minari(dataset, batch_size=256, shuffle=True):
    # Collect all data in temporary lists
    states, actions, rewards, next_states, next_actions = [], [], [], [], []

    for episode in dataset:
        num_steps = len(episode.observations) - 1
        for i in range(num_steps - 1):
            states.append(torch.tensor(episode.observations[i], dtype=torch.float32))
            actions.append(torch.tensor(episode.actions[i], dtype=torch.float32))
            rewards.append(episode.rewards[i])
            next_states.append(torch.tensor(episode.observations[i + 1], dtype=torch.float32))
            next_actions.append(torch.tensor(episode.actions[i + 1], dtype=torch.float32))

    # Convert lists to tensors in one go (much faster than doing it inside loops)
    states = torch.stack(states)
    actions = torch.stack(actions)
    rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)  # Shape (N, 1)
    next_states = torch.stack(next_states)
    next_actions = torch.stack(next_actions)

    # Create TensorDataset
    tensor_dataset = TensorDataset(states, actions, rewards, next_states, next_actions)

    # Return DataLoader
    return DataLoader(tensor_dataset, batch_size=batch_size, shuffle=shuffle)

def get_td3_model():
    kwargs = {"state_dim": 17, "action_dim": 6, "max_action": 1.0, "discount": 0.99, "tau": 0.005,
              "policy_noise": 0.2 * 1.0, "noise_clip": 0.5 * 1.0, "policy_freq": 2}

    model = TD3.TD3(**kwargs)
    model.load('TD3_HalfCheetah-v3_9')

    return model

if __name__ == '__main__':
    # Check if GPU is available, otherwise use CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    critic = Critic.load(1_000_000, device=device)
    td3 = get_td3_model()

    offline_data = create_offline_dataset_from_minari(
        minari.load_dataset("HalfCheetah-Expert-v2"),
    )

    num_batches = len(offline_data)

    total_steps = 1_000_000
    step = 0

    while step < total_steps:
        total_loss = 0.0
        normalized_value_error = 0.0
        num_batches = 0

        for batch in offline_data:
            states, actions, rewards, next_states, next_actions = [
                data.to(device) for data in batch
            ]

            q_values = critic(states, actions)
            td3_q_values = td3.critic.Q1(states, actions)

            total_loss = q_values - td3_q_values

            # Calculate residuals (element-wise difference)
            residuals = (q_values - td3_q_values).cpu().numpy()

            # Accumulate the sum of absolute errors for normalization
            batch_loss = abs(residuals).mean()
            total_loss += batch_loss


            #print("FQE")
            #print(q_values)
            #print("TD3")
            #print(td3_q_values)

            num_batches += 1
            step += 1

            if step >= total_steps:
                break

        if num_batches > 0:
            average_loss = total_loss / (num_batches * 1382.35)
            print(f"Steps: {step}/{total_steps} | Average Value Error: {average_loss:.4f}")

    print("Training complete.")