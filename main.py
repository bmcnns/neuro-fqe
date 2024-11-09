from torch import nn, optim

from critic import Critic
import minari

import torch
from torch.utils.data import TensorDataset, DataLoader

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


def polyak_update(target_critic, current_critic, tau=0.005):
    """ Soft updates for the target network """
    for target_param, source_param in zip(target_critic.parameters(), current_critic.parameters()):
        target_param.data.copy_(
            tau * source_param.data + (1.0 - tau) * target_param.data
        )

if __name__ == '__main__':

    offline_data = create_offline_dataset_from_minari(
        minari.load_dataset("HalfCheetah-Expert-v2"),
    )

    num_batches = len(offline_data)

    # Check if GPU is available, otherwise use CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    current_critic = Critic().to(device)
    target_critic = Critic().to(device)

    # Ensure target network starts with the same weights
    target_critic.load_state_dict(current_critic.state_dict())

    optimizer = optim.Adam(current_critic.parameters(), lr=3e-4)

    total_steps = 1_000_000
    step = 0

    while step < total_steps:
        total_loss = 0.0
        num_batches = 0

        for batch in offline_data:
            states, actions, rewards, next_states, next_actions = [
                data.to(device) for data in batch
            ]

            q_values = current_critic(states, actions)

            with torch.no_grad():
                next_q_values = target_critic(next_states, next_actions)
                targets = rewards + 0.99 * next_q_values

            loss = nn.MSELoss()(q_values, targets)
            optimizer.zero_grad()
            loss.backward()
            #torch.nn.utils.clip_grad_norm_(current_critic.parameters(), max_norm=1.0)
            optimizer.step()

            polyak_update(target_critic, current_critic, tau=5e-3)

            total_loss += loss.item()
            num_batches += 1
            step += 1

            if step % 100_000 == 0:
                current_critic.save(step=step)
                torch.save(optimizer.state_dict(), f'models/optimizer_{step}.pt')

            if step >= total_steps:
                break

        if num_batches > 0:
            average_loss = total_loss / num_batches
            print(f"Steps: {step}/{total_steps} | Average Loss: {average_loss:.4f}")

    print("Training complete.")