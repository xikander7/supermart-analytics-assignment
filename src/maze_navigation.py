"""
Maze navigation module using reinforcement learning.
Optional Task 2: Design a system that trains a model to navigate through a maze.
"""

import numpy as np
import matplotlib.pyplot as plt
import random
from pathlib import Path
import logging
import pickle

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MazeEnvironment:
    """
    Maze environment for reinforcement learning agent training.
    """
    
    def __init__(self, maze_size=(10, 10), start_pos=(0, 1), goal_pos=(8, 8)):
        self.maze_size = maze_size
        self.start_pos = start_pos
        self.goal_pos = goal_pos
        self.current_pos = start_pos
        self.maze = self._create_maze()
        
        # Action space: 0=up, 1=right, 2=down, 3=left
        self.actions = [(0, -1), (1, 0), (0, 1), (-1, 0)]
        self.action_names = ['up', 'right', 'down', 'left']
        
    def _create_maze(self):
        """Create a maze with walls (1) and open spaces (0)."""
        maze = np.zeros(self.maze_size)
        
        # Create walls to make it challenging
        # Outer walls
        maze[0, :] = 1  # Top wall
        maze[-1, :] = 1  # Bottom wall
        maze[:, 0] = 1  # Left wall
        maze[:, -1] = 1  # Right wall
        
        # Internal walls (creating a path-finding challenge)
        maze[2, 1:6] = 1
        maze[4, 3:8] = 1
        maze[6, 1:4] = 1
        maze[6, 6:9] = 1
        maze[1:4, 7] = 1
        maze[7:9, 3] = 1
        
        # Ensure start and goal positions are free
        maze[self.start_pos[1], self.start_pos[0]] = 0
        maze[self.goal_pos[1], self.goal_pos[0]] = 0
        
        return maze
    
    def reset(self):
        """Reset environment to start position."""
        self.current_pos = self.start_pos
        return self._get_state()
    
    def _get_state(self):
        """Get current state representation."""
        return self.current_pos
    
    def step(self, action):
        """Take action and return new state, reward, done flag."""
        # Calculate new position
        dx, dy = self.actions[action]
        new_x = self.current_pos[0] + dx
        new_y = self.current_pos[1] + dy
        
        # Check if new position is valid (within bounds and not a wall)
        if (0 <= new_x < self.maze_size[0] and 
            0 <= new_y < self.maze_size[1] and 
            self.maze[new_y, new_x] == 0):
            
            self.current_pos = (new_x, new_y)
            
            # Calculate reward
            if self.current_pos == self.goal_pos:
                reward = 100  # Large reward for reaching goal
                done = True
            else:
                # Small negative reward for each step (encourages shorter paths)
                reward = -1
                done = False
        else:
            # Penalty for hitting wall or going out of bounds
            reward = -10
            done = False
        
        return self._get_state(), reward, done
    
    def render(self):
        """Visualize the maze and agent position."""
        display = self.maze.copy()
        display[self.current_pos[1], self.current_pos[0]] = 0.5  # Agent position
        display[self.goal_pos[1], self.goal_pos[0]] = 0.8  # Goal position
        
        plt.figure(figsize=(8, 8))
        plt.imshow(display, cmap='RdYlBu', vmin=0, vmax=1)
        plt.title(f'Maze - Agent at {self.current_pos}, Goal at {self.goal_pos}')
        plt.colorbar(label='0=Path, 0.5=Agent, 0.8=Goal, 1=Wall')
        plt.show()
    
    def get_maze_array(self):
        """Return maze array for visualization."""
        return self.maze.copy()

class QLearningAgent:
    """
    Q-Learning agent for maze navigation.
    """
    
    def __init__(self, state_size, action_size, learning_rate=0.1, discount_factor=0.95, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        # Initialize Q-table
        self.q_table = {}
        
        # Training metrics
        self.training_scores = []
        self.training_steps = []
        
    def _get_state_key(self, state):
        """Convert state to string key for Q-table."""
        return str(state)
    
    def get_q_value(self, state, action):
        """Get Q-value for state-action pair."""
        state_key = self._get_state_key(state)
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.action_size)
        return self.q_table[state_key][action]
    
    def update_q_value(self, state, action, new_state, reward, done):
        """Update Q-value using Q-learning update rule."""
        state_key = self._get_state_key(state)
        new_state_key = self._get_state_key(new_state)
        
        # Initialize Q-table entries if they don't exist
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.action_size)
        if new_state_key not in self.q_table:
            self.q_table[new_state_key] = np.zeros(self.action_size)
        
        # Q-learning update rule
        current_q = self.q_table[state_key][action]
        if done:
            target_q = reward
        else:
            target_q = reward + self.discount_factor * np.max(self.q_table[new_state_key])
        
        self.q_table[state_key][action] = current_q + self.learning_rate * (target_q - current_q)
    
    def choose_action(self, state, training=True):
        """Choose action using epsilon-greedy policy."""
        if training and np.random.rand() <= self.epsilon:
            # Explore: choose random action
            return np.random.choice(self.action_size)
        else:
            # Exploit: choose best action
            state_key = self._get_state_key(state)
            if state_key not in self.q_table:
                self.q_table[state_key] = np.zeros(self.action_size)
            return np.argmax(self.q_table[state_key])
    
    def decay_epsilon(self):
        """Decay exploration rate."""
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

class MazeTrainer:
    """
    Trainer for maze navigation reinforcement learning.
    """
    
    def __init__(self, maze_size=(10, 10)):
        self.maze_size = maze_size
        self.env = MazeEnvironment(maze_size=maze_size)
        self.agent = QLearningAgent(
            state_size=maze_size[0] * maze_size[1],
            action_size=4,
            learning_rate=0.1,
            discount_factor=0.95,
            epsilon=1.0,
            epsilon_decay=0.995,
            epsilon_min=0.01
        )
        
        self.training_history = {
            'episodes': [],
            'scores': [],
            'steps': [],
            'epsilon': []
        }
    
    def train(self, episodes=1000, max_steps_per_episode=200):
        """Train the agent to navigate the maze."""
        logger.info(f"Starting maze navigation training for {episodes} episodes...")
        
        for episode in range(episodes):
            state = self.env.reset()
            total_reward = 0
            steps = 0
            
            for step in range(max_steps_per_episode):
                # Choose and take action
                action = self.agent.choose_action(state, training=True)
                new_state, reward, done = self.env.step(action)
                
                # Update Q-table
                self.agent.update_q_value(state, action, new_state, reward, done)
                
                state = new_state
                total_reward += reward
                steps += 1
                
                if done:
                    break
            
            # Decay exploration rate
            self.agent.decay_epsilon()
            
            # Record training metrics
            self.training_history['episodes'].append(episode + 1)
            self.training_history['scores'].append(total_reward)
            self.training_history['steps'].append(steps)
            self.training_history['epsilon'].append(self.agent.epsilon)
            
            # Log progress
            if (episode + 1) % 100 == 0:
                avg_score = np.mean(self.training_history['scores'][-100:])
                avg_steps = np.mean(self.training_history['steps'][-100:])
                logger.info(f"Episode {episode + 1}: Avg Score: {avg_score:.2f}, Avg Steps: {avg_steps:.2f}, Epsilon: {self.agent.epsilon:.3f}")
        
        logger.info("Training completed!")
    
    def test_agent(self, num_tests=10):
        """Test the trained agent's performance."""
        logger.info(f"Testing trained agent over {num_tests} episodes...")
        
        test_results = []
        
        for test in range(num_tests):
            state = self.env.reset()
            total_reward = 0
            steps = 0
            path = [state]
            
            for step in range(200):  # Max steps for testing
                action = self.agent.choose_action(state, training=False)
                new_state, reward, done = self.env.step(action)
                
                state = new_state
                total_reward += reward
                steps += 1
                path.append(state)
                
                if done:
                    break
            
            success = (state == self.env.goal_pos)
            test_results.append({
                'test': test + 1,
                'success': success,
                'steps': steps,
                'reward': total_reward,
                'path': path
            })
            
            logger.info(f"Test {test + 1}: {'SUCCESS' if success else 'FAILED'} - Steps: {steps}, Reward: {total_reward}")
        
        # Calculate success rate
        success_rate = sum(1 for result in test_results if result['success']) / len(test_results)
        avg_steps = np.mean([result['steps'] for result in test_results if result['success']])
        
        logger.info(f"Test Results: Success Rate: {success_rate*100:.1f}%, Average Steps (successful): {avg_steps:.1f}")
        
        return test_results
    
    def visualize_training_progress(self, save_path="report/figures"):
        """Create visualizations of the training progress."""
        save_dir = Path(save_path)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Training scores over time
        ax1.plot(self.training_history['episodes'], self.training_history['scores'])
        ax1.set_title('Training Scores Over Time')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Score')
        ax1.grid(True)
        
        # Steps per episode
        ax2.plot(self.training_history['episodes'], self.training_history['steps'])
        ax2.set_title('Steps Per Episode')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Steps')
        ax2.grid(True)
        
        # Epsilon decay
        ax3.plot(self.training_history['episodes'], self.training_history['epsilon'])
        ax3.set_title('Epsilon (Exploration Rate) Decay')
        ax3.set_xlabel('Episode')
        ax3.set_ylabel('Epsilon')
        ax3.grid(True)
        
        # Moving average of scores (last 100 episodes)
        window_size = min(100, len(self.training_history['scores']))
        if window_size > 1:
            moving_avg = np.convolve(self.training_history['scores'], 
                                   np.ones(window_size)/window_size, mode='valid')
            ax4.plot(range(window_size, len(self.training_history['scores']) + 1), moving_avg)
            ax4.set_title(f'Moving Average Score (Window: {window_size})')
            ax4.set_xlabel('Episode')
            ax4.set_ylabel('Average Score')
            ax4.grid(True)
        
        plt.tight_layout()
        plt.savefig(save_dir / 'maze_training_progress.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Maze visualization
        self.visualize_maze_and_solution(save_dir)
        
        logger.info(f"Training visualizations saved to {save_dir}")
    
    def visualize_maze_and_solution(self, save_dir):
        """Visualize the maze and a solution path."""
        # Find a solution path
        state = self.env.reset()
        path = [state]
        
        for step in range(200):
            action = self.agent.choose_action(state, training=False)
            new_state, reward, done = self.env.step(action)
            state = new_state
            path.append(state)
            
            if done:
                break
        
        # Create visualization
        maze_display = self.env.get_maze_array()
        
        # Mark the path
        for i, pos in enumerate(path):
            if pos != self.env.goal_pos:
                maze_display[pos[1], pos[0]] = 0.3
        
        # Mark start and goal
        maze_display[self.env.start_pos[1], self.env.start_pos[0]] = 0.5
        maze_display[self.env.goal_pos[1], self.env.goal_pos[0]] = 0.8
        
        plt.figure(figsize=(10, 10))
        plt.imshow(maze_display, cmap='RdYlBu', vmin=0, vmax=1)
        plt.title('Maze with Solution Path')
        plt.colorbar(label='0=Path, 0.3=Solution, 0.5=Start, 0.8=Goal, 1=Wall')
        
        # Add grid
        for i in range(self.maze_size[0] + 1):
            plt.axhline(y=i-0.5, color='gray', linewidth=0.5)
        for i in range(self.maze_size[1] + 1):
            plt.axvline(x=i-0.5, color='gray', linewidth=0.5)
        
        plt.savefig(save_dir / 'maze_solution.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_model(self, filepath="models/maze_agent.pkl"):
        """Save the trained agent."""
        save_path = Path(filepath)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        model_data = {
            'q_table': self.agent.q_table,
            'training_history': self.training_history,
            'maze_size': self.maze_size,
            'agent_params': {
                'learning_rate': self.agent.learning_rate,
                'discount_factor': self.agent.discount_factor,
                'epsilon': self.agent.epsilon
            }
        }
        
        with open(save_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Maze navigation model saved to {save_path}")

def main():
    """Main function to run maze navigation training."""
    # Create trainer
    trainer = MazeTrainer(maze_size=(10, 10))
    
    # Train agent
    trainer.train(episodes=1000, max_steps_per_episode=200)
    
    # Test agent
    test_results = trainer.test_agent(num_tests=10)
    
    # Create visualizations
    trainer.visualize_training_progress()
    
    # Save model
    trainer.save_model()
    
    # Print final results
    success_count = sum(1 for result in test_results if result['success'])
    print(f"\n" + "="*50)
    print("MAZE NAVIGATION RESULTS")
    print("="*50)
    print(f"Training Episodes: 1000")
    print(f"Test Success Rate: {success_count}/10 ({success_count*10}%)")
    
    if success_count > 0:
        successful_tests = [r for r in test_results if r['success']]
        avg_steps = np.mean([r['steps'] for r in successful_tests])
        print(f"Average Steps (Successful): {avg_steps:.1f}")
    
    print("="*50)

if __name__ == "__main__":
    main()