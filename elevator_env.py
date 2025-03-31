import gymnasium as gym
from gymnasium import spaces
import numpy as np
from building import Building

class ElevatorEnv(gym.Env):
    def __init__(self, num_floors=10, num_elevators=3, episode_length=1440, num_passengers=10):
        super(ElevatorEnv, self).__init__()
        
        self.num_floors = num_floors
        self.num_elevators = num_elevators
        self.episode_length = episode_length
        self.current_step = 0
        self.num_passengers = num_passengers    # max number of passengers to generate per floor
        # Initialize building with realistic parameters
        self.building = Building(num_floors, num_elevators)
        
        # Enhanced action space: (elevator_id, destination_floor)
        self.action_space = spaces.MultiDiscrete([
            num_elevators,  # elevator_id
            num_floors       # destination_floor
        ])
        
        # Enhanced observation space with more state information
        self.observation_space = spaces.Dict({
            "elevator_positions": spaces.Box(0, num_floors-1, shape=(num_elevators,), dtype=np.int32),
            "elevator_directions": spaces.Box(-1, 1, shape=(num_elevators,), dtype=np.int32),
            "elevator_loads": spaces.Box(0, 1, shape=(num_elevators,), dtype=np.float32),
            "waiting_counts": spaces.Box(0, 20, shape=(num_floors,), dtype=np.int32),
            "waiting_times": spaces.Box(0, 100, shape=(num_floors,), dtype=np.int32),
            "time_step": spaces.Box(0, episode_length, shape=(1,), dtype=np.int32)
        })

    def reset(self, seed=None, **kwargs):
        super().reset(seed=seed)
        self.current_step = 0
        self.building = Building(self.num_floors, self.num_elevators)
        return self._get_observation(), {}

    def step(self, action):
        # print(f"\n--- Step {self.current_step} ---")
        # print(f"Action taken: Elevator {action[0]} to floor {action[1]}")
        self.current_step += 1
        
        # Parse action
        elevator_id, destination_floor = action
        
        # Validate action
        if not (0 <= elevator_id < self.num_elevators) or not (0 <= destination_floor < self.num_floors):
            reward = -10  # Penalize invalid actions
        else:
            # Execute building step and action
            self.building.step(self.current_step)
            reward = self.building.take_action((elevator_id, destination_floor))
        
        # Force pickup if idle and passengers waiting
        if self.building.elevators[elevator_id].is_idle():
            for floor, passengers in enumerate(self.building.waiting_passengers):
                if passengers:
                    destination = floor  # Override with nearest waiting floor
                    break

        # Get new observation
        obs = self._get_observation()
        # Print summary
        # print(f"Reward: {reward}")
        # print(f"Next state:")
        # print(f"  Elevator positions: {obs['elevator_positions']}")
        # print(f"  Waiting counts: {obs['waiting_counts']}")
        # Check termination conditions
        done = self.current_step >= self.episode_length
        truncated = False
        info = {
            "total_wait_time": sum(
                p.wait_time 
                for floor_passengers in self.building.waiting_passengers.values() 
                for p in floor_passengers
            ),
            "elevator_utilization": sum(
                len(e.passengers)/e.capacity 
                for e in self.building.elevators
            ) / self.num_elevators
        }
        # print(f"Info: {info}")
        return obs, reward, done, truncated, info

    def _get_observation(self):
        state = self.building.state
        return {
            "elevator_positions": np.array(
                [elevator.current_floor for elevator in self.building.elevators],
                dtype=np.int32
            ),
            "elevator_directions": np.array(
                [elevator.direction for elevator in self.building.elevators],
                dtype=np.int32
            ),
            "elevator_loads": np.array(
                [len(elevator.passengers)/elevator.capacity for elevator in self.building.elevators],
                dtype=np.float32
            ),
            "waiting_counts": np.array(
                [len(self.building.waiting_passengers[floor]) for floor in range(self.num_floors)],
                dtype=np.int32
            ),
            "waiting_times": np.array(
                [max([p.wait_time for p in self.building.waiting_passengers[floor]], default=0)
                for floor in range(self.num_floors)],
                dtype=np.int32
            ),
            "time_step": np.array([self.current_step], dtype=np.int32)
        }

    def render(self, mode='human'):
        if mode == 'human':
            print(f"\nStep {self.current_step}")
            print("Elevators:")
            for elevator in self.building.elevators:
                print(f"  {str(elevator)}")
            print("Waiting passengers:")
            for floor, passengers in self.building.waiting_passengers.items():
                if passengers:
                    print(f"  Floor {floor}: {len(passengers)} waiting")
        elif mode == 'rgb_array':
            # Could implement visual rendering here
            pass