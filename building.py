import random
import numpy as np
from elevator import Elevator

class Passenger:
    def __init__(self, start_floor, destination_floor, spawn_time):
        self.start_floor = start_floor
        self.destination = destination_floor
        self.spawn_time = spawn_time
        self.wait_time = 0
        
    def increment_wait(self):
        self.wait_time += 1

class Building:
    def __init__(self, num_floors, num_elevators):
        self.num_floors = num_floors
        self.elevators = [Elevator(i, num_floors, building=self) for i in range(num_elevators)]
        self.waiting_passengers = {floor: [] for floor in range(num_floors)}
        self.state = self._get_state()
        self.next_state = None

    def step(self, time_step=None):  # Make time_step optional
        # Update passenger wait times
        for floor, passengers in self.waiting_passengers.items():
            for p in passengers:
                p.increment_wait()
        
        # Move elevators
        for elevator in self.elevators:
            elevator.move()
            elevator.remove_passengers()

        # Generate new passengers (pass time_step if needed)
        if time_step is not None:
            self._generate_passengers(time_step)
        else:
            self._generate_passengers()

        # Update state
        self.state = self.next_state if self.next_state else self._get_state()
        self.next_state = self._get_state()

    def take_action(self, action):
        elevator_id, destination_floor = action
        
        # Validate action
        if not (0 <= elevator_id < len(self.elevators)):
            return -10  # Large penalty for invalid elevator ID
            
        elevator = self.elevators[elevator_id]
        
        if not (0 <= destination_floor < self.num_floors):
            return -10  # Large penalty for invalid floor
        
        # Only set destination if elevator isn't full
        if len(elevator.passengers) < elevator.capacity:
            elevator.destination = destination_floor
        
        return self._calculate_reward()

    def _generate_passengers(self, time_step=0):
        # Simulate peak hours (morning rush - going up, evening - going down)
        is_morning_peak = (time_step % 1440) in range(480, 600)  # 8-10AM
        is_evening_peak = (time_step % 1440) in range(1020, 1140)  # 5-7PM
        
        for floor in range(self.num_floors):
            # Higher probability for lobby during morning, top floors during evening
            prob = 0.05  # base probability
            if floor == 0 and is_morning_peak:
                prob = 0.3
            elif floor == self.num_floors-1 and is_evening_peak:
                prob = 0.3
                
            if random.random() < prob:
                if is_morning_peak and floor == 0:
                    destination = random.randint(floor+1, self.num_floors-1)
                elif is_evening_peak and floor == self.num_floors-1:
                    destination = 0
                else:
                    destination = random.choice([f for f in range(self.num_floors) if f != floor])
                self.waiting_passengers[floor].append(Passenger(floor, destination, time_step))

    def _get_state(self):
        return {
            'elevator_positions': [elevator.current_floor for elevator in self.elevators],
            'elevator_directions': [elevator.direction for elevator in self.elevators],  # 1=up, -1=down, 0=idle
            'elevator_loads': [len(elevator.passengers)/elevator.capacity for elevator in self.elevators],
            'waiting_passengers': {floor: len(passengers) for floor, passengers in self.waiting_passengers.items()},
            'waiting_times': {floor: max([p.wait_time for p in passengers], default=0) 
                            for floor, passengers in self.waiting_passengers.items()}
        }

    def get_all_waiting(self):
        """Returns list of all waiting passengers"""
        return [p for passengers in self.waiting_passengers.values() for p in passengers]
        
    def _calculate_reward(self):
        # Delivery bonus (most important)
        delivered = sum(
            1 for e in self.elevators
            if any(p.destination == e.current_floor for p in e.passengers)
        ) * 5.0  # Increased bonus
        
        # Wait penalty (capped per passenger)
        wait_penalty = sum(min(p.wait_time, 20) for p in self.get_all_waiting()) * 0.1
        
        # Movement penalty (only when empty)
        move_penalty = sum(
            abs(e.direction) * (0.1 if e.passengers else 0.05)
            for e in self.elevators
        )
        
        reward = delivered - wait_penalty - move_penalty
        return np.clip(reward/10, -1, 1)  # Scaled and bounded

    def __str__(self):
        return f"Building with {self.num_floors} floors and {len(self.elevators)} elevators"