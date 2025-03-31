import tkinter as tk
from tkinter import ttk
from elevator_env import ElevatorEnv
from stable_baselines3 import PPO
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np

class ElevatorGUI:
    def __init__(self, master, env, model):
        self.master = master
        self.env = env
        self.model = model
        self.master.title("Elevator Dispatch RL Simulation")
        self.master.geometry("1000x700")
        
        # Configure grid layout
        self.master.grid_columnconfigure(0, weight=1)
        self.master.grid_rowconfigure(0, weight=1)
        
        # Create main frames
        self.simulation_frame = ttk.Frame(self.master, padding="10")
        self.control_frame = ttk.Frame(self.master, padding="10")
        self.stats_frame = ttk.Frame(self.master, padding="10")
        
        self.simulation_frame.grid(row=0, column=0, sticky="nsew")
        self.control_frame.grid(row=1, column=0, sticky="ew")
        self.stats_frame.grid(row=0, column=1, rowspan=2, sticky="nsew")
        
        # Simulation canvas
        self.canvas = tk.Canvas(self.simulation_frame, width=600, height=500, bg="white")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # Statistics display
        self.setup_stats_panel()
        
        # Controls
        self.setup_controls()
        
        # Initialize
        self.obs, _ = self.env.reset()
        self.step_count = 0
        self.rewards = []
        self.wait_times = []
        self.draw_building()
        
    def setup_stats_panel(self):
        """Set up the statistics and metrics display"""
        # Create main container for stats
        stats_container = ttk.Frame(self.stats_frame)
        stats_container.pack(fill=tk.BOTH, expand=True)
        
        # Create figure with subplots
        self.fig = plt.figure(figsize=(8, 5), dpi=100)
        gs = self.fig.add_gridspec(2, 2)
        
        # Create subplots
        self.ax_reward = self.fig.add_subplot(gs[0, 0])
        self.ax_reward.set_title("Reward History")
        
        self.ax_wait = self.fig.add_subplot(gs[0, 1])
        self.ax_wait.set_title("Wait Time Distribution")
        
        self.ax_util = self.fig.add_subplot(gs[1, 0])
        self.ax_util.set_title("Elevator Utilization")
        
        self.ax_flow = self.fig.add_subplot(gs[1, 1])
        self.ax_flow.set_title("Passenger Flow")
        
        # Embed matplotlib figure
        self.canvas_stats = FigureCanvasTkAgg(self.fig, master=stats_container)
        self.canvas_stats.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        # Add text stats below the plots
        self.stats_text = tk.Text(stats_container, height=8, width=80, font=("Consolas", 10))
        self.stats_text.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)
        
        # Additional stats tracking
        self.delivered_history = []
        self.utilization_history = []
        self.wait_time_distribution = []
        
    def setup_controls(self):
        """Set up control buttons and speed control"""
        control_panel = ttk.Frame(self.control_frame)
        control_panel.pack()
        
        self.step_button = ttk.Button(control_panel, text="Step", command=self.step)
        self.step_button.grid(row=0, column=0, padx=5)
        
        self.reset_button = ttk.Button(control_panel, text="Reset", command=self.reset)
        self.reset_button.grid(row=0, column=1, padx=5)
        
        self.auto_button = ttk.Button(control_panel, text="Auto Run", command=self.toggle_auto)
        self.auto_button.grid(row=0, column=2, padx=5)
        
        self.speed_scale = ttk.Scale(control_panel, from_=100, to=1000, 
                                    command=self.set_speed, orient=tk.HORIZONTAL)
        self.speed_scale.set(500)
        self.speed_scale.grid(row=0, column=3, padx=5)
        ttk.Label(control_panel, text="Speed:").grid(row=0, column=4, padx=5)
        
        self.auto_running = False
        self.speed = 500
        
    def draw_building(self):
        """Draw the building with elevators and passengers"""
        self.canvas.delete("all")
        num_floors = self.env.building.num_floors
        num_elevators = len(self.env.building.elevators)
        floor_height = 450 / num_floors
        elevator_width = 30
        
        # Draw building outline
        self.canvas.create_rectangle(50, 20, 550, 470, outline="black")
        
        # Draw floors and labels
        for i in range(num_floors):
            y = 450 - i * floor_height
            self.canvas.create_line(50, y, 550, y, fill="gray")
            self.canvas.create_text(30, y - floor_height/2, text=f"F{i}", anchor=tk.E, font=("Arial", 12))
            
            # Draw waiting passengers
            wait_count = len(self.env.building.waiting_passengers[i])
            if wait_count > 0:
                self.canvas.create_text(570, y - floor_height/2, text=f"{wait_count} waiting", anchor=tk.W, font=("Arial", 12))
                
        # Draw elevators
        for i, elevator in enumerate(self.env.building.elevators):
            x = 100 + i * (400 / max(1, num_elevators-1))
            y = 450 - elevator.current_floor * floor_height
            
            # Elevator car
            fill_color = "blue" if elevator.direction == 1 else \
                        "red" if elevator.direction == -1 else "gray"
            self.canvas.create_rectangle(
                x - elevator_width/2, y - floor_height,
                x + elevator_width/2, y,
                fill=fill_color, outline="black"
            )
            
            # Passenger count
            load_percent = len(elevator.passengers) / elevator.capacity
            self.canvas.create_text(
                x, y - floor_height/2,
                text=f"{len(elevator.passengers)}/{elevator.capacity}",
                fill="white" if load_percent > 0.5 else "black",
                font=("Arial", 12)
            )
            
            # Destination indicator
            if elevator.destination is not None:
                dest_y = 450 - elevator.destination * floor_height
                self.canvas.create_line(x, y, x, dest_y, arrow=tk.LAST, dash=(2,2))
            
            # Passenger destinations (NEW)
            if elevator.passengers:
                dest_counts = {}
                for p in elevator.passengers:
                    dest_counts[p.destination] = dest_counts.get(p.destination, 0) + 1
                
                dest_text = ",".join(f"{k}({v})" for k,v in dest_counts.items())
                self.canvas.create_text(
                    x, y - floor_height + 10,
                    text=f"→{dest_text}",
                    font=("Arial", 8),
                    fill="darkgreen"
                )

        # Update stats display
        self.update_stats()
        
    def update_stats(self):
        """Update all statistics displays"""
        # Clear all axes
        for ax in [self.ax_reward, self.ax_wait, self.ax_util, self.ax_flow]:
            ax.clear()
        
        # 1. Reward History Plot
        if len(self.rewards) > 1:
            self.ax_reward.plot(self.rewards, label='Reward')
            self.ax_reward.set_ylabel("Reward")
            self.ax_reward.grid(True)
        
        # 2. Wait Time Distribution
        if self.wait_times:
            self.ax_wait.hist(self.wait_times, bins=20, alpha=0.7)
            self.ax_wait.set_xlabel("Wait Time (steps)")
            self.ax_wait.set_ylabel("Count")
        
        # 3. Elevator Utilization
        util = [len(e.passengers)/e.capacity for e in self.env.building.elevators]
        self.utilization_history.append(np.mean(util))
        if len(self.utilization_history) > 1:
            self.ax_util.plot(self.utilization_history)
            self.ax_util.set_ylim(0, 1)
            self.ax_util.set_ylabel("Utilization %")
        
        # 4. Passenger Flow
        delivered = sum(1 for e in self.env.building.elevators 
                    if any(p.destination == e.current_floor for p in e.passengers))
        self.delivered_history.append(delivered)
        waiting = sum(len(p) for p in self.env.building.waiting_passengers.values())
        
        if len(self.delivered_history) > 1:
            self.ax_flow.plot(self.delivered_history, label='Delivered')
            self.ax_flow.plot([waiting]*len(self.delivered_history), 'r--', label='Waiting')
            self.ax_flow.legend()
        
        
        # Calculate metrics
        total_waiting = sum(len(p) for p in self.env.building.waiting_passengers.values())
        avg_wait_time = np.mean(self.wait_times) if self.wait_times else 0
        current_reward = self.rewards[-1] if self.rewards else 0
        
        # Update text stats
        self.stats_text.delete(1.0, tk.END)
        self.stats_text.insert(tk.END, 
            f"Step: {self.step_count}\n"
            f"Current Reward: {current_reward:.2f}\n"
            f"Total Waiting: {total_waiting}\n"
            f"Avg Wait Time: {avg_wait_time:.1f} steps\n"
        )
        self.stats_text.insert(tk.END,
            f"Elevators:\n"
            f"{'ID':<3}{'Pos':<5}{'Load':<7}{'Dest':<5}{'Passengers'}\n"
        )
        for i, elevator in enumerate(self.env.building.elevators):
            passenger_dests = [p.destination for p in elevator.passengers]
            dest_counts = {d:passenger_dests.count(d) for d in set(passenger_dests)}
            dest_str = ",".join(f"{k}({v})" for k,v in dest_counts.items()) if dest_counts else "Empty"
            
            self.stats_text.insert(tk.END,
                f"  E{i:<3}: Floor {elevator.current_floor:<5} | "
                f"{len(elevator.passengers)}/{elevator.capacity:<7} | "
                f"{'▲' if elevator.direction == 1 else '▼' if elevator.direction == -1 else '■'} | "
                f"{elevator.destination if elevator.destination else '-':<5}"
                f"{dest_str}\n"
            )
        
        self.canvas_stats.draw()
    
    def get_passenger_flow(self):
        """Track origin-destination pairs"""
        flow = np.zeros((self.env.num_floors, self.env.num_floors))
        for floor in range(self.env.num_floors):
            for p in self.env.building.waiting_passengers[floor]:
                flow[floor, p.destination] += 1
        for elevator in self.env.building.elevators:
            for p in elevator.passengers:
                flow[p.start_floor, p.destination] += 1
        return flow

    def update_flow_heatmap(self):
        """Update passenger flow visualization"""
        flow = self.get_passenger_flow()
        self.ax_flow.clear()
        self.ax_flow.imshow(flow, cmap='YlOrRd')
        self.ax_flow.set_title("Passenger Flow Heatmap")
        self.ax_flow.set_xlabel("Destination Floor")
        self.ax_flow.set_ylabel("Origin Floor")
        
    def step(self):
        """Perform one simulation step"""
        action, _ = self.model.predict(self.obs, deterministic=True)
        self.obs, reward, done, _, info = self.env.step(action)
        
        self.step_count += 1
        self.rewards.append(reward)
        self.wait_times.append(info.get("total_wait_time", 0))
        
        self.draw_building()
        
        if done:
            self.obs, _ = self.env.reset()
            
        if self.auto_running:
            self.master.after(self.speed, self.step)
            
    def reset(self):
        """Reset the simulation"""
        self.obs, _ = self.env.reset()
        self.step_count = 0
        self.rewards = []
        self.wait_times = []
        self.delivered_history = []
        self.utilization_history = []
        self.wait_time_distribution = []
        self.draw_building()
        
    def toggle_auto(self):
        """Toggle automatic simulation"""
        self.auto_running = not self.auto_running
        self.auto_button.config(text="Stop" if self.auto_running else "Auto Run")
        if self.auto_running:
            self.step()
            
    def set_speed(self, value):
        """Set simulation speed"""
        self.speed = int(float(value))

def run_gui(num_floors, num_elevators, model_path="elevator_ppo_model"):
    """Run the GUI with specified parameters"""
    env = ElevatorEnv(num_floors, num_elevators)
    try:
        model = PPO.load(model_path+f"_{num_floors}_{num_elevators}")
    except:
        print(f"Could not load model from {model_path}. Using random actions.")
        model = None
    
    root = tk.Tk()
    gui = ElevatorGUI(root, env, model)
    root.mainloop()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--floors", type=int, default=10, help="Number of floors")
    parser.add_argument("--elevators", type=int, default=3, help="Number of elevators")
    parser.add_argument("--model", type=str, default="elevator_ppo_model", help="Model path")
    args = parser.parse_args()
    
    run_gui(args.floors, args.elevators, args.model)