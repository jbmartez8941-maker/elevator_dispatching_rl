class Elevator:
    def __init__(self, id, num_floors, building=None, capacity=10, speed=1):
        self.id = id
        self.current_floor = 0
        self.destination = None
        self.direction = 0  # 0: idle, 1: up, -1: down
        self.passengers = []
        self.num_floors = num_floors
        self.capacity = capacity
        self.speed = speed  # floors per step
        self.door_open = False
        self.door_timer = 0
        self.loading_time = 2  # steps needed for boarding/alighting
        self.building = building

    def move(self):
        # Movement logic
        if self.destination is not None:
            if self.current_floor < self.destination:
                self.direction = 1
                self.current_floor += self.speed
            elif self.current_floor > self.destination:
                self.direction = -1
                self.current_floor -= self.speed
            
            # Auto-pickup if enabled
            if hasattr(self, 'building'):  # Safety check
                self._try_pickup_passengers()
            
            # Snap to destination if overshot
            if (self.direction == 1 and self.current_floor >= self.destination) or \
            (self.direction == -1 and self.current_floor <= self.destination):
                self.current_floor = self.destination
                self._handle_arrival()

    def _try_pickup_passengers(self):
        if (not self.is_full() and 
            self.current_floor in self.building.waiting_passengers and
            self.building.waiting_passengers[self.current_floor]):
            
            # Sort passengers by longest waiting first
            passengers = sorted(self.building.waiting_passengers[self.current_floor],
                            key=lambda p: p.wait_time, reverse=True)
            
            # Take as many as capacity allows
            for p in passengers[:self.get_available_space()]:
                self.add_passenger(p)
                self.building.waiting_passengers[self.current_floor].remove(p)
                # print(f"Picked up passenger waiting {p.wait_time} steps")

    def _handle_arrival(self):
        """Helper method for destination arrival logic"""
        # print(f"Elevator {self.id} reached destination {self.destination}")
        departing = self.remove_passengers()
        # if departing:
            # print(f"Dropped off {len(departing)} passengers")
        self.destination = None
        self.direction = 0
        self.door_open = True
        self.door_timer = 0

    def add_passenger(self, passenger):
        if len(self.passengers) < self.capacity and self.door_open:
            self.passengers.append(passenger)
            return True
        return False

    def remove_passengers(self):
        if not self.door_open:
            return []
        
        departing = [p for p in self.passengers if p.destination == self.current_floor]
        self.passengers = [p for p in self.passengers if p.destination != self.current_floor]
        return departing

    def get_available_space(self):
        return self.capacity - len(self.passengers)

    def is_full(self):
        return len(self.passengers) >= self.capacity

    def is_idle(self):
        return self.destination is None and not self.door_open

    def get_next_destinations(self):
        """Returns sorted list of floors the elevator needs to visit"""
        floors = {p.destination for p in self.passengers}
        if self.destination:
            floors.add(self.destination)
        return sorted(floors, reverse=(self.direction == 1)) if self.direction else sorted(floors)

    def __str__(self):
        status = []
        if self.door_open:
            status.append("doors open")
        else:
            status.append("doors closed")
        
        if self.direction == 1:
            status.append("moving up")
        elif self.direction == -1:
            status.append("moving down")
        
        return (f"Elevator {self.id} at floor {self.current_floor} "
                f"({', '.join(status)}), {len(self.passengers)}/{self.capacity} passengers")