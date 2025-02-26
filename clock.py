from queue import PriorityQueue


class ClockEvent:
    valid_event_type = {"TRAIN"}

    def __init__(self, time, node_id, event_type, event_type_counter):
        self._node_id = node_id
        self._time = time
        if event_type not in self.valid_event_type:
            raise ValueError("results: status must be one of %r." %
                             self.valid_event_type)
        self._event_type = event_type
        self._event_type_counter = event_type_counter

    @property
    def node_id(self):
        return self._node_id

    @node_id.setter
    def node_id(self, node_id):
        self._node_id = node_id

    @property
    def time(self):
        return self._time

    @time.setter
    def time(self, time):
        self._time = time

    @property
    def event_type(self):
        return self._event_type

    @event_type.setter
    def event_type(self, event_type):
        self._event_type = event_type

    @property
    def event_type_counter(self):
        return self._event_type_counter

    @event_type_counter.setter
    def event_type_counter(self, event_type_counter):
        self._event_type_counter = event_type_counter

    def __lt__(self, other):
        # definition of the < operator:  sort first by time of event, then by node_id
        return (self.time < other.time) or (self.time == other.time and self.node_id < other.node_id)

    def __iter__(self):
        ''' Returns the Iterator object '''
        return iter([self.time, self.node_id, self.event_type, self.event_type_counter])


class Clock:
    def __init__(self, node_list, max_comm_rounds=10, start_round=0):
        self.clock = PriorityQueue()
        for node in node_list:
            # to be replaced with clock.put((random.uniform(0,1), node)) for desynch start
            #self.clock.put(ClockEvent(0, node, "TRAIN", 1))
            # start at different times if checkpoit exists
            #self.clock.put(ClockEvent(0, node, "TRAIN", start_round))
            self.clock.put(ClockEvent(start_round, node, "TRAIN", 1))
        self.max_comm_rounds = max_comm_rounds

    def empty(self):
        return self.clock.empty()

    def get_next(self):
        return self.clock.get()

    def time_of_next(self):
        if self.empty():
            return -1
        else:
            return self.clock.queue[0].time

    def push_next(self, time: float, node: int, event_type: str, event_counter: int) -> None:
        # add the event to the clock queue
        self.clock.put(ClockEvent(time, node, event_type, event_counter))
        return

    def training_finished(self, event_counter: int, still_gaining: bool = True) -> bool:
        """Check if the training for the current node has finished

        Very simple implementation: the training has finished if either we have reach the set number of epochs
        or if the loss gain started decreasing (this means that the loss at the previous step was smaller than
        at the current one)
        """

        if event_counter > self.max_comm_rounds or not still_gaining:
            return True
        return False
