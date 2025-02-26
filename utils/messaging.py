from collections import deque


class Message():
    def __init__(self, time, sender, model, trust, gradient=None) -> None:
        self.time = time
        self.sender = sender
        self.model = model
        self.trust = trust

    def get_time(self):
        return self.time

    def get_sender(self):
        return self.sender

    def get_model(self):
        return self.model

# NOT USED at the moment


class MsgBuffer():
    def __init__(self) -> None:
        self.queue = deque()

    def append(self, msg: Message):
        self.queue.append(msg)

    def get_all_models(self):
        models = list()
        for msg in self.queue:
            models.append(msg.get_model())
        return models
