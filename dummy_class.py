import time

class Dummy:
    def __init__(self):
        print("Dummy class initialized")
        self.name = "Dummy"
        self.value = 42

    def train(self, stop_event):
        print("Training started")
        while not stop_event.is_set():
            print("Training...")
            time.sleep(1)
        print("Training stopped")