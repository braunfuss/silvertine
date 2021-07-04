
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from pyrocko.guts import Object, Timestamp, String, Float


class Watcher:
    DIRECTORY_TO_WATCH = String.T()

    def __init__(self):
        self.observer = Observer()

    def run(self):
        event_handler = Handler()
        self.observer.schedule(event_handler, self.DIRECTORY_TO_WATCH, recursive=False)
        self.observer.start()
        try:
            while True:
                time.sleep(5)
        except:
            self.observer.stop()
            print("Error")

        self.observer.join()


class Handler(FileSystemEventHandler):

    @staticmethod
    def on_any_event(event):

        if event.is_directory is False:
            return None

        elif event.event_type == 'created':
            # Take any action here when a file is first created.
            print("Received created event - %s." % event.src_path)

        elif event.event_type == 'deleted':
            # Taken any action here when a file is modified.
            print("Received deleted event - %s." % event.src_path)
        else:
            return None


if __name__ == '__main__':
    w = Watcher()
    w.DIRECTORY_TO_WATCH = "/home/asteinbe/openvpn/test"
    w.run()
