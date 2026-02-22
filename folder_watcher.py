import time
import os
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import database_manager

DATA_PATH = "data"

class FileEventHandler(FileSystemEventHandler):
    """Handles file system events for the data directory."""
    
    def on_created(self, event):
        if not event.is_directory and event.src_path.endswith(".pdf"):
            print(f"File created: {event.src_path}")
            # Add a small delay to ensure file write is complete
            time.sleep(1)
            database_manager.process_file(event.src_path)

    def on_modified(self, event):
        if not event.is_directory and event.src_path.endswith(".pdf"):
            print(f"File modified: {event.src_path}")
            # Add a small delay to ensure file write is complete
            time.sleep(1)
            database_manager.process_file(event.src_path)

def main():
    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH)
        print(f"Created data directory at {DATA_PATH}")

    event_handler = FileEventHandler()
    observer = Observer()
    observer.schedule(event_handler, path=DATA_PATH, recursive=False)
    
    observer.start()
    print(f"Monitoring {DATA_PATH} for new PDF files...")
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    
    observer.join()

if __name__ == "__main__":
    main()
