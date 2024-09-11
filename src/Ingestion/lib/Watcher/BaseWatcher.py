import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import logging

logger = logging.getLogger(__name__)
logger.propagate = True

class BaseWatcher(FileSystemEventHandler):
    def __init__(self, watch_directory: str, **kwargs):
        self._watch_directory = watch_directory
        self._sleep_time = kwargs.get('sleep_time', 1)
        self._observer = Observer()

    def on_created(self, event):
        if not event.is_directory:
            logger.info(f'Archivo creado: {event.src_path}')

    def on_deleted(self, event):
        print(f'Archivo eliminado: {event.src_path}')
        if not event.is_directory:
            logger.info(f'Archivo eliminado: {event.src_path}')

    def on_modified(self, event):
        if not event.is_directory:
            logger.info(f'Archivo modificado: {event.src_path}')

    def on_moved(self, event):
        if not event.is_directory:
            logger.info(f'Archivo movido de {event.src_path} a {event.dest_path}')

    def start_watch(self):
        self._observer.schedule(self, self._watch_directory, recursive=True)
        self._observer.start()
        logger.info(f"Observando cambios en el directorio: {self._watch_directory}")
        try:
            while True:
                time.sleep(self._sleep_time)
        except KeyboardInterrupt:
            self._observer.stop()
            logger.info("Observador detenido.")
        self._observer.join()
