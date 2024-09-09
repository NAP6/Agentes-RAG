import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import logging

class BaseWatcher(FileSystemEventHandler):
    def __init__(self, watch_directory: str, **kwargs):
        self._watch_directory = watch_directory
        self._sleep_time = kwargs.get('sleep_time', 1)
        self._observer = Observer()
        # Obtener el logger que podría estar configurado en el nivel superior
        self._logger = logging.getLogger(__name__)
        self._logger.propagate = True

    def on_created(self, event):
        print(f'3 Archivo creado: {event.src_path}')
        if not event.is_directory:
            print(f'Archivo creado: {event.src_path}')
            self._logger.info(f'Archivo creado: {event.src_path}')

    def on_deleted(self, event):
        print(f'Archivo eliminado: {event.src_path}')
        if not event.is_directory:
            print(f'Archivo eliminado: {event.src_path}')
            self._logger.info(f'Archivo eliminado: {event.src_path}')

    def on_modified(self, event):
        if not event.is_directory:
            print(f'Archivo modificado: {event.src_path}')
            self._logger.info(f'Archivo modificado: {event.src_path}')

    def on_moved(self, event):
        if not event.is_directory:
            print(f'Archivo movido de {event.src_path} a {event.dest_path}')
            self._logger.info(f'Archivo movido de {event.src_path} a {event.dest_path}')

    def start_watch(self):
        self._observer.schedule(self, self._watch_directory, recursive=True)
        self._observer.start()
        print(f"Wathcing for changes in directory: {self._watch_directory}")
        self._logger.info(f"Observando cambios en el directorio: {self._watch_directory}")
        try:
            while True:
                time.sleep(self._sleep_time)
        except KeyboardInterrupt:
            self._observer.stop()
            self._logger.info("Observador detenido.")
            print("Observador detenido.")
        self._observer.join()
