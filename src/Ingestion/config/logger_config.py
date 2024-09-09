import logging
import sys
import os

def setup_logger(log_dir):
    # Configurar el nivel de logging del logger raíz
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    # Crear handlers para registrar en la salida estándar y en un archivo
    stdoutHandler = logging.StreamHandler(stream=sys.stdout)
    errHandler = logging.FileHandler(os.path.join(log_dir, "error.log"))

    # Establecer los niveles de log en los handlers
    stdoutHandler.setLevel(logging.DEBUG)
    errHandler.setLevel(logging.ERROR)

    # Crear un formato de log usando atributos de Log Record
    fmt = logging.Formatter(
        "%(name)s: %(asctime)s | %(levelname)s | %(filename)s:%(lineno)s | %(process)d >>> %(message)s"
    )

    # Establecer el formato de log en cada handler
    stdoutHandler.setFormatter(fmt)
    errHandler.setFormatter(fmt)

    # Añadir cada handler al objeto Logger
    logger.addHandler(stdoutHandler)
    logger.addHandler(errHandler)

    return logger
