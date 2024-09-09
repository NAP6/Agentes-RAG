import os

class Config:
    def __init__(self):
        # Configurar rutas de directorios
        self.base_dir = self.get_env_variable('SRC_DATA_PATH', '/')
        self.source_dir = os.path.join(self.base_dir, "raw_files")
        self.out_dir = os.path.join(self.base_dir, "nodes")
        self.images_dir = os.path.join(self.base_dir, "img")
        self.log_dir = os.path.join(self.base_dir, "logs")
        self.meta_dir = os.path.join(self.base_dir, "metadata")

        # Configurar otras variables específicas
        self.llm_model = "gemini-1.5-pro-001"
        self.gcp_credentials_path = self.get_env_variable('GCP_CREDENTIALS_PATH', None)

        # Verificar que se haya definido la variable de entorno 'GCP_CREDENTIALS_PATH'
        if self.gcp_credentials_path is None:
            print("Error: No se ha definido la variable de entorno 'GCP_CREDENTIALS_PATH'.")
            raise ValueError("No se ha definido la variable de entorno 'GCP_CREDENTIALS_PATH'.")

        # Asegurar la existencia de directorios
        self.ensure_directories_exist()

        # Imprimir configuración
        print("Configuración:")
        print(f"  source_dir: {self.source_dir}")
        print(f"  out_dir: {self.out_dir}")
        print(f"  images_dir: {self.images_dir}")
        print(f"  log_dir: {self.log_dir}")
        print(f"  llm_model: {self.llm_model}")
        print(f"  meta_dir: {self.meta_dir}")

    def get_env_variable(self, var_name, default=None):
        """Obtiene una variable de entorno y emite una advertencia si no está definida."""
        value = os.getenv(var_name)
        if value is None:
            if default is None:
                print(f"Advertencia: Variable de entorno '{var_name}' no está definida y no hay un valor por defecto.")
            else:
                print(f"Advertencia: Variable de entorno '{var_name}' no está definida. Usando valor por defecto: {default}")
            return default
        return value

    def ensure_directories_exist(self):
        """Crea los directorios si no existen."""
        os.makedirs(self.source_dir, exist_ok=True)
        os.makedirs(self.out_dir, exist_ok=True)
        os.makedirs(self.images_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.meta_dir, exist_ok=True)
