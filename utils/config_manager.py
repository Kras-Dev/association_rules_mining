#config_manager.py
import yaml

CONFIG_FILE_PATH = 'config.yaml'

def load_config(config_path=CONFIG_FILE_PATH):
    """Загружает конфигурацию из YAML файла."""
    try:
        with open(config_path, 'r', encoding='utf-8') as file:
            config_data = yaml.safe_load(file)
        return config_data
    except FileNotFoundError:
        print(f"Ошибка: Файл конфигурации '{config_path}' не найден.")
        return None
    except yaml.YAMLError as e:
        print(f"Ошибка при парсинге YAML файла: {e}")
        return None