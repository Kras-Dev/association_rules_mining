import inspect
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)

class BaseLogger:
    """
    Миксин для унифицированного логирования.
    Автоматически определяет имя класса и метода, откуда был вызван лог.
    """
    def __init__(self, verbose: bool = True):
        self.verbose = verbose

    def _get_context(self) -> str:
        """Определяет контекст вызова [ИмяКласса. Метод]"""
        cls_name = self.__class__.__name__
        stack = inspect.stack()

        # Ищем в стеке первый метод, который не является частью BaseLogger
        # Обычно это индекс 2, но мы подстрахуемся
        method_name = "unknown"
        for frame in stack[1:]:
            if frame.function not in ['_log_info', '_log_warning', '_log_error', '_get_context']:
                method_name = frame.function
                break

        return f"[{cls_name}.{method_name}]"

    def _log_info(self, message: str):
        """Информационные сообщения (фильтруются по verbose)"""
        if self.verbose:
            logger.info(f"{self._get_context()}: {message}")

    def _log_warning(self, message: str):
        """Важные предупреждения (пишутся всегда)"""
        logger.warning(f"{self._get_context()}: {message}")

    def _log_error(self, message: str):
        """Критические ошибки (пишутся всегда)"""
        logger.error(f"{self._get_context()}: {message}")