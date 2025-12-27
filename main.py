import pickle
from pathlib import Path

from back_test.back_test_runner import BacktestRunner
import multiprocessing as mp


if __name__ == "__main__":

    # Указываем путь к папке
    real_path = Path("history/26-12-2025/models")

    # Проверяем наличие папки
    if real_path.exists():
        # Ищем все файлы с расширением .pkl
        for file_path in real_path.glob("*.pkl"):
            print(f"Файл: {file_path.name}")

            try:
                # Открываем в режиме 'rb' (чтение байтов)
                with open(file_path, 'rb') as f:
                    file_data = pickle.load(f)

                    # Если нужно красиво вывести большой словарь:
                    # from pprint import pprint
                    # pprint(file_data)
                    print(file_data)

            except Exception as e:
                print(f"Ошибка при чтении файла {file_path.name}: {e}")

            print("-" * 40)
    else:
        print("Папка не найдена")


    # mp.set_start_method('spawn', force=True)
    # runner = BacktestRunner(max_workers=4)
    # runner.run_parallel()
    # runner.print_summary()
    # runner.save_results()