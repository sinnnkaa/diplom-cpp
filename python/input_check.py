from rknn.api import RKNN

rknn = RKNN()

# 1. Загружаем
if rknn.load_rknn('yolo11_blind.rknn') != 0:
    print("Ошибка загрузки!")
    exit()

# 2. Инициализируем (это напечатает таблицу входов/выходов)
print("\n--- СЕЙЧАС ПОЯВИТСЯ ТАБЛИЦА С ВЫХОДАМИ ---")
ret = rknn.init_runtime()
if ret != 0:
    print(f"Ошибка инициализации! Код: {ret}")
