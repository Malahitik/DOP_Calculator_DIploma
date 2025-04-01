import tkinter as tk
from tkinter import simpledialog
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.optimize import minimize
import csv
from tkinter import filedialog, messagebox
import webbrowser
import os
import base64
import tempfile
import os
import sys
from PIL import Image, ImageTk


# --- FUNCTIONS FOR DOP CALCULATION ---

def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)


UPDATES_CONTENT_BASE64 = (
    "PT09PT0g0JLRgdC1INC+0LHQvdC+0LLQu9C10L3QuNGPINC4INC00L7RgNCw0LHQvtGC0LrQuCA9PT09PQ0KDQo9PT0g0JLQtdGA0YHQuNGPIDEuMCA9PT0NCi0g0JTQvtCx0LDQstC70LXQvSDRgNCw0YHRh9C10YIgRE9QLg0KLSDQlNC+0LHQsNCy0LvQtdC90Ysg0LrQvdC+0L/QutC4INC40L3RgtC10YDRhNC10LnRgdCwLg0KLSDQoNC10LDQu9C40LfQvtCy0LDQvSDRjdC60YHQv9C+0YDRgiDQs9GA0LDRhNC40LrQvtCyINCyIFBORy4NCi0g0KDQtdCw0LvQuNC30L7QstCw0L0g0YDQsNGB0YfQtdGCIERPUCDQuCDQv9C+0YHRgtGA0L7QtdC90LjQtSDQs9GA0LDRhNC40LrQvtCyDQotINCg0LXQsNC70LjQt9C+0LLQsNC9INGA0LDRgdGH0LXRgiDQuNC00LXQsNC70YzQvdC+0LPQviDRgNCw0YHQv9C+0LvQvtC20LXQvdC40LUg0YDQsNC00LDRgNC+0LINCg0KPT09INCS0LXRgNGB0LjRjyAxLjEgPT09DQotINCY0YHQv9GA0LDQstC70LXQvdCwINC+0YjQuNCx0LrQsCDRgSDQstCy0L7QtNC+0Lwg0LrQvtC+0YDQtNC40L3QsNGCINGA0LDQtNCw0YDQvtCyLg0KLSDQo9C70YPRh9GI0LXQvSDQuNC90YLQtdGA0YTQtdC50YEg0L/QvtC70YzQt9C+0LLQsNGC0LXQu9GPLg0KLSDQlNC+0LHQsNCy0LvQtdC90LjQtSDQn9GA0LjQvNC10YDQsCDQstGF0L7QtNC90YvRhSDQtNCw0L3QvdGL0YUg0L3QsCDQs9C70LDQstC90YvQuSDRjdC60YDQsNC9INC40L3RgtC10YDRhNC10LnRgdCwDQoNCj09PSDQktC10YDRgdC40Y8gMS4yID09PQ0KLSDQntC/0YLQuNC80LjQt9C40YDQvtCy0LDQvSDQsNC70LPQvtGA0LjRgtC8INGA0LDRgdGH0LXRgtCwINC40LTQtdCw0LvRjNC90YvRhSDQv9C+0LfQuNGG0LjQuSDRgNCw0LTQsNGA0L7Qsi4NCi0g0JTQvtCx0LDQstC70LXQvdCwINCy0L7Qt9C80L7QttC90L7RgdGC0Ywg0Y3QutGB0L/QvtGA0YLQsCDRgNC10LfRg9C70YzRgtCw0YLQvtCyINCyIENTVi4NCi0g0JTQvtCx0LDQstC70LXQvdCwICLQodC/0YDQsNCy0LrQsCIgLSDRgNGD0LrQvtCy0L7QtNGB0YLQstC+INC/0L4g0Y3QutGB0L/Qu9GD0LDRgtCw0YbQuNC4DQoNCj09PSDQktC10YDRgdC40Y8gMS4zID09PQ0KLSDQlNC+0LHQsNCy0LvQtdC90L4g0YHQvtC30LTQsNC90LjQtSDQs9GA0LDRhNC40LrQsCDQsiDQv9GA0LDQstC+0Lkg0YfQsNGB0YLQuCDRjdC60YDQsNC90LANCi0g0JTQvtCx0LDQstC70LXQvdCwINC60L3QvtC/0LrQsCDQv9C+0YHQt9Cy0L7Qu9GP0Y7RidCw0Y8g0L7RgtC+0LHRgNCw0LbQsNGC0Ywg0LrQvtC90YLRg9GAINCz0YDQsNGE0LjQuiDQutC+0L3RgtGD0YDQsCAi0J/QvtGB0YLRgNC+0LjRgtGMINC60L7QvdGC0YPRgCINCg0KPT09INCS0LXRgNGB0LjRjyAxLjQgPT09DQotINCU0L7QsdCw0LLQu9C10L3QsCDQutC90L7Qv9C60LAg0L/QvtC30LLQvtC70Y/RjtGJ0LDRjyDQvtGC0L7QsdGA0LDQttCw0YLRjCDQvtGB0L3QvtCy0L3Rg9GOINGC0L7Rh9C60YMg0L3QsCDQs9GA0LDRhNC40LrQtSAi0JTQvtCx0LDQstC40YLRjCDQvtGB0L3QvtCy0L3Rg9GOINGC0L7Rh9C60YMiDQotINCU0L7QsdCw0LLQu9C10L3RiyDQutC90L7Qv9C60Lgg0YPQtNCw0LvQtdC90LjRjyDRgtC+0YfQtdC6DQotINCU0L7QsdCw0LLQu9C10L3QsCDRgdC10YLQutCwINCz0YDQsNGE0LjQutCwINC4INC+0YHQuA0KDQo9PT0g0JLQtdGA0YHQuNGPIDEuNSA9PT0NCi0g0JTQvtCx0LDQstC70LXQvdCwINGE0YPQvdC60YbQuNGPINGD0LTQsNC70LXQvdC40Y8g0L7Qv9GA0LXQtNC10LvQtdC90L3QvtCz0L4g0YDQsNC00LDRgNCwDQotINCU0L7QsdCw0LLQu9C10L3QsCDRhNGD0L3QutGG0LjRjyDRgdC+0YXRgNCw0L3QtdC90LjRjyDRhNCw0LnQu9CwINCyINC20LXQu9Cw0LXQvNGD0Y4g0LTQuNGA0LXQutGC0L7RgNC40Y4NCi0g0JTQvtCx0LDQstC70LXQvdCwINGE0YPQvdC60YbQuNGPICLQn9C10YDQtdC30LDQs9GA0YPQt9C40YLRjCDQn9GA0LjQu9C+0LbQtdC90LjQtSIg0LIg0YHQu9GD0YfQsNC1INC90LXQv9GA0LXQtNCy0LjQtNC10L3QvdC+0Lkg0L7RiNC40LHQutC4DQoNCj09PSDQktC10YDRgdC40Y8gMS42ID09PQ0KLSDQmNGB0L/RgNCw0LLQu9C10L3QsCDQstC+0LfQvNC+0LbQvdC+0YHRgtGMINC90LUg0LfQsNC60YDRi9Cy0LDRjyDQvtC60L3QviDQstCy0L7QtNCwINC00LDQvdC90YvRhSDRgNCw0LTQsNGA0LAg0LLQstC+0LTQuNGC0Ywg0LTQsNC90L3Ri9C1INGB0L3QvtCy0LAo0JzQsNC60YEg0LHQvtC70YzRiNC1INC90LUg0L/RgNC40L/RgNGP0YfQtdGCINGO0YDQutC40YUg0L/Rh9C10LspDQotINCU0L7QsdCw0LLQu9C10L3QsCDQstC+0LfQvNC+0LbQvdC+0YHRgtGMINC/0YDQvtGB0LzQvtGC0YDQsCDQuNGB0YLQvtGA0LjQuCDQvtCx0L3QvtCy0LvQtdC90LjQuQ0KLSDQktCy0LLQtdC00LXQvdCwINGE0YPQvdC60YbQuNGPINCx0LvQvtC60LjRgNGD0Y7RidCw0Y8g0LTQvtGB0YLRg9C/INC6ICLQktGL0YfQuNGB0LvQuNGC0YwgRE9QIiwg0LHQtdC3INC/0L7QtNGC0LLQtdGA0LbQtNC10L3QuNGPINCy0LLQtdC00LXQvdC90YvRhSDQtNCw0L3QvdGL0YUNCi0g0JTQvtCx0LDQstC70LXQvdCwINC60L3QvtC/0LrQsCAi0J7QsdC90L7QstC70LXQvdC40Y8iLg==")


def open_updates():
    try:
        # Декодирование содержимого из base64
        updates_content = base64.b64decode(UPDATES_CONTENT_BASE64).decode('utf-8')

        # Создание временного файла для отображения содержимого
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as temp_file:
            temp_file.write(updates_content)
            temp_file_path = temp_file.name

            # Открытие временного файла с помощью стандартного приложения системы
            if os.name == 'nt':  # Для Windows
                os.startfile(temp_file_path)
            else:  # Для других систем
                webbrowser.open(temp_file_path)
    except Exception as e:
        messagebox.showerror("Ошибка", f"Не удалось открыть файл обновлений: {str(e)}")


def calculate_dop(point, radars):
    """
    Вычисляет DOP по формуле: DOP = sqrt(trace(P)), где P = (H^T H)^{-1}.
    H формируется из единичных векторов, направленных от целевой точки к радару.
    """
    x, y = point
    H = []
    for radar in radars:
        xi, yi = radar
        ri = np.sqrt((xi - x) ** 2 + (yi - y) ** 2)
        if ri == 0:
            raise ValueError("Радар не может находиться в той же точке, что и цель.")
        H.append([(xi - x) / ri, (yi - y) / ri])
    H = np.array(H)
    try:
        P = np.linalg.inv(H.T @ H)
    except np.linalg.LinAlgError:
        return float('Самопересечение или сингулярная матрица')  # Сингулярная матрица
    dop = np.sqrt(np.trace(P))
    return dop


# --- HELPER FUNCTIONS ---

def is_point_inside_polygon(point, polygon):
    x, y = point
    n = len(polygon)
    inside = False

    p1x, p1y = polygon[0]
    for i in range(n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xints = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xints:
                        inside = not inside
        p1x, p1y = p2x, p2y

    return inside


def is_valid_polygon(polygon):
    n = len(polygon)
    if n < 3:
        print("Контур должен состоять минимум из 3-х точек")
        return False

    def orientation(a, b, c):
        val = (b[1] - a[1]) * (c[0] - b[0]) - (b[0] - a[0]) * (c[1] - b[1])
        if abs(val) < 1e-9:  # Учитываем погрешность вычислений
            return 0  # Коллинеарные
        return 1 if val > 0 else 2  # По часовой/против часовой

    def on_segment(p, q, r):
        if (min(p[0], r[0]) <= q[0] <= max(p[0], r[0]) and
                min(p[1], r[1]) <= q[1] <= max(p[1], r[1])):
            return True
        return False

    def do_segments_intersect(p1, p2, q1, q2):
        o1 = orientation(p1, p2, q1)
        o2 = orientation(p1, p2, q2)
        o3 = orientation(q1, q2, p1)
        o4 = orientation(q1, q2, p2)

        # Общее пересечение
        if o1 != o2 and o3 != o4:
            return True

        # Специальные случаи (коллинеарность)
        if o1 == 0 and on_segment(p1, q1, p2):
            return True
        if o2 == 0 and on_segment(p1, q2, p2):
            return True
        if o3 == 0 and on_segment(q1, p1, q2):
            return True
        if o4 == 0 and on_segment(q1, p2, q2):
            return True

        return False

    for i in range(n):
        p1, p2 = polygon[i], polygon[(i + 1) % n]
        for j in range(i + 2, n):
            q1, q2 = polygon[j], polygon[(j + 1) % n]

            # Исключаем случай, когда отрезки являются соседними
            if (j != (i + 1) % n) and ((i + 1) % n != j):
                if do_segments_intersect(p1, p2, q1, q2):
                    print(f"Пересечение отрезков: {p1} -> {p2} и {q1} -> {q2}")
                    return True

    # Проверка на коллинеарность
    vectors = [np.array(polygon[i + 1]) - np.array(polygon[i]) for i in range(n - 1)]
    cross_products = [np.cross(vectors[i], vectors[i + 1]) for i in range(len(vectors) - 1)]
    if any(abs(cp) < 1e-9 for cp in cross_products):
        print("Обнаружены коллинеарные точки.")
        return False

    return True


def distance_point_to_segment(point, seg_start, seg_end):
    """
    Вычисляет расстояние от точки до отрезка.
    """
    p = np.array(point)
    a = np.array(seg_start)
    b = np.array(seg_end)
    if np.all(a == b):
        return np.linalg.norm(p - a)
    # Проекция точки p на прямую ab:
    t = np.dot(p - a, b - a) / np.dot(b - a, b - a)
    t = max(0, min(1, t))
    projection = a + t * (b - a)
    return np.linalg.norm(p - projection)


def generate_ideal_radars_inside_polygon(polygon, num_radars, target):
    """
    Генерирует идеальные позиции для радаров. Центр распределения — заданная пользователем точка (target).
    Чтобы гарантировать, что радары находятся внутри полигона, вычисляется минимальное расстояние от target до ребер полигона,
    и используется доля от этого расстояния в качестве радиуса размещения.
    """
    # Вычисление минимального расстояния от target до каждого ребра полигона:
    distances = []
    n = len(polygon)
    for i in range(n):
        seg_start = polygon[i]
        seg_end = polygon[(i + 1) % n]
        d = distance_point_to_segment(target, seg_start, seg_end)
        distances.append(d)
    min_distance = min(distances)

    # Используем, например, 80% от минимального расстояния для безопасного размещения:
    radar_radius = 0.8 * min_distance

    radars = []
    angle_step = 2 * np.pi / num_radars
    for i in range(num_radars):
        angle = i * angle_step
        x = target[0] + radar_radius * np.cos(angle)
        y = target[1] + radar_radius * np.sin(angle)
        # Если по какой-либо причине точка выходит за пределы полигона, можно скорректировать (хотя выбор 0.8 гарантирует,
        # что радары останутся внутри, если target внутри полигона)
        if not is_point_inside_polygon([x, y], polygon):
            # Если точка оказалась вне полигона, используем target как координату радара (резервный вариант)
            x, y = target
        radars.append([x, y])
    return radars


def save_results_to_csv(point, radars, user_dop, ideal_radars, ideal_dop, filename="results.csv"):
    with open(filename, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Тип", "X", "Y"])
        writer.writerow(["Цель"] + point)
        writer.writerow(["DOP (пользовательские данные)", "", user_dop])
        for radar in radars:
            writer.writerow(["Радар (пользовательский)"] + radar)
        writer.writerow(["DOP (идеальное расположение)", "", ideal_dop])
        for radar in ideal_radars:
            writer.writerow(["Радар (идеальный)"] + radar)


# --- ГЛАВНЫЙ КЛАСС GUI ---
# Основной интерфейс окна

class DopCalculatorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("DOP Calculator")

        self.top_frame = tk.Frame(root)
        self.top_frame.grid(row=0, column=0, columnspan=3, padx=10, pady=10, sticky='w')

        # Настройка веса строк и столбцов для гибкого размещения виджетов
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_rowconfigure(1, weight=1)

        # Добавляем кнопки в новый фрейм
        self.example_button = tk.Button(self.top_frame, text="Пример", command=self.show_example, font=('Arial', 15))
        self.example_button.pack(side=tk.LEFT, padx=5)

        self.reset_button = tk.Button(self.top_frame, text="Перезагрузить приложение", command=self.reset,
                                      font=('Arial', 15))
        self.reset_button.pack(side=tk.LEFT, padx=5)

        self.update_button = tk.Button(self.top_frame, text="Обновления", command=open_updates, font=('Arial', 15))
        self.update_button.pack(side=tk.LEFT, padx=5)

        self.save_plot_button = tk.Button(self.top_frame, text="Сохранить график", command=self.export_plot,
                                          font=('Arial', 15))
        self.save_plot_button.pack(side=tk.LEFT, padx=5)

        self.help_button = tk.Button(self.top_frame, text="Справка", command=self.show_help, font=('Arial', 15))
        self.help_button.pack(side=tk.LEFT, padx=2)

        # Создаем фреймы для организации элементов UI
        self.left_frame = tk.Frame(root)
        self.left_frame.grid(row=1, column=0, padx=10, pady=10, sticky='nsew')

        self.center_frame = tk.Frame(root)  # Новый центральный фрейм
        self.center_frame.grid(row=1, column=1, padx=10, pady=10, sticky='nsw')

        self.center2_frame = tk.Frame(root)
        self.center2_frame.grid(row=1, column=2, padx=10, pady=10, sticky='nsw')

        self.center3_frame = tk.Frame(root)
        self.center3_frame.grid(row=1, column=3, padx=0, pady=10, sticky='nsw')

        self.right_frame = tk.Frame(root)
        self.right_frame.grid(row=1, column=4, padx=10, pady=10, sticky='nsw')



        # Вместо того чтобы сразу размещать поля в self.left_frame,
        # создадим вложенный фрейм для точек полигона
        self.polygon_frame = tk.Frame(self.center_frame)
        self.polygon_frame.pack(anchor='w')

        font_center = ('Times New Roman', 15)
        font_right = ('Times New Roman', 18)
        font_left = ('Arial', 11)
        font_condition = ('Times New Roman', 25)
        # Флаг для отслеживания состояния нового окна
        self.example_window_open = False

        # Легенда цвета остается справа
        tk.Label(self.right_frame, text="Условные обозначения:",  justify='left', font=font_condition,fg='black').pack(anchor='w',padx=500, pady=10)
        tk.Label(self.right_frame, text="Черный цвет: Контур", justify='left', font=font_right,fg='black').pack(anchor='w',padx=500)
        tk.Label(self.right_frame, text="Красный цвет: Целевая (основная точка)", justify='left', font=font_right,fg='tomato').pack(anchor='w',padx=500)
        tk.Label(self.right_frame, text="Синий цвет: Введенные пользователем радары", justify='left',font=font_right,fg='blue').pack(anchor='w',padx=500)
        tk.Label(self.right_frame, text="Зеленый цвет: Идеальные радары подставленные системой", justify='left',font=font_right,fg='green').pack(anchor='w',padx=500)

        # Ввод координат полигона слева
        tk.Label(self.left_frame, text="Координаты контура(полигона)", font=font_left).pack(anchor='w')
        self.polygon_entries = []
        self.add_polygon_button = tk.Button(self.left_frame, text="Добавить точку контура(полигона)",command=self.add_polygon_entry, font=font_left).pack(anchor='w', pady=10)

        # Кнопка для удаления точки контура
        tk.Button(self.left_frame, text="Удалить точку контура", command=self.remove_polygon_entry,font=font_left).pack(anchor='w')

        # Кнопка для построения контура
        tk.Button(self.left_frame, text="Построить контур", command=self.plot_polygon, font=font_left).pack(anchor='w',pady=10)

        # Ввод целевой точки и радаров слева
        tk.Label(self.center2_frame, text="Введите координаты основной точки(x, y):", font=font_left).pack(anchor='w', pady=10)

        tk.Label(self.center3_frame, text="Координата X:", font=font_left).pack(anchor='w',pady=5)
        self.point_entry_x = tk.Entry(self.center3_frame)
        self.point_entry_x.pack(anchor='nw',)
        tk.Label(self.center3_frame, text="Координата Y:", font=font_left).pack(anchor='w', pady=5)
        self.point_entry_y = tk.Entry(self.center3_frame)
        self.point_entry_y.pack(anchor='sw')


        for _ in range(3):  # Например, изначально 4 точки полигона
            self.add_polygon_entry()

        # Кнопка для построения основной точки
        tk.Button(self.center2_frame, text="Построить основную точку", command=self.plot_main_point, font=font_left).pack(
            anchor='w')

        tk.Label(self.center2_frame, text="Кол-во Радаров:", font=font_left).pack(anchor='w', pady=10)
        self.num_radars_entry = tk.Entry(self.center2_frame)
        self.num_radars_entry.pack(anchor='w', pady=5)

        tk.Button(self.center2_frame, text="Добавить радары", command=self.add_radars, font=font_left).pack(anchor='w', pady = 10)
        tk.Label(self.center3_frame, text="Список радаров", font=font_left).pack(anchor='w', pady=10)
        self.radars_listbox = tk.Listbox(self.center3_frame)
        self.radars_listbox.pack(anchor='w', pady=10)

        # Удаления радаров
        tk.Button(self.center2_frame, text="Удалить выбранный радар", command=self.remove_selected_radar,
                  font=font_left).pack(anchor='w', pady=5)

        # Кнопки Reset и Confirm слева
        tk.Button(self.left_frame, text="Сбросить все значения", command=self.reset, font=font_left).pack(
            anchor='w', pady=50, )
        tk.Button(self.left_frame, text="Подтвердить значения введенных данных", command=self.confirm,
                  font=font_left).pack(anchor='w', pady=5)

        # Кнопка для расчета DOP слева
        self.calculate_button = tk.Button(self.left_frame, text="Вычислить DOP", command=self.calculate,
                                          font=font_left, state=tk.DISABLED)
        self.calculate_button.pack(anchor='w', pady=5)

        # Метка для вывода результатов
        self.result_label = tk.Label(self.left_frame, text="")
        self.result_label.pack(anchor='w')

        # Поле для графика справа
        self.figure = plt.Figure(figsize=(8, 6), dpi=150)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.right_frame)
        self.canvas.get_tk_widget().pack(pady= 20, padx=20)

        # Инициализация флага подтверждения данных
        self.data_confirmed = False

    def update_scale(self, value):
        scale_factor = float(value) / 100.0

        for child in self.root.winfo_children():
            if isinstance(child, tk.Frame):
                for sub_child in child.winfo_children():
                    self.update_widget(sub_child, scale_factor)
            else:
                self.update_widget(child, scale_factor)

    def update_widget(self, widget, scale_factor):
        if isinstance(widget, (tk.Button, tk.Label, tk.Entry)):
            current_font = widget.cget("font")
            if current_font:
                font_family, *font_info = current_font.split(' ')
                if len(font_info) >= 1:
                    try:
                        font_size = int(font_info[0])
                        new_font_size = min(int(font_size * scale_factor),
                                            30)  # Ограничение максимального размера шрифта
                        widget.config(font=(font_family, new_font_size))
                    except ValueError:
                        pass

            if isinstance(widget, (tk.Button, tk.Label)):
                padding = widget.cget("padx")
                if isinstance(padding, (int, float)):
                    new_padding = int(padding * scale_factor)
                    widget.config(padx=new_padding)

        if isinstance(widget, tk.Canvas):
            width = widget.cget("width")
            height = widget.cget("height")
            if isinstance(width, str) and width.isdigit():
                width = int(width)
            if isinstance(height, str) and height.isdigit():
                height = int(height)
            if isinstance(width, (int, float)) and isinstance(height, (int, float)):
                new_width = min(int(width * scale_factor), 800)  # Ограничение максимальной ширины
                new_height = min(int(height * scale_factor), 600)  # Ограничение максимальной высоты
                widget.config(width=new_width, height=new_height)

    def show_example(self):
        help_dialog = tk.Toplevel(self.root)
        help_dialog.title("Пример")

        # Устанавливаем размер окна
        width = 1024
        height = 768

        # Получаем размеры экрана
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()

        # Рассчитываем положение окна для центрирования
        x = (screen_width - width) // 2
        y = (screen_height - height) // 2

        # Устанавливаем размер и положение окна
        help_dialog.geometry(f"{width}x{height}+{x}+{y}")

        # Запрещаем изменение размера окна
        help_dialog.resizable(False, False)

        # Блокировка других действий в программе
        help_dialog.grab_set()

        # Устанавливаем изображение на фон окна
        background_image_path = resource_path("Help.jpg")
        background_image = Image.open(background_image_path)
        background_photo = ImageTk.PhotoImage(background_image)

        # Создаем Canvas для отображения изображения и текста
        canvas = tk.Canvas(help_dialog, width=width, height=height, highlightthickness=0)
        canvas.pack(fill="both", expand=True)

        # Отображение изображения на Canvas
        canvas.create_image(0, 0, image=background_photo, anchor=tk.NW)
        canvas.image = background_photo  # Сохраняем ссылку на изображение

        # Добавляем кнопку для закрытия окна
        close_button = tk.Button(help_dialog, text="Закрыть", command=help_dialog.destroy, font=('Arial', 12))
        close_button_window = canvas.create_window(
            width // 2,
            height - 110,
            window=close_button
        )

        # Устанавливаем фокус на диалоговое окно
        help_dialog.focus_set()

    def on_example_window_close(self, window):
        self.example_window_open = False
        self.example_button.config(state=tk.NORMAL)
        window.destroy()

    # Метод для построения контура
    def plot_polygon(self):
        # Очистка предыдущего графика
        self.figure.clear()
        self.ax = self.figure.add_subplot(111)  # Инициализация self.ax

        # Извлечение координат точек контура
        polygon = []
        for frame, entry_x, entry_y in self.polygon_entries:
            try:
                x = float(entry_x.get())
                y = float(entry_y.get())
                polygon.append([x, y])
            except ValueError:
                messagebox.showwarning("Ошибка", "Некорректные координаты точки контура.")
                return

        if not polygon:
            # Если нет точек, рисуем пустой график с сеткой
            self.ax.set_title("Контур не задан")
            self.ax.set_xlabel("X")
            self.ax.set_ylabel("Y")
            self.ax.grid(True)  # Добавляем сетку
        else:
            # Преобразование списка в массив numpy
            polygon = np.array(polygon)

            # Построение графика
            if len(polygon) == 1:
                # Одна точка
                self.ax.plot(polygon[0][0], polygon[0][1], 'ko', markersize=5, label="Точка контура")
                self.ax.set_title("Одна точка контура")
            elif len(polygon) == 2:
                # Две точки, соединяем их линией
                self.ax.plot(polygon[:, 0], polygon[:, 1], 'k-', linewidth=2, label="Линия контура")
                self.ax.plot(polygon[:, 0], polygon[:, 1], 'ko', markersize=5, label="Точки контура")
                self.ax.set_title("Линия между двумя точками")
            else:
                # Замкнутый контур
                closed_polygon = np.vstack([polygon, polygon[0]])  # Замыкаем контур
                self.ax.plot(closed_polygon[:, 0], closed_polygon[:, 1], 'k-', linewidth=2, label="Контур")
                self.ax.plot(closed_polygon[:, 0], closed_polygon[:, 1], 'ko', markersize=5, label="Точки контура")
                self.ax.set_title("Замкнутый контур")

            # Автоматическое масштабирование осей
            self.ax.set_xlabel("X")
            self.ax.set_ylabel("Y")
            self.ax.grid(True)  # Добавляем сетку

        # Добавление легенды
        self.ax.legend()

        # Обновление графика
        self.canvas.draw()

    def remove_polygon_entry(self):
        if self.polygon_entries:
            # Удаляем последнюю добавленную точку контура
            frame, entry_x, entry_y = self.polygon_entries.pop()
            frame.destroy()  # Удаляем виджет фрейма вместе с полями ввода
        else:
            messagebox.showwarning("Ошибка", "Нет точек контура для удаления.")

    # Метод для построения основной точки
    def plot_main_point(self):
        try:
            # Извлечение координат основной точки
            point_x = float(self.point_entry_x.get())
            point_y = float(self.point_entry_y.get())
            point = [point_x, point_y]

            # Проверка, существует ли уже график
            if not hasattr(self, 'ax') or self.ax is None:
                # Если график еще не создан, создаем его
                self.figure.clear()
                self.ax = self.figure.add_subplot(111)
                self.ax.set_title("Основная точка")
                self.ax.set_xlabel("X")
                self.ax.set_ylabel("Y")

            # Очистка только данных основной точки (если они были)
            for artist in self.ax.lines + self.ax.collections:
                if artist.get_label() == "Основная точка":
                    artist.remove()

            # Построение основной точки
            self.ax.plot(point[0], point[1], 'ro', markersize=8, label="Основная точка")  # Красная точка

            # Автоматическое масштабирование осей
            self.ax.set_xlabel("X")
            self.ax.set_ylabel("Y")
            self.ax.grid(True)  # Добавляем сетку

            # Обновление легенды
            self.ax.legend()

            # Обновление графика
            self.canvas.draw()

        except ValueError:
            messagebox.showwarning("Ошибка", "Некорректные координаты основной точки.")

    # Метод для кнопки удаления радаров
    def remove_selected_radar(self):
        selected_index = self.radars_listbox.curselection()
        if selected_index:
            self.radars_listbox.delete(selected_index)
        else:
            messagebox.showwarning("Ошибка", "Выберите радар для удаления.")

    def add_polygon_entry(self):
        frame = tk.Frame(self.polygon_frame)
        frame.pack(anchor='w', pady=10, padx=10)

        tk.Label(frame, text="X:").pack(side=tk.LEFT)
        entry_x = tk.Entry(frame, width=10)
        entry_x.pack(side=tk.LEFT)

        tk.Label(frame, text="Y:").pack(side=tk.LEFT)
        entry_y = tk.Entry(frame, width=10)
        entry_y.pack(side=tk.LEFT)

        # Сохраняем точку полигона
        self.polygon_entries.append((frame, entry_x, entry_y))

    def add_radars(self):
        try:
            num_radars = int(self.num_radars_entry.get())
            if num_radars <= 0:
                raise ValueError("Количество радаров должно быть положительным числом.")
            if num_radars == 0:
                raise ValueError("Минимальное количество радаров = 1")

            self.root.attributes('-topmost', True)  # Держим основное окно сверху

            for _ in range(num_radars):
                dialog = tk.Toplevel(self.root)  # Создаём диалоговое окно вручную
                dialog.title("Ввод координат радаров")
                dialog.transient(self.root)  # Привязываем к основному окну
                dialog.grab_set()  # Захватываем фокус

                # Определяем размеры экрана
                screen_width = self.root.winfo_screenwidth()
                screen_height = self.root.winfo_screenheight()

                # Размеры окна диалога
                width, height = 300, 150

                # Вычисляем координаты для центрирования окна
                x = (screen_width - width) // 2
                y = (screen_height - height) // 2

                # Устанавливаем геометрию окна
                dialog.geometry(f"{width}x{height}+{x}+{y}")

                tk.Label(dialog, text="Введите координаты радара (x y):").pack(pady=5)

                entry = tk.Entry(dialog)
                entry.pack(pady=5)
                entry.focus_set()  # Устанавливаем фокус на поле ввода

                error_label = tk.Label(dialog, fg='red')  # Метка для отображения ошибок
                error_label.pack(pady=5)

                valid_input = False  # Флаг для проверки валидности ввода

                # Функция для подтверждения данных
                def submit():
                    nonlocal valid_input
                    radar = entry.get()
                    if radar:
                        try:
                            x, y = map(float, radar.split())
                            self.radars_listbox.insert(tk.END, f"({x}, {y})")
                            dialog.destroy()
                            valid_input = True
                        except ValueError:
                            error_label.config(text="Введите два числа через пробел.")  # Отображаем ошибку в метке

                # Привязка клавиши Enter к функции submit
                dialog.bind("<Return>", lambda event=None: submit())
                # Кнопка "Принять"
                tk.Button(dialog, text="Принять", command=submit).pack(pady=5)

                while not valid_input:
                    dialog.wait_window()  # Ждём закрытия окна перед созданием следующего
                    if not valid_input:
                        entry.delete(0, tk.END)  # Очищаем поле ввода
                        entry.focus_set()  # Возвращаем фокус на поле ввода
                        error_label.config(text="")  # Сбрасываем сообщение об ошибке

                self.root.attributes('-topmost', False)  # Снимаем topmost
        except ValueError as e:
            messagebox.showerror("Ошибка", str(e))

    def reset(self):
        # Очистка всех полей ввода полигона
        for frame, entry_x, entry_y in self.polygon_entries:
            frame.destroy()
        self.polygon_entries.clear()

        # Очистка всех полей ввода
        self.point_entry_x.delete(0, tk.END)
        self.point_entry_y.delete(0, tk.END)
        self.num_radars_entry.delete(0, tk.END)
        self.radars_listbox.delete(0, tk.END)

        # Очистка поля результата
        self.result_label.config(text="")

        # Очистка графика
        self.figure.clear()
        self.canvas.draw()

        # Деактивация кнопки "Вычислить DOP"
        self.calculate_button.config(state=tk.DISABLED)

        # Сброс флага подтверждения данных
        self.data_confirmed = False

        # Пересоздание начального набора полей ввода точек полигона
        for _ in range(3):  # Например, изначально 4 точки полигона
            self.add_polygon_entry()

    def confirm(self):
        try:
            polygon = []
            for frame, entry_x, entry_y in self.polygon_entries:
                x = float(entry_x.get())
                y = float(entry_y.get())
                polygon.append([x, y])

            if not is_valid_polygon(polygon):
                raise ValueError("Контур некорректен (самопересечение или недостаточно точек).")

            point_x = float(self.point_entry_x.get())
            point_y = float(self.point_entry_y.get())
            point = [point_x, point_y]

            if not is_point_inside_polygon(point, polygon):
                raise ValueError("Точка должна находиться внутри контура.")

            num_radars = int(self.num_radars_entry.get())
            if num_radars <= 0:
                raise ValueError("Количество радаров должно быть положительным числом.")
            if num_radars != int(num_radars):
                raise ValueError("Число радаров должно быть целым")

            radars = []
            for i in range(self.radars_listbox.size()):
                radar_str = self.radars_listbox.get(i)
                if not radar_str:
                    raise ValueError("Список радаров содержит пустые значения.")
                radar = list(map(float, radar_str.strip("()").split(", ")))
                if not is_point_inside_polygon(radar, polygon):
                    raise ValueError(f"Радар ({radar[0]}, {radar[1]}) должен находиться внутри контура.")
                radars.append(radar)
            messagebox.showinfo("Успешно", "Можно вычислить DOP")
            self.calculate_button.config(state=tk.NORMAL)
            self.data_confirmed = True

        except Exception as e:
            messagebox.showerror("Ошибка", str(e))

    def calculate(self):
        try:
            polygon = []
            for frame, entry_x, entry_y in self.polygon_entries:
                x = float(entry_x.get())
                y = float(entry_y.get())
                polygon.append([x, y])

            if not is_valid_polygon(polygon):
                raise ValueError("Подтвердите значения введенных данных")

            point_x = float(self.point_entry_x.get())
            point_y = float(self.point_entry_y.get())
            point = [point_x, point_y]

            if not is_point_inside_polygon(point, polygon):
                raise ValueError("Точка радара должна быть внутри полигона")

            radars = []
            for i in range(self.radars_listbox.size()):
                radar_str = self.radars_listbox.get(i)
                if not radar_str:
                    raise ValueError("Количество радаров не может быть 0")
                radar = list(map(float, radar_str.strip("()").split(", ")))
                if not is_point_inside_polygon(radar, polygon):
                    raise ValueError(f"Радар ({radar[0]}, {radar[1]}) должен находиться внутри контура.")
                radars.append(radar)

            user_dop = calculate_dop(point, radars)
            ideal_radars = generate_ideal_radars_inside_polygon(polygon, len(radars), point)
            ideal_dop = calculate_dop(point, ideal_radars)

            # Сохранение результатов в CSV
            save_results_to_csv(point, radars, user_dop, ideal_radars, ideal_dop)

            self.figure.clear()
            ax = self.figure.add_subplot(111)

            polygon_x, polygon_y = zip(*polygon)
            polygon_x = list(polygon_x) + [polygon_x[0]]
            polygon_y = list(polygon_y) + [polygon_y[0]]
            ax.plot(polygon_x, polygon_y, label="Контур(полигон)", color="black")

            ax.scatter(*zip(*radars), label="Радары пользователя", color="blue")
            ax.scatter(*point, label="Целевая точка", color="red")
            ax.scatter(*zip(*ideal_radars), label="Идеальные радары", color="green")

            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width * 0.55, box.height])  # Сужаем ширину осей

            # Перемещаем легенду выше графика и добавляем отступ
            ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.25), fancybox=True, ncol=3)

            # Увеличиваем нижний отступ графика для текста DOP и координат
            self.figure.subplots_adjust(bottom=0.4)  # Увеличиваем нижний отступ

            # Добавляем текст с DOP под осями
            dop_text = f"DOP (User Data): {user_dop:.2f}\nDOP (Ideal Placement): {ideal_dop:.2f}"
            ax.text(
                0.5, -0.15,  # Координаты внутри области Axes
                dop_text,
                fontsize=12,
                ha='center',
                va='top',
                transform=ax.transAxes
            )

            # Добавляем координаты идеальных радаров под графиком
            ideal_radars_text = "Идеальные радары:\n"
            for radar in ideal_radars:
                ideal_radars_text += f"({radar[0]:.2f}, {radar[1]:.2f})\n"

            ax.text(
                0.5, -0.4,  # Увеличиваем расстояние от DOP
                ideal_radars_text,
                fontsize=10,
                ha='center',
                va='top',
                transform=ax.transAxes
            )

            ax.set_aspect('equal', adjustable='box')
            self.canvas.draw()
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def show_help(self):
        # Создаем новое окно справки
        help_dialog = tk.Toplevel(self.root)
        help_dialog.title("Справка")

        # Устанавливаем размер окна
        width = 1024
        height = 768

        # Получаем размеры экрана
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()

        # Рассчитываем положение окна для центрирования
        x = (screen_width - width) // 2
        y = (screen_height - height) // 2

        # Устанавливаем размер и положение окна
        help_dialog.geometry(f"{width}x{height}+{x}+{y}")

        # Запрещаем изменение размера окна
        help_dialog.resizable(False, False)

        # Блокировка других действий в программе
        help_dialog.grab_set()

        # Устанавливаем изображение на фон окна
        background_image_path = resource_path("back1.jpg")
        background_image = Image.open(background_image_path)
        background_photo = ImageTk.PhotoImage(background_image)

        # Создаем Canvas для отображения изображения и текста
        canvas = tk.Canvas(help_dialog, width=width, height=height, highlightthickness=0)
        canvas.pack(fill="both", expand=True)

        # Отображение изображения на Canvas
        canvas.create_image(0, 0, image=background_photo, anchor=tk.NW)
        canvas.image = background_photo  # Сохраняем ссылку на изображение

        # Текст справки
        help_text = """
        Инструкция по использованию DOP Calculator:

        1. Введите координаты контура полигона.

        2. Нажмите кнопку "Построить контур".

        3. Введите координаты основной точки (цели).

        4. Нажмите кнопку "Построить основную точку".

        5. Введите количество радаров.

        6. Нажмите кнопку "Добавить радары" и введите координаты радаров.

        7. Нажмите кнопку "Подтвердить значения введенных данных".

        8. Нажмите кнопку "Вычислить DOP" для получения результатов.

        9. Используйте кнопку "Сохранить график" для экспорта графика.

        10. Используйте кнопку "Справка" для просмотра этой инструкции.

        11. Используйте кнопку "Обновления" для просмотра информации об обновлениях.
        """

        # Увеличиваем шрифт текста
        font_help = ('Arial', 16)

        # Добавляем текст на Canvas
        canvas.create_text(
            width // 2,
            height // 2 - 80,
            text=help_text,
            font=font_help,
            justify='center',
            fill='black',
        )

        # Добавляем кнопку для закрытия окна
        close_button = tk.Button(help_dialog, text="Закрыть", command=help_dialog.destroy, font=('Arial', 12))
        close_button_window = canvas.create_window(
            width // 2,
            height - 110,
            window=close_button
        )

        # Устанавливаем фокус на диалоговое окно
        help_dialog.focus_set()

    def export_plot(self):
        if self.figure is None:
            messagebox.showerror("Ошибка", "График не создан.")
            return

        # Открываем диалоговое окно для выбора имени файла
        filename = filedialog.asksaveasfilename(defaultextension=".png",
                                                filetypes=[("PNG files", "*.png"),
                                                           ("JPEG files", "*.jpg"),
                                                           ("All files", "*.*")])

        if filename:  # Если пользователь выбрал файл
            try:
                self.figure.savefig(filename)
                messagebox.showinfo("Успех", f"График сохранен в файле {filename}")
            except Exception as e:
                messagebox.showerror("Ошибка", f"Не удалось сохранить график: {str(e)}")
        else:
            messagebox.showinfo("Отмена", "Сохранение отменено.")


# --- ЗАПУСК ПРИЛОЖЕНИЯ ---

if __name__ == "__main__":
    root = tk.Tk()
    app = DopCalculatorApp(root)
    root.mainloop()
