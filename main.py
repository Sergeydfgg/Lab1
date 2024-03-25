import pydoc

from numpy.random import randint, shuffle, choice
import numpy as np
from datetime import datetime, time
import pandas as pd
from functools import wraps
from pathlib import Path
from time import time as time_calc
from os import getcwd, mkdir
import matplotlib.pyplot as plt
from typing import Callable
import copy

executed_times = dict()


def prepare_data() -> None:
    if not Path(f'{getcwd()}/data').is_dir():
        mkdir(f'{getcwd()}/data')

    for i in range(7):
        passengers = np.arange(1, 100000 * (i+1) + 1)
        shuffle(passengers)
        flight_numbers = np.arange(1, 100000  * (i+1) + 1)
        shuffle(flight_numbers)

        table_to_write = {
                    'Номер рейса': flight_numbers,
                    'Название авиакомпании': [choice(MySorter.available_companies) for _ in range(100000 * (i+1))],
                    'Дата прилета': [datetime(randint(2015, 2024), randint(1, 12), randint(1, 28))
                                     for _ in range(100000 * (i+1))],
                    'Время  прилета по расписанию': [str(time(randint(0, 24), randint(0, 60), randint(0, 60)))
                                                     for _ in range(100000 * (i+1))],
                    'Число пассажиров на борту': passengers,
            }

        table_to_save = pd.DataFrame.from_dict(table_to_write)
        table_to_save.to_csv(f'data/dataset_{i}')


def lab_1():
    prepare_data()

    for dataset_number in range(7):
        test_obj = MySorter(dataset_number)
        print(f'Размер объеката: {len(test_obj)}')

        print(test_obj)
        test_obj.bubble_sort()
        print(test_obj)
        test_obj.reset()

        print(test_obj)
        test_obj.cocktail_sort()
        print(test_obj)
        test_obj.reset()

        print(test_obj)
        test_obj.quicksort()
        print(test_obj)
        test_obj.reset()

    print(*[f'{key}: {val}\n' for key, val in executed_times.items()], sep='')

    prepared_data = np.array(list(executed_times.values()))
    data_size = len(executed_times) / 3 * 1000 + 1000

    plt.plot(np.arange(1000, data_size, 1000), prepared_data[::3])
    plt.plot(np.arange(1000, data_size, 1000), prepared_data[1::3])
    plt.plot(np.arange(1000, data_size, 1000), prepared_data[2::3])
    plt.title('Зависимость скорости сортировки от размера набора')
    plt.xlabel('Размер')
    plt.ylabel('Время (сек)')
    plt.legend(['Сортировка пузырьком', 'Шейкер-сортировка', 'Быстрая сортировка'])
    plt.savefig('compare_plot.png')


def get_time(func: Callable) -> Callable:
    @wraps(func)
    def wrapper(*args, **kwargs):
        st = time_calc()
        res = func(*args, **kwargs)
        ed = time_calc()
        print(st, ed)
        executed_times[f'{func.__name__}_{np.random.randint(1000, 10000)}'] = ed - st
        return res

    return wrapper


class MySorter:
    available_companies = ['S7', 'Turkish Flights', 'Emirates Airlines', 'TickleAir', 'Pin Airlines']
    current_object_len = 0

    class ArrayRecord:
        val_transform = {
            0: lambda val: val,
            1: lambda val: ord(val[0].lower()),
            2: lambda val: datetime(*list(map(int, val.split('-')))),
            3: lambda val: time(*list(map(int, val.split(':')))),
            4: lambda val: val,
        }

        def __init__(self, tuple_to_save: tuple):
            self.record = tuple_to_save

        def __lt__(self, other: 'MySorter.ArrayRecord') -> bool:
            transform_rules = MySorter.ArrayRecord.val_transform
            for ind, values in enumerate(zip(self.record, other.record)):
                if transform_rules[ind](values[0]) < transform_rules[ind](values[1]):
                    return True
                else:
                    if transform_rules[ind](values[0]) == transform_rules[ind](values[1]):
                        pass
                    else:
                        return False
            return False

        def __gt__(self, other: 'MySorter.ArrayRecord') -> bool:
            transform_rules = MySorter.ArrayRecord.val_transform
            for ind, values in enumerate(zip(self.record, other.record)):
                if transform_rules[ind](values[0]) > transform_rules[ind](values[1]):
                    return True
                else:
                    if transform_rules[ind](values[0]) == transform_rules[ind](values[1]):
                        pass
                    else:
                        return False
            return False

        def __le__(self, other: 'MySorter.ArrayRecord') -> bool:
            transform_rules = MySorter.ArrayRecord.val_transform
            for ind, values in enumerate(zip(self.record, other.record)):
                if transform_rules[ind](values[0]) <= transform_rules[ind](values[1]):
                    return True
                else:
                    return False

        def __ge__(self, other: 'MySorter.ArrayRecord') -> bool:
            transform_rules = MySorter.ArrayRecord.val_transform
            for ind, values in enumerate(zip(self.record, other.record)):
                if transform_rules[ind](values[0]) >= transform_rules[ind](values[1]):
                    return True
                else:
                    return False

        def __eq__(self, other: 'MySorter.ArrayRecord') -> bool:
            transform_rules = MySorter.ArrayRecord.val_transform
            if all([transform_rules[0](self.record[0]) == transform_rules[0](other.record[0]),
                    transform_rules[1](self.record[1]) == transform_rules[1](other.record[1]),
                    transform_rules[2](self.record[2]) == transform_rules[2](other.record[2]),
                    transform_rules[3](self.record[3]) == transform_rules[3](other.record[3]),
                    transform_rules[4](self.record[4]) == transform_rules[4](other.record[4])]):
                return True
            else:
                return False

        def __repr__(self) -> str:
            return str(self.record)

        def __str__(self) -> str:
            return str(self.record)

    def __init__(self, set_number: int):
        assert 0 <= set_number < 7

        self.dataset = list()
        current_data = pd.read_csv(f'data/dataset_{set_number}')
        data_as_array = np.array(
                [current_data['Номер рейса'],
                 current_data['Название авиакомпании'],
                 current_data['Дата прилета'],
                 current_data['Время  прилета по расписанию'],
                 current_data['Число пассажиров на борту']]
        )
        self.dataset = [MySorter.ArrayRecord(tuple(data_as_array[:, ind])) for ind in range(0, current_data.shape[0])]
        MySorter.current_object_len = len(self.dataset)

    @get_time
    def cocktail_sort(self) -> None:
        left = 0
        right = len(self.dataset) - 1

        while left <= right:
            for i in range(left, right, 1):
                if self.dataset[i] > self.dataset[i + 1]:
                    self.dataset[i], self.dataset[i + 1] = self.dataset[i + 1], self.dataset[i]
            right -= 1

            for i in range(right, left, -1):
                if self.dataset[i - 1] > self.dataset[i]:
                    self.dataset[i], self.dataset[i - 1] = self.dataset[i - 1], self.dataset[i]
            left += 1

    @get_time
    def bubble_sort(self) -> None:
        n = len(self.dataset)
        for i in range(n - 1):
            for j in range(0, n - i - 1):
                if self.dataset[j] > self.dataset[j + 1]:
                    self.dataset[j], self.dataset[j + 1] = self.dataset[j + 1], self.dataset[j]

    def quicksort_calc(self, data: list) -> list:
        if len(data) <= 1:
            return data
        else:
            pivot = choice(data)
        less_num = [n for n in data if n < pivot]
        eq_num = [pivot] * data.count(pivot)
        big_num = [n for n in data if n > pivot]
        return self.quicksort_calc(less_num) + eq_num + self.quicksort_calc(big_num)

    @get_time
    def quicksort(self) -> None:
        self.dataset = self.quicksort_calc(self.dataset)

    def reset(self) -> None:
        shuffle(self.dataset)

    def __repr__(self) -> str:
        return (5 * '{}\n').format(*list(map(lambda val: str(val).lstrip('(').rstrip(')'), self.dataset[:5]))) + \
               '...\n' + \
               (5 * '{}\n').format(*list(map(lambda val: str(val).lstrip('(').rstrip(')'), self.dataset[-6:])))

    def __str__(self) -> str:
        return (5 * '{}\n').format(*list(map(lambda val: str(val).lstrip('(').rstrip(')'), self.dataset[:5]))) + \
               '...\n' + \
               (5 * '{}\n').format(*list(map(lambda val: str(val).lstrip('(').rstrip(')'), self.dataset[-6:])))

    def __len__(self) -> int:
        return len(self.dataset)


if __name__ == '__main__':
    prepare_data()
    lab_1()
    print(executed_times)
