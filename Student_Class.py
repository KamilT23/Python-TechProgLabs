from typing import Union, List, Dict
from collections import namedtuple
import datetime
from datetime import date
import os.path
import json
import csv

LAB_WORK_SESSION_KEYS = ("presence", "lab_work_n", "lab_work_mark", "lab_work_date")
STUDENT_KEYS = ("unique_id", "name", "surname", "group", "subgroup", "lab_works_sessions")

class LabWorkSession(namedtuple('LabWorkSession', 'presence, lab_work_number, lab_work_mark, lab_work_date')):
    """
    Информация о лабораторном занятии, которое могло или не могло быть посещено студентом
    """
    __slots__ = ()
    def __new__(cls, presence: bool, lab_work_number: int, lab_work_mark: int, lab_work_date: date):
        """
            param: presence: присутствие студента на л.р.(bool)
            param: lab_work_number: номер л.р.(int)
            param: lab_work_mark: оценка за л.р.(int)
            param: lab_work_date: дата л.р.(date)
        """
        if not LabWorkSession._validate_args(presence, lab_work_number, lab_work_mark, lab_work_date):
            raise ValueError(f"LabWorkSession ::"
                             f"incorrect args :\n"
                             f"presence       : {presence},\n"
                             f"lab_work_number: {lab_work_number},\n"
                             f"lab_work_mark  : {lab_work_mark},\n"
                             f"lab_work_date  : {lab_work_date}")

        return super(LabWorkSession, cls).__new__(cls, presence, lab_work_number, lab_work_mark, lab_work_date)

    @staticmethod
    def _validate_args(presence: bool, lab_work_number: int, lab_work_mark: int, lab_work_date: date) -> bool:
        """
            param: presence: присутствие студента на л.р.(bool)
            param: lab_work_number: номер л.р.(int)
            param: lab_work_mark: оценка за л.р.(int)
            param: lab_work_date: дата л.р.(date)
        """
        rule_1 = True if presence and lab_work_mark > -1 else False
        correct = isinstance(presence, bool) and isinstance(lab_work_number, int) \
                  and isinstance(lab_work_mark, int) and isinstance(lab_work_date, date)
        if correct and rule_1:
            return True
        else:
            return False

    def __str__(self) -> str:
        """
            Строковое представление LabWorkSession
            Пример:
            {
                    "presence":      1,
                    "lab_work_n":    4,
                    "lab_work_mark": 3,
                    "date":          "15:12:23"
            }
        """
        return f'\t\t{{\n' \
               f'\t\t\t"presence":      {1 if self.presence else 0},\n' \
               f'\t\t\t"lab_work_n":    {self.lab_work_number},\n' \
               f'\t\t\t"lab_work_mark": {self.lab_work_mark},\n' \
               f'\t\t\t"lab_work_date":          "{self.lab_work_date.strftime("%d:%m:%y")}"\n' \
               f'\t\t}}'
class Student:
    __slots__ = ('_unique_id', '_name', '_surname', '_group', '_subgroup', '_lab_work_sessions')

    def __init__(self, unique_id: int, name: str, surname: str, group: int, subgroup: int):
        """
            param: unique_id: уникальный идентификатор студента (int)
            param: name: имя студента (str)
            param: surname: фамилия студента (str)
            param: group: номер группы в которой студент обучается (int)
            param: subgroup: номер подгруппы (int)
        """
        self._unique_id = unique_id
        self._name = name
        self._surname = surname
        self._group = group
        self._subgroup = subgroup
        self._lab_work_sessions = []
        if not self._validate_args(unique_id, name, surname, group, subgroup):
            raise ValueError(f'Что-то пошло не так')

    @staticmethod
    def _validate_args(unique_id: int, name: str, surname: str, group: int, subgroup: int) -> bool:
        """
            param: unique_id: уникальный идентификатор студента (int)
            param: name: имя студента (str)
            param: surname: фамилия студента (str)
            param: group: номер группы в которой студент обучается (int)
            param: subgroup: номер подгруппы (int)
        """
        correct = isinstance(unique_id, int) and isinstance(name, str) and isinstance(surname, str) \
                  and isinstance(group, int) and isinstance(subgroup, int)

        rule1 = True if group == 6407 or group == 6408 or group == 6409 else False
        rule2 = True if subgroup == 1 or subgroup == 2 else False

        if correct and rule1 and rule2: return True
        else: return False

    def __str__(self) -> str:
        """
        Строковое представление Student
        Пример:
        {
                "unique_id":          26,
                "name":               "Щукарев",
                "surname":            "Даниил",
                "group":              6408,
                "subgroup":           2,
                "lab_works_sessions": [
                    {
                        "presence":      1,
                        "lab_work_n":    1,
                        "lab_work_mark": 4,
                        "date":          "15:9:23"
                    },
                    {
                        "presence":      1,
                        "lab_work_n":    2,
                        "lab_work_mark": 4,
                        "date":          "15:10:23"
                    },
                    {
                        "presence":      1,
                        "lab_work_n":    3,
                        "lab_work_mark": 4,
                        "date":          "15:11:23"
                    },
                    {
                        "presence":      1,
                        "lab_work_n":    4,
                        "lab_work_mark": 3,
                        "date":          "15:12:23"
                    }]
        }
        """
        sep = ',\n'
        # lab_sessions_str = ',\n'.join(str(session) for session in self.lab_work_sessions)

        return f'\t{{\n' \
               f'\t\t"unique_id":          {self._unique_id},\n' \
               f'\t\t"name":               "{self._name}",\n' \
               f'\t\t"surname":            "{self._surname}",\n' \
               f'\t\t"group":              {self._group},\n' \
               f'\t\t"subgroup":           {self._subgroup},\n' \
               f'\t\t"lab_works_sessions": [\n{sep.join(str(session) for session in self.lab_work_sessions)}]\n' \
               f'\t}}'

    @property
    def unique_id(self) -> int:
        """
        Метод доступа для unique_id
        """
        return self._unique_id

    @property
    def group(self) -> int:
        """
        Метод доступа для номера группы
        """
        return self._group

    @property
    def subgroup(self) -> int:
        """
        Метод доступа для номера подгруппы
        """
        return self._subgroup

    @property
    def name(self) -> str:
        """
        Метод доступа для имени студента
        """
        return self._name

    @property
    def surname(self) -> str:
        """
        Метод доступа для фамилии студента
        """
        return self._surname

    @unique_id.setter
    def unique_id(self, value: int) -> None:
        """
        Метод для измения значения id студента
        """
        assert isinstance(value, int) #утверждение, которое проверяет, является ли условие истинным. Если условие программы
        assert value > 0  #истино, то работа программы продолжается. В ином случае - вызывает исключение AssertError
        self._unique_id = value

    @name.setter
    def name(self, name: str) -> None:
        """
        Метод для изменения значения имени студента
        """
        assert isinstance(name, str)
        assert len(name) != 0
        self._name = name

    @surname.setter
    def surname(self, surname: str) -> None:
        """
        Метод для изменения значения фамилии студента
        """
        assert isinstance(surname, str)
        assert len(surname) != 0
        self._surname = surname

    @property
    def lab_work_sessions(self):
        """
        Метод доступа для списка лабораторных работ, которые студент посетил или не посетил.
        Использовать yield.
        """
        for lab in self._lab_work_sessions:
            yield lab

    def append_lab_work_session(self, session: LabWorkSession):
        """
        Метод для регистрации нового лабораторного занятия
        """
        if not isinstance(session, LabWorkSession):
            raise ValueError(f'Error append lab work session')
        self._lab_work_sessions.append(session)

    @lab_work_sessions.setter
    def lab_work_sessions(self, lab):
        self._lab_work_sessions = lab


def _load_lab_work_session(json_node) -> LabWorkSession:
    """
        Создание из под-дерева json файла экземпляра класса LabWorkSession.
        hint: чтобы прочитать дату из формата строки, указанного в json используйте
        date(*tuple(map(int, json_node['date'].split(':'))))
    """
    for key in LAB_WORK_SESSION_KEYS:
        if key not in json_node:
            raise KeyError(f"load_lab_work_session:: key \"{key}\" not present in json_node")

    return LabWorkSession(True if json_node['presence'] == 1 else False,
                          int(json_node['lab_work_n']),
                          int(json_node['lab_work_mark']),
                          datetime.datetime.strptime(json_node['lab_work_date'], '%d:%m:%y').date())

def _load_student(json_node) -> Student:
    """
        Создание из под-дерева json файла экземпляра класса Student.
        Если в процессе создания LabWorkSession у студента случается ошибка,
        создание самого студента ломаться не должно.
    """
    for key in STUDENT_KEYS:
        if key not in json_node:
            raise KeyError(f"_load_student:: key \"{key}\" not present in json_node ")
    student = Student(json_node['unique_id'],
                      json_node['name'],
                      json_node['surname'],
                      int(json_node['group']),
                      int(json_node['subgroup']))
    for session in json_node['lab_works_sessions']:
        try:
            student.append_lab_work_session(_load_lab_work_session(session))
        except ValueError as ver:
            print(ver)
            continue
        except KeyError as ker:
            print(ker)
            continue
        except Exception as exc:
            print(exc)
            continue
    return student


# csv header
#     0    |   1  |   2   |   3  |    4    |  5  |    6    |        7       |       8     |
# unique_id; name; surname; group; subgroup; date; presence; lab_work_number; lab_work_mark
UNIQUE_ID = 0
STUD_NAME = 1
STUD_SURNAME = 2
STUD_GROUP = 3
STUD_SUBGROUP = 4
LAB_WORK_DATE = 5
STUD_PRESENCE = 6
LAB_WORK_NUMBER = 7
LAB_WORK_MARK = 8

def load_students_csv(file_path: str) -> Union[List[Student], None]:
    # csv header
    #     0    |   1  |   2   |   3  |    4    |  5  |    6    |        7       |       8     |
    # unique_id; name; surname; group; subgroup; date; presence; lab_work_number; lab_work_mark
    assert isinstance(file_path, str)
    if not os.path.exists(file_path):
        return None

    students_raw: Dict[int, Student] = {}

    with open(file_path, 'r', encoding='utf-8') as input_file:
        csv_reader = csv.reader(input_file, delimiter=';')
        next(csv_reader)

        for line in csv_reader:
            try:
                unique_id = int(line[0])
                name = line[2]
                surname = line[1]
                group = int(line[3])
                subgroup = int(line[4])
                presence = True if int(line[6]) == 1 else False
                lab_work_number = int(line[7])
                lab_work_mark = int(line[8])
                lab_work_date = datetime.datetime.strptime(line[5], '%d:%m:%y').date()

                if unique_id not in students_raw:

                    students_raw[unique_id] = Student(unique_id, name, surname, group, subgroup)

                session = LabWorkSession(presence, lab_work_number, lab_work_mark, lab_work_date)
                students_raw[unique_id].append_lab_work_session(session)
            except Exception as ex:
                print(f"Ошибка при обработке строки: {ex}")
                continue
    return list(students_raw.values())

def load_students_json(file_path: str) -> Union[List[Student], None]:
    """
    Загрузка списка студентов из json файла.
    Ошибка создания экземпляра класса Student не должна приводить к поломке всего чтения.
    """
    assert isinstance(file_path, str)  # Путь к файлу должен быть строкой
    if not os.path.exists(file_path):  # и, желательно, существовать...
        print('Файла не существует')
        return None
    students = []
    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)
        for student in data['students']:
            students.append(_load_student(student))
    return students

# def save_students_json(file_path: str, students: List[Student]) -> bool:
#     """
#     Запись списка студентов в json файл
#     """
#     try:
#         with open(file_path, 'w', encoding='utf-8') as output_file:
#             students_data = []
#             for student in students:
#                 lab_works_sessions = []
#                 if student._lab_work_sessions is not None:
#                     for session in student._lab_work_sessions:
#                         session_dict = {
#                             "date": session.lab_work_date.strftime("%d:%m:%y"),
#                             "presence": 1 if session.presence else 0,
#                             "lab_work_n": session.lab_work_number,
#                             "lab_work_mark": session.lab_work_mark
#                         }
#                         lab_works_sessions.append(session_dict)
#
#                 student_dict = {
#                     "unique_id": student.unique_id,
#                     "name": student._name,
#                     "surname": student._surname,
#                     "group": student._group,
#                     "subgroup": student._subgroup,
#                     "lab_works_sessions": lab_works_sessions
#                 }
#                 students_data.append(student_dict)
#             json.dump(students_data, output_file, ensure_ascii=False, indent=4)
#             print('Запись в JSON файл прошла успешно')
#             return True
#
#     except Exception as exc:
#         print(f'Ошибка при записи в JSON файл: {exc}')
#         return False

def save_students_json(file_path: str, students: List[Student]) -> None:
    """
    Запись списка студентов в json файл
    """
    try:
        with open(file_path, 'w', encoding='utf-8') as output_file:
            output_file.write('{\n\t"students":[\n')
            output_file.write(',\n'.join(str(i) for i in students))
            output_file.write(']\n}')
            print('Запись в JSON файл прошла успешно')
            return None

    except Exception as exc:
        print(f'Ошибка при записи в JSON файл: {exc}')
        return None


def save_students_csv(file_path: str, students: List[Student]) -> bool:
    """
    Запись списка студентов в csv файл
    """
    try:
        with open(file_path, 'w', newline='', encoding='utf-8') as output_file:
            writer = csv.writer(output_file, delimiter=';')

            writer.writerow([
                'unique_id', 'name', 'surname', 'group', 'subgroup',
                'date', 'presence','lab_work_n', 'lab_work_mark'
            ])

            for student in students:
                if student._lab_work_sessions is not None:
                    for session in student._lab_work_sessions:
                        writer.writerow([
                            student._unique_id, student._name, student._surname,
                            student._group, student._subgroup, session.lab_work_date.strftime("%d:%m:%y"),
                            1 if session.presence else 0, session.lab_work_number, session.lab_work_mark
                        ])
            print(f'Запись в CSV файл прошла успешно')
            return True
    except Exception as exc:
        print(f'Ошибка при записи в CSV файл: {exc}')
        return False

if __name__ == '__main__':
    students = load_students_csv("students.csv")
    save_students_json("saved_students.json", students)
    save_students_csv("saved_students.csv", students)

    students3 = load_students_json("saved_students.json")
    print('\n'.join(str(v) for v in students3))