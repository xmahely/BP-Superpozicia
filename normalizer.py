import datetime as dt

import sqlalchemy
import unicodedata


def unicodeToAscii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) in ['Ll', 'Nd'])


def normalizeWord(s):
    if s is not None and s is not sqlalchemy.sql.null():
        return unicodeToAscii(s.lower())


def wordToAsciiValueList(s):
    return [ord(c) for c in s]


def asciiValueListToWord(s):
    return ''.join(chr(c) for c in s)


def validateDate(date, format):
    res = 0
    try:
        if bool(dt.datetime.strptime(date, format)):
            res = 0
    except ValueError as e:
        res = -1
        if str(e) == 'day is out of range for month':
            res = -2
    return res


def findFormat(date):
    date_new = None
    poznamka = None
    if validateDate(date, "%Y-%m-%d") == 0:
        date_new = dt.datetime.strptime(date, "%Y-%m-%d")
    elif validateDate(date, "%Y-%m-%d") == -2:
        poznamka = "day is out of range for month: " + date
    elif validateDate(date, "%Y-%d-%m") == 0:
        date_new = dt.datetime.strptime(date, "%Y-%d-%m")
    elif validateDate(date, "%Y-%d-%m") == -2:
        poznamka = "day is out of range for month: " + date
    elif validateDate(date, "%d-%m-%Y") == 0:
        date_new = dt.datetime.strptime(date, "%d-%m-%Y")
    elif validateDate(date, "%d-%m-%Y") == -2:
        poznamka = "day is out of range for month: " + date
    elif validateDate(date, "%m-%d-%Y") == 0:
        date_new = dt.datetime.strptime(date, "%m-%d-%Y")
    elif validateDate(date, "%m-%d-%Y") == -2:
        poznamka = "day is out of range for month: " + date
    elif validateDate(date, "%d.%m.%Y") == 0:
        date_new = dt.datetime.strptime(date, "%d.%m.%Y")
    elif validateDate(date, "%d.%m.%Y") == -2:
        poznamka = "day is out of range for month: " + date
    elif validateDate(date, "%m.%d.%Y") == 0:
        date_new = dt.datetime.strptime(date, "%m.%d.%Y")
    elif validateDate(date, "%m.%d.%Y") == -2:
        poznamka = "day is out of range for month: " + date
    elif validateDate(date, "%d. %m. %Y") == 0:
        date_new = dt.datetime.strptime(date, "%d. %m. %Y")
    elif validateDate(date, "%d. %m. %Y") == -2:
        poznamka = "day is out of range for month: " + date
    elif validateDate(date, "%m. %d. %Y") == 0:
        date_new = dt.datetime.strptime(date, "%m. %d. %Y")
    elif validateDate(date, "%m. %d. %Y") == -2:
        poznamka = "day is out of range for month: " + date
    else:
        poznamka = "day is out of range: " + date
    return date_new, poznamka


def normalizeDate(date_old):
    date_new, poznamka = findFormat(date_old)
    return date_new, poznamka


def normalizeSex(sex):
    if sex.lower() in ['f', 'female', 'ž', 'z', 'žena', 'zena', 'frau']:
        return 0
    if sex.lower() in ['m', 'male', 'muž', 'muz', 'mann']:
        return 1
    return -1
