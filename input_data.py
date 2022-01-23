import pandas as pd
import datetime as dt
import sqlalchemy
import glob, os

def read(name):
    df = pd.read_csv('data/' + name + '.csv', sep=';', header=0)
    return df


def validate_date(date, format):
    res = 0
    try:
        if bool(dt.datetime.strptime(date, format)):
            res = 0
    except ValueError as e:
        res = -1
        if str(e) == 'day is out of range for month':
            res = -2
    return res


def find_format(date):
    date_new = None
    poznamka = None
    if validate_date(date, "%Y-%m-%d") == 0:
        date_new = dt.datetime.strptime(date, "%Y-%m-%d")
    elif validate_date(date, "%Y-%m-%d") == -2:
        poznamka = "day is out of range for month: "+date
    elif validate_date(date, "%Y-%d-%m") == 0:
        date_new = dt.datetime.strptime(date, "%Y-%d-%m")
    elif validate_date(date, "%Y-%d-%m") == -2:
        poznamka = "day is out of range for month: "+date
    elif validate_date(date, "%d-%m-%Y") == 0:
        date_new = dt.datetime.strptime(date, "%d-%m-%Y")
    elif validate_date(date, "%d-%m-%Y") == -2:
        poznamka = "day is out of range for month: "+date
    elif validate_date(date, "%m-%d-%Y") == 0:
        date_new = dt.datetime.strptime(date, "%m-%d-%Y")
    elif validate_date(date, "%m-%d-%Y") == -2:
        poznamka = "day is out of range for month: " + date
    elif validate_date(date, "%d.%m.%Y") == 0:
        date_new = dt.datetime.strptime(date, "%d.%m.%Y")
    elif validate_date(date, "%d.%m.%Y") == -2:
        poznamka = "day is out of range for month: "+date
    elif validate_date(date, "%m.%d.%Y") == 0:
        date_new = dt.datetime.strptime(date, "%m.%d.%Y")
    elif validate_date(date, "%m.%d.%Y") == -2:
        poznamka = "day is out of range for month: "+date
    elif validate_date(date, "%d. %m. %Y") == 0:
        date_new = dt.datetime.strptime(date, "%d. %m. %Y")
    elif validate_date(date, "%d. %m. %Y") == -2:
        poznamka = "day is out of range for month: "+date
    elif validate_date(date, "%m. %d. %Y") == 0:
        date_new = dt.datetime.strptime(date, "%m. %d. %Y")
    elif validate_date(date, "%m. %d. %Y") == -2:
        poznamka = "day is out of range for month: "+date
    else:
        poznamka = "day is out of range wtf: "+date
    return date_new, poznamka


def normalize_date(date_old):
    date_new, poznamka = find_format(date_old)
    return date_new, poznamka


def normalize_sex(sex):
    if sex.lower() in ['f', 'female', 'ž', 'z', 'žena', 'zena', 'frau']:
        return 0
    if sex.lower() in ['m', 'male', 'muž', 'muz', 'mann']:
        return 1
    return -1


def split_address_evenly(address, delimeter):
    address = address.strip()
    if address[0] == ",":
        address = address[1:]
        address = address.split(delimeter)
    else:
        address = address.split(delimeter)
    return address


def split_by_first_left(string, delimeter):
    split_string = string.split(delimeter, 1)
    return split_string[0], split_string[1]


def split_by_first_right(string, delimeter):
    split_string = string.rsplit(delimeter, 1)
    return split_string[0], split_string[1]


def find_max_len():
    max = 1
    for file in glob.glob(os.path.join('data', '*.csv')):
        df = pd.read_csv(file, sep=';', header=0)
        for c in df:
            if df[c].dtype == 'object' and not (df[c].isnull().values.any()):
                if (df[c].map(len).max()) > max:
                    max = df[c].map(len).max()
    return max


def insert_data(database, company):
    df = read(company)
    print(max)
    if company == 'SLSP':
        header = ['index', 'CID', 'Meno', 'Priezvisko', 'Tituly', 'Pohlavie', 'DatumNarodenia', 'PSC', 'Mesto', 'Kraj',
                  'DanDom', 'Stlpec2', 'Dom2']
        for row in df.itertuples():
            date, poznamka = normalize_date(row[header.index('DatumNarodenia')])
            sex = normalize_sex(row[header.index('Pohlavie')])
            tituly = sqlalchemy.sql.null() if pd.isna(row[4]) else row[4]
            database.insert_row(cid=row[1], priority=1, meno=row[2], priezvisko=row[3], pohlavie=sex,
                                tituly=tituly, datum_narodenia=date,
                                rc=sqlalchemy.sql.null(), ulica=sqlalchemy.sql.null(), mesto=row[8], kraj=row[9],
                                psc=row[7], danovy_domicil=row[10], stlpec2=row[11], stlpec3=row[12], poznamka=poznamka)
            database.insert_row_sup(cid=row[1], meno=row[2], priezvisko=row[3], pohlavie=sex,
                                tituly=tituly, datum_narodenia=date,
                                rc=sqlalchemy.sql.null(), ulica=sqlalchemy.sql.null(), mesto=row[8], kraj=row[9],
                                psc=row[7], danovy_domicil=row[10], identifikatory=sqlalchemy.sql.null(), poznamka=poznamka)
    if company == 'NN':
        header = ['index', 'CID', 'Meno', 'Priezvisko', 'Tituly', 'Pohlavie', 'DatumNarodenia', 'Adresa',
                  'DanDom', 'Stlpec2', 'Stlpec3']
        for row in df.itertuples():
            date, poznamka = normalize_date(row[header.index('DatumNarodenia')])
            sex = normalize_sex(row[header.index('Pohlavie')])
            address = split_address_evenly(row[header.index('Adresa')], ",")
            tituly = sqlalchemy.sql.null() if pd.isna(row[4]) else row[4]
            database.insert_row(cid=row[1], priority=2, meno=row[2], priezvisko=row[3], pohlavie=sex,
                                tituly=tituly,
                                datum_narodenia=date, rc=sqlalchemy.sql.null(), ulica=sqlalchemy.sql.null(),
                                mesto=address[0], kraj=address[1], psc=address[2], danovy_domicil=row[8], stlpec2=row[9],
                                stlpec3=row[10], poznamka=poznamka)
    if company == 'PSLSP':
        header = ['index', 'CID', 'Meno', 'Priezvisko', 'Pohlavie', 'DatumNarodenia', 'PSC', 'DanDom', 'Adresa']
        for row in df.itertuples():
            date, poznamka = normalize_date(row[header.index('DatumNarodenia')])
            sex = normalize_sex(row[header.index('Pohlavie')])
            address = split_address_evenly(row[header.index('Adresa')], ",")
            database.insert_row(cid=row[1], priority=3, meno=row[2], priezvisko=row[3], pohlavie=sex,
                                tituly=sqlalchemy.sql.null(),
                                datum_narodenia=date, rc=sqlalchemy.sql.null(), ulica=sqlalchemy.sql.null(),
                                mesto=address[1], kraj=address[0], psc=row[6], danovy_domicil=row[7],
                                stlpec2=sqlalchemy.sql.null(), stlpec3=sqlalchemy.sql.null(), poznamka=poznamka)

    if company == 'AM_SLSP':
        header = ['index', 'CID', 'Meno/Priezvisko', 'Pohlavie', 'DatumNarodenia', 'Adresa', 'Stat', 'DanDom']
        for row in df.itertuples():
            company, surname = split_by_first_right(row[header.index('Meno/Priezvisko')], " ")
            date, poznamka = normalize_date(row[header.index('DatumNarodenia')])
            sex = normalize_sex(row[header.index('Pohlavie')])
            address = split_by_first_left(row[header.index('Adresa')], " ")
            print("som tu idem insertovat")
            database.insert_row(cid=row[1], priority=4, meno=company, priezvisko=surname, pohlavie=sex,
                                tituly=sqlalchemy.sql.null(),
                                datum_narodenia=date, rc=sqlalchemy.sql.null(), ulica=sqlalchemy.sql.null(),
                                mesto=address[1], kraj=row[6], psc=address[0], danovy_domicil=row[7],
                                stlpec2=sqlalchemy.sql.null(), stlpec3=sqlalchemy.sql.null(), poznamka=poznamka)
    if company == 'SLSP_L' or company == 'KOOP':
        header = ['index', 'CID', 'Meno', 'Priezvisko', 'Tituly', 'Pohlavie', 'DatumNarodenia', 'PSC', 'Mesto',
                  'Kraj', 'DanDom']
        priority = 5 if company == 'SLSP_L' else 6
        for row in df.itertuples():
            date, poznamka = normalize_date(row[header.index('DatumNarodenia')])
            sex = normalize_sex(row[header.index('Pohlavie')])
            tituly = sqlalchemy.sql.null() if pd.isna(row[4]) else row[4]
            database.insert_row(cid=row[1], priority=priority, meno=row[2], priezvisko=row[3], pohlavie=sex,
                                tituly=tituly,
                                datum_narodenia=date,
                                rc=sqlalchemy.sql.null(), ulica=sqlalchemy.sql.null(), mesto=row[8], kraj=row[9],
                                psc=row[7], danovy_domicil=row[10], stlpec2=sqlalchemy.sql.null(),
                                stlpec3=sqlalchemy.sql.null(), poznamka=poznamka)