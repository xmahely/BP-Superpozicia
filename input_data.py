import pandas as pd
import sqlalchemy
import sqlalchemy.exc as exc
import normalizer as n

CID_sequencer = -1


def read(name):
    df = pd.read_csv('data/' + name + '.csv', sep=';', header=0)
    return df


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


def concat_string(string1, string2, string3):
    if string1:
        return string1 + "; " + string2 + string3
    return string2 + string3


def find_max_len(lst):
    return max(lst, key=len)


def insert_data(database, company):
    global CID_sequencer
    df = read(company)
    if company == 'SLSP':
        header = ['index', 'CID', 'Meno', 'Priezvisko', 'Tituly', 'Pohlavie', 'DatumNarodenia', 'PSC', 'Mesto', 'Kraj',
                  'DanDom', 'Stlpec2', 'Dom2']
        for row in df.itertuples():
            date, poznamka = n.normalizeDate(row[header.index('DatumNarodenia')])
            sex = n.normalizeSex(row[header.index('Pohlavie')])
            tituly = sqlalchemy.sql.null() if pd.isna(row[4]) else row[4]

            try:
                database.insert_row(cid=row[1], priorita=1, meno=row[2], priezvisko=row[3], pohlavie=sex,
                                    tituly=tituly, datum_narodenia=date,
                                    rc=sqlalchemy.sql.null(), ulica=sqlalchemy.sql.null(), mesto=row[8], kraj=row[9],
                                    psc=row[7], danovy_domicil=row[10], stlpec2=row[11], stlpec3=row[12],
                                    poznamka=poznamka)
                database.insert_row_sup(cid=row[1], meno=row[2], priezvisko=row[3], pohlavie=sex,
                                        tituly=tituly, datum_narodenia=date,
                                        rc=sqlalchemy.sql.null(), ulica=sqlalchemy.sql.null(), mesto=row[8],
                                        kraj=row[9],
                                        psc=row[7], danovy_domicil=row[10], identifikatory=sqlalchemy.sql.null(),
                                        poznamka=poznamka)
            except exc.IntegrityError:
                poznamka = concat_string(poznamka, "Duplicitne CID:", str(row[1]))
                CID_sequencer = database.get_min_CID() if database.get_min_CID() < 0 else CID_sequencer
                database.insert_row(cid=CID_sequencer, priorita=1, meno=row[2], priezvisko=row[3], pohlavie=sex,
                                    tituly=tituly, datum_narodenia=date,
                                    rc=sqlalchemy.sql.null(), ulica=sqlalchemy.sql.null(), mesto=row[8], kraj=row[9],
                                    psc=row[7], danovy_domicil=row[10], stlpec2=row[11], stlpec3=row[12],
                                    poznamka=poznamka)
                database.insert_row_sup(cid=CID_sequencer, meno=row[2], priezvisko=row[3], pohlavie=sex,
                                        tituly=tituly, datum_narodenia=date,
                                        rc=sqlalchemy.sql.null(), ulica=sqlalchemy.sql.null(), mesto=row[8],
                                        kraj=row[9],
                                        psc=row[7], danovy_domicil=row[10], identifikatory=sqlalchemy.sql.null(),
                                        poznamka=poznamka)

            database.update_processed_bit(row[1])

    if company == 'NN':
        header = ['index', 'CID', 'Meno', 'Priezvisko', 'Tituly', 'Pohlavie', 'DatumNarodenia', 'Adresa',
                  'DanDom', 'Stlpec2', 'Stlpec3']
        for row in df.itertuples():
            date, poznamka = n.normalizeDate(row[header.index('DatumNarodenia')])
            sex = n.normalizeSex(row[header.index('Pohlavie')])
            address = split_address_evenly(row[header.index('Adresa')], ",")
            tituly = sqlalchemy.sql.null() if pd.isna(row[4]) else row[4]
            try:
                database.insert_row(cid=row[1], priorita=2, meno=row[2], priezvisko=row[3], pohlavie=sex,
                                    tituly=tituly, datum_narodenia=date, rc=sqlalchemy.sql.null(),
                                    ulica=sqlalchemy.sql.null(), mesto=address[0], kraj=address[1], psc=address[2],
                                    danovy_domicil=row[8], stlpec2=row[9], stlpec3=row[10], poznamka=poznamka)
            except exc.IntegrityError:
                poznamka = concat_string(poznamka, "Duplicitne CID:", str(row[1]))
                CID_sequencer = database.get_min_CID() if database.get_min_CID() < 0 else CID_sequencer
                database.insert_row(cid=CID_sequencer, priorita=2, meno=row[2], priezvisko=row[3], pohlavie=sex,
                                    tituly=tituly, datum_narodenia=date, rc=sqlalchemy.sql.null(),
                                    ulica=sqlalchemy.sql.null(), mesto=address[0], kraj=address[1], psc=address[2],
                                    danovy_domicil=row[8], stlpec2=row[9], stlpec3=row[10], poznamka=poznamka)
    if company == 'PSLSP':
        header = ['index', 'CID', 'Meno', 'Priezvisko', 'Pohlavie', 'DatumNarodenia', 'PSC', 'DanDom', 'Adresa']
        for row in df.itertuples():
            date, poznamka = n.normalizeDate(row[header.index('DatumNarodenia')])
            sex = n.normalizeSex(row[header.index('Pohlavie')])
            address = split_address_evenly(row[header.index('Adresa')], ",")
            try:
                database.insert_row(cid=row[1], priorita=3, meno=row[2], priezvisko=row[3], pohlavie=sex,
                                    tituly=sqlalchemy.sql.null(),
                                    datum_narodenia=date, rc=sqlalchemy.sql.null(), ulica=sqlalchemy.sql.null(),
                                    mesto=address[1], kraj=address[0], psc=row[6], danovy_domicil=row[7],
                                    stlpec2=sqlalchemy.sql.null(), stlpec3=sqlalchemy.sql.null(), poznamka=poznamka)
            except exc.IntegrityError:
                poznamka = concat_string(poznamka, "Duplicitne CID:", str(row[1]))
                CID_sequencer = database.get_min_CID() if database.get_min_CID() < 0 else CID_sequencer
                database.insert_row(cid=CID_sequencer, priorita=3, meno=row[2], priezvisko=row[3], pohlavie=sex,
                                    tituly=sqlalchemy.sql.null(),
                                    datum_narodenia=date, rc=sqlalchemy.sql.null(), ulica=sqlalchemy.sql.null(),
                                    mesto=address[1], kraj=address[0], psc=row[6], danovy_domicil=row[7],
                                    stlpec2=sqlalchemy.sql.null(), stlpec3=sqlalchemy.sql.null(), poznamka=poznamka)
    if company == 'AM_SLSP':
        header = ['index', 'CID', 'Meno/Priezvisko', 'Pohlavie', 'DatumNarodenia', 'Adresa', 'Stat', 'DanDom']
        for row in df.itertuples():
            company, surname = split_by_first_right(row[header.index('Meno/Priezvisko')], " ")
            date, poznamka = n.normalizeDate(row[header.index('DatumNarodenia')])
            sex = n.normalizeSex(row[header.index('Pohlavie')])
            address = split_by_first_left(row[header.index('Adresa')], " ")
            try:
                database.insert_row(cid=row[1], priorita=4, meno=company, priezvisko=surname, pohlavie=sex,
                                    tituly=sqlalchemy.sql.null(),
                                    datum_narodenia=date, rc=sqlalchemy.sql.null(), ulica=sqlalchemy.sql.null(),
                                    mesto=address[1], kraj=row[6], psc=address[0], danovy_domicil=row[7],
                                    stlpec2=sqlalchemy.sql.null(), stlpec3=sqlalchemy.sql.null(), poznamka=poznamka)
            except exc.IntegrityError:
                poznamka = concat_string(poznamka, "Duplicitne CID:", str(row[1]))
                CID_sequencer = database.get_min_CID() if database.get_min_CID() < 0 else CID_sequencer
                database.insert_row(cid=CID_sequencer, priorita=4, meno=company, priezvisko=surname, pohlavie=sex,
                                    tituly=sqlalchemy.sql.null(),
                                    datum_narodenia=date, rc=sqlalchemy.sql.null(), ulica=sqlalchemy.sql.null(),
                                    mesto=address[1], kraj=row[6], psc=address[0], danovy_domicil=row[7],
                                    stlpec2=sqlalchemy.sql.null(), stlpec3=sqlalchemy.sql.null(), poznamka=poznamka)
    if company == 'SLSP_L' or company == 'KOOP':
        header = ['index', 'CID', 'Meno', 'Priezvisko', 'Tituly', 'Pohlavie', 'DatumNarodenia', 'PSC', 'Mesto',
                  'Kraj', 'DanDom']
        priorita = 5 if company == 'SLSP_L' else 6
        for row in df.itertuples():
            date, poznamka = n.normalizeDate(row[header.index('DatumNarodenia')])
            sex = n.normalizeSex(row[header.index('Pohlavie')])
            tituly = sqlalchemy.sql.null() if pd.isna(row[4]) else row[4]
            try:
                database.insert_row(cid=row[1], priorita=priorita, meno=row[2], priezvisko=row[3], pohlavie=sex,
                                    tituly=tituly,
                                    datum_narodenia=date,
                                    rc=sqlalchemy.sql.null(), ulica=sqlalchemy.sql.null(), mesto=row[8], kraj=row[9],
                                    psc=row[7], danovy_domicil=row[10], stlpec2=sqlalchemy.sql.null(),
                                    stlpec3=sqlalchemy.sql.null(), poznamka=poznamka)
            except exc.IntegrityError:
                poznamka = concat_string(poznamka, "Duplicitne CID:", str(row[1]))
                CID_sequencer = database.get_min_CID() if database.get_min_CID() < 0 else CID_sequencer
                database.insert_row(cid=CID_sequencer, priorita=priorita, meno=row[2], priezvisko=row[3], pohlavie=sex,
                                    tituly=tituly,
                                    datum_narodenia=date,
                                    rc=sqlalchemy.sql.null(), ulica=sqlalchemy.sql.null(), mesto=row[8], kraj=row[9],
                                    psc=row[7], danovy_domicil=row[10], stlpec2=sqlalchemy.sql.null(),
                                    stlpec3=sqlalchemy.sql.null(), poznamka=poznamka)
