import pandas as pd
import sqlalchemy
import normalizer as n
import glob


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


def insert_into_dbo(database,
                    company,
                    cid,
                    priority,
                    first_name,
                    last_name,
                    sex,
                    titles,
                    date_of_birth,
                    street,
                    city,
                    region,
                    psc,
                    domicile,
                    note):
    database.insert_into_dbo_partner(cid=cid, priority=priority, first_name=first_name, last_name=last_name,
                                     sex=sex, titles=titles, dob=date_of_birth, street=street, city=city,
                                     region=region, psc=psc, domicile=domicile, note=note)
    database.insert_into_dbo_partner_norm(cid=cid, priority=priority, first_name=n.normalizeWord(first_name),
                                          last_name=n.normalizeWord(last_name), sex=sex,
                                          titles=n.normalizeWord(titles), dob=date_of_birth,
                                          street=n.normalizeWord(street), city=n.normalizeWord(city),
                                          region=n.normalizeWord(region), psc=psc,
                                          domicile=n.normalizeWord(domicile), note=note)

    if company == 'SLSP':
        database.insert_into_dbo_superposition(cid=cid, first_name=first_name, last_name=last_name, sex=sex,
                                               titles=titles, dob=date_of_birth, street=street, city=city,
                                               region=region, psc=psc, domicile=domicile,
                                               identifiers=sqlalchemy.sql.null(), note=note)
        database.update_processed_bit(cid)


def insert_data(database, company):
    df = read(company)

    if company == 'SLSP':
        header = ['index', 'CID', 'Meno', 'Priezvisko', 'Tituly', 'Pohlavie', 'DatumNarodenia', 'PSC', 'Mesto', 'Kraj',
                  'DanDom', 'Stlpec2', 'Dom2']
        for row in df.itertuples():
            cid = row[1]
            priority = 1
            first_name = str(row[2]).strip()
            last_name = str(row[3]).strip()
            sex = n.normalizeSex(row[header.index('Pohlavie')])
            titles = sqlalchemy.sql.null() if pd.isna(row[4]) else str(row[4]).strip()
            date_of_birth, note = n.normalizeDate(row[header.index('DatumNarodenia')])
            street = sqlalchemy.sql.null()
            city = str(row[8]).strip()
            region = str(row[9]).strip()
            # psc = str(row[7]).strip()
            psc = ''.join(str(row[7]).split())
            domicile = str(row[10]).strip()
            insert_into_dbo(database, company, cid, priority, first_name, last_name, sex, titles,
                            date_of_birth, street, city, region, psc, domicile, note)
    if company == 'NN':
        header = ['index', 'CID', 'Meno', 'Priezvisko', 'Tituly', 'Pohlavie', 'DatumNarodenia', 'Adresa',
                  'DanDom', 'Stlpec2', 'Stlpec3']
        for row in df.itertuples():
            cid = row[1]
            priority = 2
            first_name = str(row[2]).strip()
            last_name = str(row[3]).strip()
            sex = n.normalizeSex(row[header.index('Pohlavie')])
            titles = sqlalchemy.sql.null() if pd.isna(row[4]) else str(row[4]).strip()
            date_of_birth, note = n.normalizeDate(row[header.index('DatumNarodenia')])
            address = split_address_evenly(row[header.index('Adresa')], ",")
            street = sqlalchemy.sql.null()
            city = address[0]
            region = address[1]
            # psc = str(address[2]).strip()
            psc = ''.join(str(address[2]).split())
            domicile = str(row[8]).strip()
            insert_into_dbo(database, company, cid, priority, first_name, last_name, sex, titles,
                            date_of_birth, street, city, region, psc, domicile, note)
    if company == 'PSLSP':
        header = ['index', 'CID', 'Meno', 'Priezvisko', 'Pohlavie', 'DatumNarodenia', 'PSC', 'DanDom', 'Adresa']
        for row in df.itertuples():
            cid = row[1]
            priority = 3
            first_name = str(row[2]).strip()
            last_name = str(row[3]).strip()
            sex = n.normalizeSex(row[header.index('Pohlavie')])
            titles = sqlalchemy.sql.null()
            date_of_birth, note = n.normalizeDate(row[header.index('DatumNarodenia')])
            address = split_address_evenly(row[header.index('Adresa')], ",")
            street = sqlalchemy.sql.null()
            city = address[1]
            region = address[0]
            # psc = str(row[6]).strip()
            psc = ''.join(str(row[6]).split())
            domicile = row[7]
            insert_into_dbo(database, company, cid, priority, first_name, last_name, sex, titles,
                            date_of_birth, street, city, region, psc, domicile, note)
    if company == 'AM_SLSP':
        header = ['index', 'CID', 'Meno/Priezvisko', 'Pohlavie', 'DatumNarodenia', 'Adresa', 'Stat', 'DanDom']
        for row in df.itertuples():
            cid = row[1]
            priority = 4
            first_name, last_name = split_by_first_left(row[header.index('Meno/Priezvisko')], " ")
            sex = n.normalizeSex(row[header.index('Pohlavie')])
            titles = sqlalchemy.sql.null()
            date_of_birth, note = n.normalizeDate(row[header.index('DatumNarodenia')])
            address = split_by_first_left(row[header.index('Adresa')], " ")
            street = sqlalchemy.sql.null()
            city = address[1]
            region = row[6]
            # psc = str(address[0]).strip()
            psc = ''.join(str(address[0]).split())
            domicile = row[7]
            insert_into_dbo(database, company, cid, priority, first_name, last_name, sex, titles,
                            date_of_birth, street, city, region, psc, domicile, note)
    if company == 'SLSP_L' or company == 'KOOP':
        header = ['index', 'CID', 'Meno', 'Priezvisko', 'Tituly', 'Pohlavie', 'DatumNarodenia', 'PSC', 'Mesto',
                  'Kraj', 'DanDom']
        priority = 5 if company == 'SLSP_L' else 6
        for row in df.itertuples():
            cid = row[1]
            first_name = str(row[2]).strip()
            last_name = str(row[3]).strip()
            sex = n.normalizeSex(row[header.index('Pohlavie')])
            titles = sqlalchemy.sql.null() if pd.isna(row[4]) else row[4]
            date_of_birth, note = n.normalizeDate(row[header.index('DatumNarodenia')])
            street = sqlalchemy.sql.null()
            city = str(row[8]).strip()
            region = str(row[9]).strip()
            # psc = str(row[7]).strip()
            psc = ''.join(str(row[7]).split())
            domicile = str(row[10]).strip()
            insert_into_dbo(database, company, cid, priority, first_name, last_name, sex, titles,
                            date_of_birth, street, city, region, psc, domicile, note)

def createInputTables(database):
    # # pre všetky csv, ktoré sú v priečinku /data vytvorí záznamy vo vstupnej tabuľke
    # # zároveň pre dáta SLSP platí, že sa prenesú do výstupnej tabuľky s prázdnym identifikátorom
    for file in glob.glob("data/*.csv", recursive=False):
        print(file)
        variant = file.replace('data\\', '').replace('.csv', '')
        insert_data(database, variant)
