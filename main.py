
import glob
import os

from database import database as db
import input_data as data


def main():


    # #pripojí sa k lokálnej databáze
    database = db.DBHandler()

    # # ak treba, vytvorí všetky zadefinované tabuľky
    database.create_all_tables()

    # # pre všetky csv, ktoré sú v priečinku /data vytvorí záznamy vo vstupnej tabuľke
    # # zároveň pre dáta SLSP platí, že sa prenesú do výstupnej tabuľky s prázdnym identifikátorom
    for file in glob.glob(os.path.join('data', '*.csv')):
        variant = file.replace('data\\', '').replace('.csv', '')
        data.insert_data(database, variant)

    # data.insert_data(database, 'SLSP')

    # test.MAX_LEN = database.get_max_string_len()
    # data.CID = database.get_max_CID()

    # table = database.cross_join("Partner")
    # test.temp(database, table)


if __name__ == "__main__":
    main()








