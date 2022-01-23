import glob, os
from database import database as db
import pandas as pd
import test
import input_data as data


def main():
    # #pripojí sa k lokálnej databáze
    database = db.DBHandler()

    # # vytvorí všetky potrebné tabuľky
    database.create_all_tables()

    # # pre všetky csv, ktoré sú v priečinku data vytvorí záznamy v hlavnej tabuľke
    # # naleje všetky slsp údaje do výstupnej tabuľky
    for file in glob.glob(os.path.join('data', '*.csv')):
        variant = file.replace('data\\', '').replace('.csv', '')
        data.insert_data(database, variant)
    test.MAX_LEN = data.find_max_len()

    # table = database.cross_join("Partner")
    # test.temp(database, table)

if __name__ == "__main__":
    main()




