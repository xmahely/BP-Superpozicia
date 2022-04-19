from database import database as db
import input_data
import output_data


def main():
    # # pripojí sa k lokálnej databáze
    database = db.DBHandler()

    # # ak treba, vytvorí všetky zadefinované tabuľky
    database.create_all_tables()

    # # pre všetky csv, ktoré sú v priečinku /data vytvorí záznamy vo vstupnej tabuľke
    # # zároveň pre dáta SLSP platí, že sa prenesú do výstupnej tabuľky s prázdnym identifikátorom
    input_data.createInputTables(database)

    # # vykoná deduplikáciu
    output_data.dedupe(database)


if __name__ == "__main__":
    main()








