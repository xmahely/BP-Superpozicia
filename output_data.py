import jaro
from rapidfuzz.distance import Levenshtein as levenshtein
import torch
import MLP


def createSimilarityTable(database, priority):
    # potential_duplicates = database.get_potential_duplicates_old(priority)
    database.delete_similarity_table()
    potential_duplicates = database.get_potential_duplicates(priority)
    for row in potential_duplicates:
        name_similarity = jaro.jaro_winkler_metric(row[2].Meno, row[1].Meno)
        last_name_similarity = jaro.jaro_winkler_metric(row[2].Priezvisko, row[1].Priezvisko)
        titles_similarity = jaro.jaro_winkler_metric('' if row[2].Tituly is None else row[2].Tituly,
                                                     '' if row[1].Tituly is None else row[1].Tituly)
        dob_similarity = levenshtein.distance(str(row[2].Datum_Narodenia), str(row[1].Datum_Narodenia))
        dob_similarity = 1 - (dob_similarity / max(len(str(row[2].Datum_Narodenia)), len(str(row[1].Datum_Narodenia))))

        city_similarity = jaro.jaro_winkler_metric(row[2].Mesto, row[1].Mesto)
        region_similarity = jaro.jaro_winkler_metric(row[2].Kraj, row[1].Kraj)
        psc_similarity = levenshtein.distance(row[2].PSC, row[1].PSC)
        domicile_similarity = levenshtein.distance(row[2].Danovy_Domicil, row[1].Danovy_Domicil)
        psc_similarity = (max(len(row[1].PSC), len(row[2].PSC)) - psc_similarity) / max(len(row[1].PSC), len(row[2].PSC))
        domicile_similarity = abs((len(row[1].Danovy_Domicil) - domicile_similarity)) / len(row[1].Danovy_Domicil)
        database.insert_into_dbo_similarity_table(row[1].CID, row[2].CID, name_similarity, last_name_similarity,
                                                  titles_similarity, dob_similarity,
                                                  city_similarity, region_similarity, psc_similarity,
                                                  domicile_similarity)


def dedupe(database):
    companies = ['', 'SLSP', 'NN', 'PSLSP', 'AM_SLSP', 'SLSP_L', 'KOOP']

    for i in range(2, 7):
        createSimilarityTable(database, i)
        model_dict = torch.load('model.pt')
        model = MLP.MLP()
        model.load_state_dict(model_dict)
        table = database.get_table('similarity_table')
        input = []
        cids = []
        for row in table:
            input.append([row.Meno, row.Priezvisko, row.Tituly, row.Datum_Narodenia, row.Mesto, row.Kraj,
                          row.PSC, row.Danovy_Domicil])
            cids.append([row.CID1, row.CID2])

        tensor = torch.tensor(input, dtype=torch.float64)
        tensor = tensor.float()
        test = model(tensor)

        notin_list = []
        for a, data in enumerate(test):
            # print(a)
            # print(cids[a][0], ", ", cids[a][1])
            # print(data.item())
            if data.item() > 0.85:
                notin_list.append(cids[a][1])
                database.update_superposition(cids[a][0], companies[i], cids[a][1])

        new_clients_table = database.get_clients_not_in_list(notin_list, i)

        for row in new_clients_table:
            database.insert_superpositon(companies[i], row.CID)

        database.delete_similarity_table()

