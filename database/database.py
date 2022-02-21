from sqlalchemy import create_engine, update
from sqlalchemy.orm import sessionmaker, aliased
from sqlalchemy.sql import func
from pyjarowinkler import distance
import Levenshtein as levenshtein

from tables import partner as par, partner_norm as par_norm, input_probability_table as prob, superposition_table as sup, input_table as inp

Session = sessionmaker()


class DBHandler:
    server = 'localhost'
    database = 'superpozicia'
    driver = 'ODBC Driver 17 for SQL Server'
    database_con = f'mssql://@{server}/{database}?driver={driver}'

    def __init__(self):
        self.engine = create_engine(self.database_con, echo=True)
        self.connect = self.connect()

    def connect(self):
        return self.engine.connect()

    def create_all_tables(self):
        par.PARTNER.create(self)
        par_norm.PARTNER_NORM.create(self)
        # prob.PROB.create(self)
        # inp.INPUT.create(self)
        sup.SUPERPOSITION.create(self)

    def get_max_string_len(self):
        result = self.connect.execute('select [dbo].[get_max_string_len]()')
        for row in result:
            return row[0]

    def get_max_CID(self):
        result = self.connect.execute('select max(cid)+1 from [dbo].[Partner] where priorita = 1')
        for row in result:
            return row[0]

    def get_min_CID(self):
        result = self.connect.execute('select min(cid)-1 from [dbo].[Superposition]')
        for row in result:
            return row[0]

    # def create_probability_table(self):
    #     table = self.cross_join("input_table")
    #     for row in table:
    #         priezvisko_dist = distance.get_jaro_distance(row[0].Priezvisko, row[1].Priezvisko, winkler=True, scaling=0.1)
    #         if priezvisko_dist > 0.8:
    #             meno_dist = distance.get_jaro_distance(row[0].Meno, row[1].Meno, winkler=True, scaling=0.1)
    #             pohlavie_dist = 1 if row[0].Pohlavie == row[1].Pohlavie else 0
    #             if row[0].Ulica is not None and row[1].Ulica is not None:
    #                 ulica_dist = distance.get_jaro_distance(row[0].Ulica, row[1].Ulica, winkler=True, scaling=0.1)
    #             else:
    #                 ulica_dist = 69
    #             if row[0].Mesto is not None and row[1].Mesto is not None:
    #                 mesto_dist = distance.get_jaro_distance(row[0].Mesto, row[1].Mesto, winkler=True, scaling=0.1)
    #             else:
    #                 mesto_dist = 69
    #             if row[0].Kraj is not None and row[1].Kraj is not None:
    #                 kraj_dist = distance.get_jaro_distance(row[0].Kraj, row[1].Kraj, winkler=True, scaling=0.1)
    #             else:
    #                 kraj_dist = 69
    #             if row[0].PSC is not None and row[1].PSC is not None:
    #                 psc_dist = 0 if levenshtein.distance(row[0].PSC, row[1].PSC) > 1 else levenshtein.distance(row[0].PSC, row[1].PSC)
    #             else:
    #                 psc_dist = 69
    #             domicil_dist = 0 if levenshtein.distance(row[0].Danovy_Domicil, row[1].Danovy_Domicil) > 1 else levenshtein.distance(row[0].PSC, row[1].PSC)
    #             self.insert_row_prob(row[0].CID, row[1].CID,
    #                             meno_dist, priezvisko_dist, pohlavie_dist,
    #                             0, 1, 0,
    #                             ulica_dist, mesto_dist, kraj_dist, psc_dist, domicil_dist)

    def cross_join(self, table_name):
        session = Session(bind=self.engine)
        if table_name.lower() == "partner":
            a = aliased(par.PARTNER)
            b = aliased(par.PARTNER)
            result = session.query(a, b)\
                .join(b, a.Datum_Narodenia == b.Datum_Narodenia)\
                .filter(a.CID != b.CID)\
                .all()
            return result
        return None

    def update_processed_bit(self, cid):
        local_session = Session(bind=self.engine)
        result = local_session.query(par.PARTNER).filter(par.PARTNER.CID == cid).one()
        result.Spracovane = 1
        local_session.commit()

    def insert_into_dbo_partner(self, cid, priority, first_name, last_name, sex, titles, dob,
                                street, city, region, psc, domicile, note):
        local_session = Session(bind=self.engine)
        if priority in range(1, 7):
            row = par.PARTNER(CID=cid, Spracovane=0, Priorita=priority,
                              Meno=first_name, Priezvisko=last_name, Pohlavie=sex,
                              Tituly=titles, Datum_Narodenia=dob,
                              Ulica=street, Mesto=city, Kraj=region, PSC=psc, Danovy_Domicil=domicile,
                              Poznamka=note)
            local_session.add(row)
            local_session.commit()

    def insert_into_dbo_partner_norm(self, cid, priority, first_name, last_name, sex, titles, dob,
                                street, city, region, psc, domicile, note):
        local_session = Session(bind=self.engine)
        if priority in range(1, 7):
            row = par_norm.PARTNER_NORM(CID=cid, Priorita=priority,
                              Meno=first_name, Priezvisko=last_name, Pohlavie=sex,
                              Tituly=titles, Datum_Narodenia=dob,
                              Ulica=street, Mesto=city, Kraj=region, PSC=psc, Danovy_Domicil=domicile,
                              Poznamka=note)
            local_session.add(row)
            local_session.commit()


    def insert_into_dbo_superposition(self, cid, first_name, last_name, sex, titles, dob,
                                      street, city, region, psc, domicile, identifiers, note):
        local_session = Session(bind=self.engine)
        row = sup.SUPERPOSITION(CID=cid,
                       Meno=first_name, Priezvisko=last_name, Pohlavie=sex,
                       Tituly=titles, Datum_Narodenia=dob,
                       Ulica=street, Mesto=city, Kraj=region, PSC=psc, Danovy_Domicil=domicile,
                       Identifikatory=identifiers,  Poznamka=note)
        local_session.add(row)
        local_session.commit()

    # def insert_row_inp(self, cid1, cid2, meno1, meno2, priezvisko1, priezvisko2, pohlavie1, pohlavie2, tituly1, tituly2,
    #                    mesto1, mesto2, kraj1, kraj2, psc1, psc2, danovy_domicil1,
    #                    danovy_domicil2):
    #     local_session = Session(bind=self.engine)
    #     row = inp.INPUT(CID1=cid1, CID2=cid2, Meno1=meno1, Meno2=meno2, Priezvisko1=priezvisko1, Priezvisko2=priezvisko2,
    #                     Pohlavie1=pohlavie1, Pohlavie2=pohlavie2, Tituly1=tituly1, Tituly2=tituly2,
    #                     Mesto1=mesto1, Mesto2=mesto2, Kraj1=kraj1, Kraj2=kraj2,
    #                     PSC1=psc1, PSC2=psc2, Danovy_Domicil1=danovy_domicil1, Danovy_Domicil2=danovy_domicil2)
    #     local_session.add(row)
    #     local_session.commit()
    #
    # def insert_row_prob(self, cid1, cid2, meno, priezvisko, pohlavie, tituly, datum_narodenia, rc,
    #                ulica, mesto, kraj, psc, danovy_domicil):
    #     local_session = Session(bind=self.engine)
    #     row = prob.PROB(CID1=cid1, CID2=cid2,
    #                    Meno=meno, Priezvisko=priezvisko, Pohlavie=pohlavie,
    #                    Tituly=tituly, Datum_Narodenia=datum_narodenia, RC=rc,
    #                    Ulica=ulica, Mesto=mesto, Kraj=kraj, PSC=psc, Danovy_Domicil=danovy_domicil)
    #     local_session.add(row)
    #     local_session.commit()