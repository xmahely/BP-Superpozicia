from sqlalchemy import exc
from sqlalchemy import create_engine, update, or_, and_, not_, text
from sqlalchemy import func
from sqlalchemy.orm import sessionmaker, aliased
from sqlalchemy_utils import database_exists, create_database
from tables import partner as par, partner_norm as par_norm, similarity_table as sim, \
    superposition_table as sup

Session = sessionmaker()


class DBHandler:
    server = 'localhost'
    database = 'superpozicia'
    driver = 'SQL Server Native Client 11.0'
    database_con = f'mssql://@{server}/{database}?driver={driver}&trusted_connection=yes'

    def __init__(self):
        self.engine = create_engine(self.database_con, echo=True)
        try:
            if not database_exists(self.engine.url):
                create_database(self.engine.url)
        except Exception:
            create_database(self.engine.url)

        self.connect = self.connect()
        self.create_all_tables()

    def connect(self):
        return self.engine.connect()

    def create_all_tables(self):
        par.PARTNER.create(self)
        par_norm.PARTNER_NORM.create(self)
        sim.SIM.create(self)
        sup.SUPERPOSITION.create(self)

    def execute(self, sql_string):
        return self.connect.execute(sql_string)

    def get_max_CID(self):
        session = Session(bind=self.engine)
        result = session.query(func.max(sup.SUPERPOSITION.CID)).scalar()
        session.close()
        return result + 1

    def get_min_CID(self):
        result = self.connect.execute('select min(cid)-1 from [dbo].[Superposition]')
        for row in result:
            return row[0]

    def get_potential_duplicates_old(self, priority):
        session = Session(bind=self.engine)
        s = aliased(sup.SUPERPOSITION)
        a = aliased(par_norm.PARTNER_NORM)
        b = aliased(par_norm.PARTNER_NORM)
        result = session.query(s, a, b) \
            .join(a, a.CID == s.CID) \
            .join(b,
                  and_(
                      a.Datum_Narodenia == b.Datum_Narodenia,
                      a.Pohlavie == b.Pohlavie,
                      b.Priorita == priority,
                  )) \
            .filter(a.CID != b.CID) \
            .all()
        session.close()
        return result

    def get_potential_duplicates(self, priority):
        session = Session(bind=self.engine)
        s = aliased(sup.SUPERPOSITION)
        a = aliased(par_norm.PARTNER_NORM)
        b = aliased(par_norm.PARTNER_NORM)
        c = aliased(par.PARTNER)
        result = session.query(s, a, b, c) \
            .join(a, a.CID == s.CID) \
            .join(b,
                  or_(
                      and_(
                          a.Datum_Narodenia == b.Datum_Narodenia,
                          a.Pohlavie == b.Pohlavie,
                      ),
                      and_(
                          a.Meno == b.Meno,
                          a.Priezvisko == b.Priezvisko,
                          func.abs((func.datediff(text('year'), a.Datum_Narodenia, b.Datum_Narodenia))) == 1000,
                          a.Pohlavie == b.Pohlavie,
                      ),
                      and_(
                          a.Meno == b.Meno,
                          a.Priezvisko == b.Priezvisko,
                          func.year(a.Datum_Narodenia) == func.year(b.Datum_Narodenia),
                          a.Pohlavie == b.Pohlavie,
                          a.Danovy_Domicil == b.Danovy_Domicil
                      ),
                      and_(
                          a.Meno == b.Meno,
                          a.Priezvisko == b.Priezvisko,
                          a.Datum_Narodenia == b.Datum_Narodenia,
                          a.Danovy_Domicil == b.Danovy_Domicil
                      ),
                      and_(
                          or_(
                              func.month(a.Datum_Narodenia) == func.day(b.Datum_Narodenia),
                              func.day(a.Datum_Narodenia) == func.month(b.Datum_Narodenia),
                          ),
                          func.year(a.Datum_Narodenia) == func.year(b.Datum_Narodenia),
                          a.Pohlavie == b.Pohlavie,
                      )
                  )) \
            .join(c, b.CID == c.CID) \
            .filter(a.CID != b.CID, c.Spracovane == 0, b.Priorita == priority) \
            .all()
        session.close()
        return result

    def get_clients_not_in_list(self, list, priority):
        local_session = Session(bind=self.engine)
        result = local_session.query(par.PARTNER) \
            .filter( \
            and_(
                not_(par.PARTNER.CID.in_(list)),
                par.PARTNER.Priorita == priority
            )) \
            .all()
        local_session.close()
        return result

    def get_table(self, table_name):
        table = None
        if table_name == 'similarity_table':
            table = sim.SIM
        if table is not None:
            local_session = Session(bind=self.engine)
            result = local_session.query(table).all()
            local_session.close()
            return result
        return None

    def update_superposition(self, cid_slsp, company, cid_original):
        local_session = Session(bind=self.engine)
        self.update_processed_bit(cid_original)
        result = local_session.query(sup.SUPERPOSITION).filter(sup.SUPERPOSITION.CID == cid_slsp).one()
        tmp = result.Identifikatory + "; " if result.Identifikatory is not None else ''
        result.Identifikatory = tmp + company + ": " + str(cid_original)
        try:
            local_session.commit()
        except exc.IntegrityError:
            print('Snaha o insert do dbo.superpozicia. Vyhodnotená duplicita: \n', cid_slsp)
        local_session.close()

    def insert_superpositon(self, company, cid_original):
        local_session = Session(bind=self.engine)
        self.update_processed_bit(cid_original)
        result = local_session.query(par.PARTNER).filter(par.PARTNER.CID == cid_original).one()
        cid_slsp = self.get_max_CID()
        print("CID", cid_slsp)
        identifier = company + ": " + str(cid_original)
        try:
            self.insert_into_dbo_superposition(cid_slsp, result.Meno, result.Priezvisko, result.Pohlavie,
                                               result.Tituly, result.Datum_Narodenia, result.Ulica, result.Mesto,
                                               result.Kraj, result.PSC, result.Danovy_Domicil, identifier,
                                               'Novy klient')
        except exc.IntegrityError:
            print('Snaha o insert do dbo.superpozicia. Vyhodnotená duplicita: \n', cid_original)
        local_session.close()

    def update_processed_bit(self, cid):
        local_session = Session(bind=self.engine)
        result = local_session.query(par.PARTNER).filter(par.PARTNER.CID == cid).one()
        result.Spracovane = 1
        local_session.commit()
        local_session.close()

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
            try:
                local_session.commit()
            except exc.IntegrityError:
                print('Snaha o insert do dbo.partner. Vyhodnotená duplicita: \n', cid, priority, first_name, last_name,
                      sex, titles, dob, street, city, region, psc, domicile, note)
            local_session.close()

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
            try:
                local_session.commit()
            except exc.IntegrityError:
                print('Snaha o insert do dbo.partner_norm. Vyhodnotená duplicita: \n', cid, priority, first_name,
                      last_name,
                      sex, titles, dob, street, city, region, psc, domicile, note)
            local_session.close()

    def insert_into_dbo_superposition(self, cid, first_name, last_name, sex, titles, dob,
                                      street, city, region, psc, domicile, identifiers, note):
        local_session = Session(bind=self.engine)
        row = sup.SUPERPOSITION(CID=cid,
                                Meno=first_name, Priezvisko=last_name, Pohlavie=sex,
                                Tituly=titles, Datum_Narodenia=dob,
                                Ulica=street, Mesto=city, Kraj=region, PSC=psc, Danovy_Domicil=domicile,
                                Identifikatory=identifiers, Poznamka=note)
        local_session.add(row)
        try:
            local_session.commit()
        except exc.IntegrityError:
            print('Snaha o insert do dbo.superposition. Vyhodnotná duplicita: \n', cid, first_name, last_name,
                  sex, titles, dob, street, city, region, psc, domicile, note)
        local_session.close()

    def insert_into_dbo_similarity_table(self, cid1, cid2, first_name, last_name, titles, dob, city, region, psc,
                                         domicile):
        local_session = Session(bind=self.engine)
        row = sim.SIM(CID1=cid1, CID2=cid2, Meno=first_name, Priezvisko=last_name,
                      Tituly=titles, Datum_Narodenia=dob, Mesto=city, Kraj=region, PSC=psc, Danovy_Domicil=domicile)
        local_session.add(row)
        local_session.commit()
        local_session.close()

    def delete_similarity_table(self):
        local_session = Session(bind=self.engine)
        local_session.query(sim.SIM).delete()
        local_session.commit()
        local_session.close()
