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
    database = 'superpozicia_new'
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
                          a.Mesto == b.Mesto,
                          a.Kraj == b.Kraj,
                          a.PSC == b.PSC
                          # a.Danovy_Domicil == b.Danovy_Domicil
                      ),
                      and_(
                          a.Meno == b.Meno,
                          a.Priezvisko == b.Priezvisko,
                          a.Datum_Narodenia == b.Datum_Narodenia,
                          # a.Danovy_Domicil == b.Danovy_Domicil
                      ),
                      and_(
                          a.Meno == b.Meno,
                          a.Priezvisko == b.Priezvisko,
                          or_(
                              func.month(a.Datum_Narodenia) == func.day(b.Datum_Narodenia),
                              func.day(a.Datum_Narodenia) == func.month(b.Datum_Narodenia),
                          ),
                          func.year(a.Datum_Narodenia) == func.year(b.Datum_Narodenia),
                          a.Pohlavie == b.Pohlavie,
                          a.Mesto == b.Mesto,
                          a.Kraj == b.Kraj,
                          a.PSC == b.PSC
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

    def createTestProcedure(self):
        create = False
        condition = self.connect.execute("IF OBJECT_ID('testSuperpositionData') IS NULL SELECT 1\
                            IF OBJECT_ID('testSuperpositionData') IS NOT NULL SELECT 0")
        for row in condition:
            if row[0] == 0:
                create = False
            if row[0] == 1:
                create = True
        if create:
            procedure = text("CREATE PROCEDURE testSuperpositionData \
                AS\
                BEGIN\
                SET NOCOUNT ON;\
                DECLARE @result table(SLSP_Presnost FLOAT, NN_Presnost FLOAT, PSLSP_Presnost FLOAT, AM_SLSP_Presnost FLOAT, \
                SLSP_L_Presnost FLOAT, KOOP_Presnost FLOAT);\
                DECLARE @count_all_SLSP FLOAT, @count_wrong_SLSP FLOAT, @count_wrong2_SLSP FLOAT;\
                DECLARE @count_all_NN FLOAT, @count_wrong_NN FLOAT, @count_wrong2_NN FLOAT; \
                DECLARE @count_all_PSLSP FLOAT, @count_wrong_PSLSP FLOAT, @count_wrong2_PSLSP FLOAT;\
                DECLARE @count_all_AM_SLSP FLOAT, @count_wrong_AM_SLSP FLOAT, @count_wrong2_AM_SLSP FLOAT; \
                DECLARE @count_all_SLSP_L FLOAT, @count_wrong_SLSP_L FLOAT, @count_wrong2_SLSP_L FLOAT;\
                DECLARE @count_all_KOOP FLOAT, @count_wrong_KOOP FLOAT, @count_wrong2_KOOP FLOAT;\
                DROP TABLE IF EXISTS #tempTable; \
                select * \
                into #tempTable\
                from(\
                SELECT \
                a.Meno, a.Priezvisko, a.Datum_Narodenia, poznamka,\
                LEFT(trim(b.value), CHARINDEX(':', trim(b.value))-1) as indetifikator, \
                RIGHT(trim(b.value), LEN(trim(b.value))- CHARINDEX(':', trim(b.value)) - 1) as cid\
                from [dbo].[Superpozicia] a\
                cross apply STRING_SPLIT (Identifikatory, ';')b\
                union all\
                SELECT \
                a.Meno, a.Priezvisko, a.Datum_Narodenia, poznamka, 'SLSP' as indetifikator,cast(a.CID as varchar(max)) as cid\
                from [dbo].[Superpozicia] a where a.poznamka is null)a\
                DROP TABLE IF EXISTS #tempTableSLSP;\
                select Meno, Priezvisko, Datum_Narodenia, 'SLSP' as indetifikator, id as cid\
                into #tempTableSLSP\
                from  [dbo].[result_superposition] \
                cross apply OpenJson(replace(replace(slsp_data, '[', ''), ']', '')) WITH( \
                id VARCHAR(20) '$.id')\
                where slsp_data is not null \
                DROP TABLE IF EXISTS #tempTableNN;\
                select Meno, Priezvisko, Datum_Narodenia, 'NN' as indetifikator, id as cid\
                into #tempTableNN\
                from  [dbo].[result_superposition] \
                cross apply OpenJson(replace(replace(nn_data, '[', ''), ']', '')) WITH( \
                id VARCHAR(20) '$.id')\
                where nn_data is not null \
                DROP TABLE IF EXISTS #tempTablePSLSP;\
                select Meno, Priezvisko, Datum_Narodenia, 'PSLSP' as indetifikator, id as cid\
                into #tempTablePSLSP\
                from  [dbo].[result_superposition] \
                cross apply OpenJson(replace(replace(pslsp_data, '[', ''), ']', '')) WITH( \
                id VARCHAR(20) '$.id')\
                where pslsp_data is not null \
                DROP TABLE IF EXISTS #tempTableAM_SLSP;\
                select Meno, Priezvisko, Datum_Narodenia, 'AM_SLSP' as indetifikator, id as cid\
                into #tempTableAM_SLSP\
                from  [dbo].[result_superposition] \
                cross apply OpenJson(replace(replace(amslsp_data, '[', ''), ']', '')) WITH( \
                id VARCHAR(20) '$.id')\
                where amslsp_data is not null \
                DROP TABLE IF EXISTS #tempTableSLSP_L;\
                select Meno, Priezvisko, Datum_Narodenia, 'SLSP_L' as indetifikator, id as cid\
                into #tempTableSLSP_L\
                from  [dbo].[result_superposition] \
                cross apply OpenJson(replace(replace(slspleasing_data, '[', ''), ']', '')) WITH( \
                id VARCHAR(20) '$.id')\
                where slspleasing_data is not null \
                DROP TABLE IF EXISTS #tempTableKOOP;\
                select Meno, Priezvisko, Datum_Narodenia, 'KOOP' as indetifikator, id as cid\
                into #tempTableKOOP\
                from  [dbo].[result_superposition] \
                cross apply OpenJson(replace(replace(kooperativa_data, '[', ''), ']', '')) WITH( \
                id VARCHAR(20) '$.id')\
                where kooperativa_data is not null \
                select @count_all_SLSP = count(*) from #tempTable where indetifikator = 'SLSP'\
                select @count_wrong_SLSP = count(*) from(\
                select Meno, Priezvisko, Datum_Narodenia, indetifikator, cid from #tempTable where indetifikator = 'SLSP'\
                except select Meno, Priezvisko, Datum_Narodenia, indetifikator, cid from #tempTableSLSP) a \
                select @count_wrong2_SLSP = count(*) from(select Meno, Priezvisko, Datum_Narodenia, indetifikator, cid from #tempTableSLSP\
                except select Meno, Priezvisko, Datum_Narodenia, indetifikator, cid from #tempTable where indetifikator = 'SLSP')a\
                select @count_all_NN = count(*) from #tempTable where indetifikator = 'NN'\
                select @count_wrong_NN = count(*) from(\
                select Meno, Priezvisko, Datum_Narodenia, indetifikator, cid from #tempTable where indetifikator = 'NN'\
                except select Meno, Priezvisko, Datum_Narodenia, indetifikator, cid from #tempTableNN) a\
                select @count_wrong2_NN = count(*) from(select Meno, Priezvisko, Datum_Narodenia, indetifikator, cid from  #tempTableNN\
                except select Meno, Priezvisko, Datum_Narodenia, indetifikator, cid from #tempTable where indetifikator = 'NN')a\
                select @count_all_PSLSP = count(*) from #tempTable where indetifikator = 'PSLSP'\
                select @count_wrong_PSLSP = count(*) from(\
                select Meno, Priezvisko, Datum_Narodenia, indetifikator, cid from #tempTable where indetifikator = 'PSLSP'\
                except select Meno, Priezvisko, Datum_Narodenia, indetifikator, cid from #tempTablePSLSP) a\
                select @count_wrong2_PSLSP = count(*) from( select Meno, Priezvisko, Datum_Narodenia, indetifikator, cid from  #tempTablePSLSP\
                except select Meno, Priezvisko, Datum_Narodenia, indetifikator, cid from  #tempTable where indetifikator = 'PSLSP')a\
                select @count_all_AM_SLSP = count(*) from #tempTable where indetifikator = 'AM_SLSP'\
                select @count_wrong_AM_SLSP = count(*) from(\
                select Meno, Priezvisko, Datum_Narodenia, indetifikator, cid from #tempTable where indetifikator = 'AM_SLSP'\
                except select Meno, Priezvisko, Datum_Narodenia, indetifikator, cid from #tempTableAM_SLSP) a\
                select @count_wrong2_AM_SLSP = count(*) from( select Meno, Priezvisko, Datum_Narodenia, indetifikator, cid from #tempTableAM_SLSP\
                except select Meno, Priezvisko, Datum_Narodenia, indetifikator, cid from  #tempTable where indetifikator = 'AM_SLSP')a\
                select @count_all_SLSP_L = count(*) from #tempTable where indetifikator = 'SLSP_L'\
                select @count_wrong_SLSP_L = count(*) from(\
                select Meno, Priezvisko, Datum_Narodenia, indetifikator, cid from #tempTable where indetifikator = 'SLSP_L'\
                except select Meno, Priezvisko, Datum_Narodenia, indetifikator, cid from #tempTableSLSP_L) a\
                select @count_wrong2_SLSP_L = count(*) from( select Meno, Priezvisko, Datum_Narodenia, indetifikator, cid from  #tempTableSLSP_L\
                except select Meno, Priezvisko, Datum_Narodenia, indetifikator, cid from  #tempTable where indetifikator = 'SLSP_L')a\
                select @count_all_KOOP = count(*) from #tempTable where indetifikator = 'KOOP'\
                select @count_wrong_KOOP = count(*) from(\
                select Meno, Priezvisko, Datum_Narodenia, indetifikator, cid from #tempTable where indetifikator = 'KOOP'\
                except select Meno, Priezvisko, Datum_Narodenia, indetifikator, cid from #tempTableKOOP) a\
                select @count_wrong2_KOOP = count(*) from( select Meno, Priezvisko, Datum_Narodenia, indetifikator, cid from  #tempTableKOOP\
                except select Meno, Priezvisko, Datum_Narodenia, indetifikator, cid from  #tempTable where indetifikator = 'KOOP')a\
                INSERT INTO @result \
                (SLSP_Presnost, NN_Presnost, PSLSP_Presnost, AM_SLSP_Presnost, SLSP_L_Presnost, KOOP_Presnost)\
                VALUES (\
                (1 - (@count_wrong_SLSP+@count_wrong2_SLSP) / @count_all_SLSP) * 100,\
                (1 - (@count_wrong_NN+@count_wrong2_NN) / @count_all_NN) * 100,\
                (1 - (@count_wrong_PSLSP+@count_wrong2_PSLSP) / @count_all_PSLSP) * 100,\
                (1 - (@count_wrong_AM_SLSP+@count_wrong2_AM_SLSP) / @count_all_AM_SLSP) * 100,\
                (1 - (@count_wrong_SLSP_L+@count_wrong2_SLSP_L) / @count_all_SLSP_L) * 100,\
                (1 - (@count_wrong_KOOP+@count_wrong2_KOOP) / @count_all_KOOP) * 100\
                )\
                select * from @result\
                END")
            self.connect.execute(procedure)
