

from sqlalchemy import Column, BigInteger, Integer, String, Date, DateTime, Boolean
from sqlalchemy.sql import func
from sqlalchemy.orm import declarative_base
from sqlalchemy.dialects.mssql import BIT

Base = declarative_base()


class PARTNER(Base):
    __tablename__ = 'Partner'
    CID = Column(BigInteger(), primary_key=True)
    Spracovane = Column(BIT, nullable=True)
    Priorita = Column(Integer(), nullable=False)
    Meno = Column(String(50), nullable=False)
    Priezvisko = Column(String(50), nullable=False)
    Pohlavie = Column(Boolean(), nullable=False)
    Tituly = Column(String(25), nullable=True)
    Datum_Narodenia = Column(Date(), nullable=True)
    RC = Column(String(10), nullable=True)
    Ulica = Column(String(80), nullable=True)
    Mesto = Column(String(80), nullable=True)
    Kraj = Column(String(80), nullable=True)
    PSC = Column(String(80), nullable=True)
    Danovy_Domicil = Column(String(10), nullable=False)
    Stlpec2 = Column(String(10), nullable=True)
    Stlpec3 = Column(String(10), nullable=True)
    Poznamka = Column(String(250), nullable=True)
    Sprac_Dat = Column(DateTime(timezone=True), server_default=func.now())

    def __repr__(self):
        return f"<Klient " \
               f"CID={self.CID}, " \
               f"Spracovane={self.Spracovane}, " \
               f"Priorita={self.Priorita}," \
               f"Meno={self.Meno}, " \
               f"Priezvisko={self.Priezvisko}.>," \
               f"Pohlavie={self.Pohlavie}," \
               f"Tituly={self.Tituly}," \
               f"Datum narodenia={self.Datum_Narodenia}," \
               f"RC={self.RC}," \
               f"Ulica={self.Ulica}," \
               f"Mesto={self.Mesto}," \
               f"Kraj={self.Kraj}," \
               f"PSC={self.PSC}," \
               f"Danovy domicil={self.Danovy_Domicil}," \
               f"Stlpec2={self.Stlpec2}," \
               f"Stlpec3={self.Stlpec3}," \
               f"Poznamka={self.Poznamka}>"

    def create(self):
        Base.metadata.create_all(bind=self.engine)

