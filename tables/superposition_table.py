from sqlalchemy import Column, BigInteger, Integer, String, Date, text, Boolean, Identity
from sqlalchemy.orm import declarative_base

Base = declarative_base()


class SUPERPOSITION(Base):
    __tablename__ = 'Superposition'
    CID = Column(BigInteger(), primary_key=True)
    Meno = Column(String(50), nullable=False)
    Priezvisko = Column(String(50), nullable=False)
    Pohlavie = Column(Boolean(), nullable=False)
    Tituly = Column(String(25), nullable=True)
    Datum_Narodenia = Column(Date(), nullable=True)
    Ulica = Column(String(80), nullable=True)
    Mesto = Column(String(80), nullable=True)
    Kraj = Column(String(80), nullable=True)
    PSC = Column(String(80), nullable=True)
    Danovy_Domicil = Column(String(10), nullable=False)
    Identifikatory = Column(String(100), nullable=True)
    Poznamka = Column(String(250), nullable=True)

    def __repr__(self):
        return f"<Klient " \
               f"CID={self.CID}, " \
               f"Meno={self.Meno}, " \
               f"Priezvisko={self.Priezvisko}.>," \
               f"Pohlavie={self.Pohlavie}," \
               f"Tituly={self.Tituly}," \
               f"Datum narodenia={self.Datum_Narodenia}," \
               f"Ulica={self.Ulica}," \
               f"Mesto={self.Mesto}," \
               f"Kraj={self.Kraj}," \
               f"Danovy domicil={self.Danovy_Domicil}," \
               f"Identifikatory={self.Identifikatory}," \
               f"Poznamka={self.Poznamka}>"

    def create(self):
        Base.metadata.create_all(bind=self.engine)

