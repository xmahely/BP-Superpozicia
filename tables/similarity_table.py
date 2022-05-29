from sqlalchemy import Column, BigInteger, Integer, Float, String, Date, text, Boolean, Identity
from sqlalchemy.orm import declarative_base

Base = declarative_base()


class SIM(Base):
    __tablename__ = 'Potencialne_Duplicity'
    CID1 = Column(BigInteger(), primary_key=True)
    CID2 = Column(BigInteger(), primary_key=True)
    Meno = Column(Float(), nullable=False)
    Priezvisko = Column(Float(), nullable=False)
    Tituly = Column(Float(), nullable=True)
    Datum_Narodenia = Column(Float(), nullable=True)
    Mesto = Column(Float(), nullable=True)
    Kraj = Column(Float(), nullable=True)
    PSC = Column(Float(), nullable=True)
    Danovy_Domicil = Column(Float(), nullable=False)

    def __repr__(self):
        return f"<User CID1={self.CID1}, " \
               f"CID2 = {self.CID2}, " \
               f"Meno = {self.Meno}, "\
               f"Priezvisko = {self.Priezvisko}, " \
               f"Tituly = {self.Tituly}, " \
               f"Datum_Narodenia = {self.Datum_Narodenia}, " \
               f"Mesto = {self.Mesto}, " \
               f"Kraj = {self.Kraj}, " \
               f"PSC = {self.PSC}, " \
               f"Danovy_Domicil = {self.Danovy_Domicil}.>"

    def create(self):
        Base.metadata.create_all(bind=self.engine)

