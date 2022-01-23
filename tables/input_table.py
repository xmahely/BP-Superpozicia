from sqlalchemy import Column, BigInteger, Integer, String, Date, text, Boolean, Identity
from sqlalchemy.orm import declarative_base

Base = declarative_base()


class INPUT(Base):
    __tablename__ = 'Input_Data'
    CID1 = Column(BigInteger(), primary_key=True)
    CID2 = Column(BigInteger(), primary_key=True)
    Meno1 = Column(String(50), nullable=False)
    Meno2 = Column(String(50), nullable=False)
    Priezvisko1 = Column(String(50), nullable=False)
    Priezvisko2 = Column(String(50), nullable=False)
    Pohlavie1 = Column(Boolean(), nullable=False)
    Pohlavie2 = Column(Boolean(), nullable=False)
    Tituly1 = Column(String(25), nullable=True)
    Tituly2 = Column(String(25), nullable=True)
    # RC1 = Column(String(10), nullable=True)
    # RC2 = Column(String(10), nullable=True)
    # Ulica1 = Column(String(80), nullable=True)
    # Ulica2 = Column(String(80), nullable=True)
    Mesto1 = Column(String(80), nullable=True)
    Mesto2 = Column(String(80), nullable=True)
    Kraj1 = Column(String(80), nullable=True)
    Kraj2 = Column(String(80), nullable=True)
    PSC1 = Column(String(80), nullable=True)
    PSC2 = Column(String(80), nullable=True)
    Danovy_Domicil1 = Column(String(10), nullable=False)
    Danovy_Domicil2 = Column(String(10), nullable=False)

    def __repr__(self):
        return f"<Klient " \
               f"CID1={self.CID1}, " \
               f"CID2={self.CID2}, " \
               f"Meno1={self.Meno1}, " \
               f"Meno2={self.Meno2}, " \
               f"Priezvisko1={self.Priezvisko1}.>," \
               f"Priezvisko2={self.Priezvisko2}.>," \
               f"Pohlavie1={self.Pohlavie1}," \
               f"Pohlavie2={self.Pohlavie2}," \
               f"Tituly1={self.Tituly1}," \
               f"Tituly2={self.Tituly2}," \
               f"Datum narodenia1={self.Datum_Narodenia1}," \
               f"Datum narodenia2={self.Datum_Narodenia2}," \
               f"Mesto1={self.Mesto1}," \
               f"Mesto2={self.Mesto2}," \
               f"Kraj1={self.Kraj1}," \
               f"Kraj2={self.Kraj2}," \
               f"PSC1={self.PSC1}," \
               f"PSC2={self.PSC2}," \
               f"Danovy domicil1={self.Danovy_Domicil1}," \
               f"Danovy domicil2={self.Danovy_Domicil2}," \


    def create(self):
        Base.metadata.create_all(bind=self.engine)

