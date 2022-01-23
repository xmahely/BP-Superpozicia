from sqlalchemy import Column, BigInteger, Integer, Float, String, Date, text, Boolean, Identity
from sqlalchemy.orm import declarative_base

Base = declarative_base()


class PROB(Base):
    __tablename__ = 'Input_Probability_Table'
    CID1 = Column(BigInteger(), primary_key=True)
    CID2 = Column(BigInteger(), primary_key=True)
    Meno = Column(Float(), nullable=False)
    Priezvisko = Column(Float(), nullable=False)
    Pohlavie = Column(Float(), nullable=False)
    Tituly = Column(Float(), nullable=True)
    Datum_Narodenia = Column(Float(), nullable=True)
    RC = Column(Float(), nullable=True)
    Ulica = Column(Float(), nullable=True)
    Mesto = Column(Float(), nullable=True)
    Kraj = Column(Float(), nullable=True)
    PSC = Column(Float(), nullable=True)
    Danovy_Domicil = Column(Float(), nullable=False)

    def __repr__(self):
        return f"<User CID={self.CID}, meno={self.Meno}, priezvisko={self.Priezvisko}.>"

    def create(self):
        Base.metadata.create_all(bind=self.engine)

