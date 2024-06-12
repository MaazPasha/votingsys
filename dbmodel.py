from sqlalchemy import Column, Integer, String, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class Voter(Base):
    __tablename__ = 'voters'

    id = Column(Integer, primary_key=True)
    name = Column(String)
    email = Column(String, unique=True)
    password = Column(String)
    created_at = Column(DateTime)

class Vote(Base):
    __tablename__ = 'votes'

    id = Column(Integer, primary_key=True)
    voter_id = Column(Integer, ForeignKey('voters.id'))
    candidate_id = Column(Integer, ForeignKey('candidates.id'))
    timestamp = Column(DateTime)

class Candidate(Base):
    __tablename__ = 'candidates'

    id = Column(Integer, primary_key=True)
    name = Column(String)
    party = Column(String)
    votes = relationship('Vote', backref='candidate')
