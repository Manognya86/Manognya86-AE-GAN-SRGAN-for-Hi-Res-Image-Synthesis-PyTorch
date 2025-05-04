# database.py
import os
import logging
from datetime import datetime
from contextlib import contextmanager
from sqlalchemy import create_engine, Column, Integer, String, Text, Boolean, DateTime, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, scoped_session, relationship
from werkzeug.security import generate_password_hash, check_password_hash
from sqlalchemy import Column, Integer, String, DateTime, Text, ForeignKey, Float


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

Base = declarative_base()

DATABASE_URL = os.getenv('DATABASE_URL', 'sqlite:///aegan.db')
logger.info(f"Using database at: {DATABASE_URL}")


class User(Base):
    __tablename__ = 'users'
    id = Column(Integer, primary_key=True)
    username = Column(String(50), unique=True, nullable=False)
    password = Column(String(255), nullable=False)
    role = Column(String(20), default='user')
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_login = Column(DateTime)
    
    images = relationship('ImageRecord', backref='user')
    feedbacks = relationship('Feedback', backref='user')
    deployments = relationship('Deployment', backref='user')

    def verify_password(self, password):
        return check_password_hash(self.password, password)

    def __repr__(self):
        return f"<User(id={self.id}, username='{self.username}', role='{self.role}')>"

class Feedback(Base):
    __tablename__ = 'feedback'
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    rating = Column(Integer, nullable=False)
    comment = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<Feedback(id={self.id}, user_id={self.user_id}, rating={self.rating})>"
    

class ImageRecord(Base):
    __tablename__ = 'image_records'
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    original_path = Column(String(255), nullable=False)
    enhanced_path = Column(String(255), nullable=False)
    processing_time = Column(Integer)
    created_at = Column(DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<ImageRecord(id={self.id}, user_id={self.user_id})>"

class ModelEvaluation(Base):
    __tablename__ = 'model_evaluations'
    id = Column(Integer, primary_key=True)
    model_version = Column(String(50))
    psnr = Column(Float)
    ssim = Column(Float)
    evaluated_at = Column(DateTime, default=datetime.utcnow)

class Deployment(Base):
    __tablename__ = 'deployments'
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    model_version = Column(String(50), nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    status = Column(String(20), default='pending')

    def __repr__(self):
        return f"<Deployment(id={self.id}, version='{self.model_version}', status='{self.status}')>"

# Database configuration
DATABASE_URL = os.getenv('DATABASE_URL', 'sqlite:///aegan.db')
engine = create_engine(DATABASE_URL, pool_pre_ping=True, pool_recycle=3600)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Session = scoped_session(SessionLocal)

@contextmanager
def get_db():
    """Database session context manager"""
    db = Session()
    try:
        yield db
        db.commit()
    except Exception as e:
        db.rollback()
        logger.error(f"Database error: {str(e)}")
        raise
    finally:
        db.close()

def init_db():
    """Initialize database tables and create admin user"""
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables created/verified")
        
        with get_db() as db:
            admin_username = os.getenv('DEFAULT_ADMIN_USER', 'admin')
            admin_password = os.getenv('DEFAULT_ADMIN_PASSWORD', 'admin123')
            
            admin = db.query(User).filter_by(username=admin_username).first()
            if not admin:
                new_admin = User(
                    username=admin_username,
                    password=generate_password_hash(admin_password),
                    role='admin'
                )
                db.add(new_admin)
                db.commit()
                logger.info(f"Created admin user: {admin_username}")
                
    except Exception as e:
        logger.error(f"Database initialization failed: {str(e)}")
        raise

def create_user(db, username: str, password: str, role: str = 'user'):
    """Create a new user"""
    if db.query(User).filter(User.username == username).first():
        raise ValueError("Username already exists")
    
    new_user = User(
        username=username,
        password=generate_password_hash(password),
        role=role
    )
    db.add(new_user)
    return new_user

def get_user_by_username(db, username: str):
    """Get user by username"""
    return db.query(User).filter(User.username == username).first()

if __name__ == "__main__":
    init_db()
    logger.info("Database setup completed successfully")