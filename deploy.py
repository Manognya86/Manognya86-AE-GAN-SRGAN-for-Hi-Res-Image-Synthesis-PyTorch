from flask import Flask, render_template, request, jsonify, redirect, url_for, send_from_directory
from flask_jwt_extended import JWTManager, jwt_required, get_jwt_identity, unset_jwt_cookies, get_jwt, verify_jwt_in_request
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash
import os
import logging
from datetime import datetime, timedelta
import shutil
from database import Session, User, ImageRecord, init_db
from auth import configure_auth
from inference_api import AEGANInference
from database import Session, User, ImageRecord, Feedback, Deployment, ModelEvaluation
# Initialize Flask app
app = Flask(__name__, static_folder='static', template_folder='templates')

from flask_socketio import SocketIO
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

socketio = SocketIO(app, cors_allowed_origins="*")
limiter = Limiter(key_func=get_remote_address)

# Application Configuration
app.config.update({
    'SECRET_KEY': os.getenv('SECRET_KEY', 'super-secret-key-1234'),
    'JWT_SECRET_KEY': os.getenv('JWT_SECRET_KEY', 'jwt-super-secret-5678'),
    'JWT_TOKEN_LOCATION': ['cookies', 'headers'],
    'JWT_ACCESS_COOKIE_NAME': 'access_token',
    'JWT_COOKIE_CSRF_PROTECT': False,
    'JWT_ACCESS_TOKEN_EXPIRES': timedelta(hours=2),
    'JWT_COOKIE_SAMESITE': 'Lax',
    'UPLOAD_FOLDER': os.path.join('static', 'uploads'),
    'PROCESSED_FOLDER': os.path.join('static', 'processed'),
    'MODEL_DIR': os.path.join('models', 'latest'),
    'ALLOWED_EXTENSIONS': {'png', 'jpg', 'jpeg', 'gif'},
    'SQLALCHEMY_DATABASE_URI': os.getenv('DATABASE_URL', 'sqlite:///aegan.db'),
    'SQLALCHEMY_TRACK_MODIFICATIONS': False
})

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

jwt = configure_auth(app)

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['PROCESSED_FOLDER'], exist_ok=True)
os.makedirs(app.config['MODEL_DIR'], exist_ok=True)

def initialize_app():
    try:
        init_db()
        session = Session()
        admin_username = os.getenv('DEFAULT_ADMIN_USER', 'admin')
        admin_password = os.getenv('DEFAULT_ADMIN_PASSWORD', 'admin123')
        if not session.query(User).filter_by(username=admin_username).first():
            admin = User(
                username=admin_username,
                password=generate_password_hash(admin_password),
                role='admin'
            )
            session.add(admin)
            session.commit()
            logger.info(f"Created admin user: {admin_username}")
    except Exception as e:
        logger.error(f"Initialization error: {str(e)}")
        raise
    finally:
        logger.info(f"Starting with DB: {app.config['SQLALCHEMY_DATABASE_URI']}")
        session.close()

def allowed_file(filename):
    return '.' in filename and            filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/api/recent-images', methods=['GET'])
@jwt_required()
def get_recent_images():
    """Get user's recent images"""
    try:
        current_user = get_jwt_identity()
        if not current_user:
            return jsonify({"error": "Unauthorized"}), 401

        with Session() as session:
            recent_images = session.query(ImageRecord).filter_by(
                user_id=int(current_user)
            ).order_by(ImageRecord.created_at.desc()).limit(5).all()
            return jsonify({
                "images": [{
                    "id": img.id,
                    "original_path": img.original_path,
                    "enhanced_path": img.enhanced_path,
                    "created_at": img.created_at.isoformat()
                } for img in recent_images]
            })
    except Exception as e:
        logger.error(f"Recent images error: {str(e)}")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    from routes import app  
    initialize_app()  
    app.run(host='0.0.0.0', port=5000, debug=True)