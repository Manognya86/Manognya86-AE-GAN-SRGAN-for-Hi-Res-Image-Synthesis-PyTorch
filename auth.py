
from flask import request, jsonify
from flask_jwt_extended import (
    JWTManager, 
    create_access_token,
    jwt_required,
    get_jwt_identity,
    verify_jwt_in_request,
    unset_jwt_cookies,
    get_jwt
)
from werkzeug.security import generate_password_hash, check_password_hash
import os
from functools import wraps
from database import Session, User
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def configure_auth(app):
    app.config['JWT_SECRET_KEY'] = os.getenv('JWT_SECRET_KEY', 'super-secret-key-5678')
    app.config['JWT_ACCESS_TOKEN_EXPIRES'] = timedelta(hours=2)
    app.config['JWT_TOKEN_LOCATION'] = ['cookies']
    app.config['JWT_COOKIE_SECURE'] = False
    app.config['JWT_COOKIE_CSRF_PROTECT'] = False
    app.config['JWT_COOKIE_SAMESITE'] = 'Lax'
    app.config['JWT_ACCESS_COOKIE_NAME'] = 'access_token'

    jwt = JWTManager(app)

    @jwt.user_lookup_loader
    def user_lookup_callback(_jwt_header, jwt_data):
        identity = jwt_data['sub']
        session = Session()
        try:
            return session.query(User).filter_by(id=int(identity)).first()
        except Exception as e:
            logger.error(f"User lookup failed: {str(e)}")
            return None
        finally:
            session.close()

    def role_required(roles):
        def wrapper(fn):
            @wraps(fn)
            def decorator(*args, **kwargs):
                verify_jwt_in_request()
                claims = get_jwt()
                if claims.get('role') not in roles:
                    return jsonify({"error": "Unauthorized"}), 403
                return fn(*args, **kwargs)
            return decorator
        return wrapper

    @app.route('/api/login', methods=['POST'])
    def login():
        data = request.get_json()
        if not data or 'username' not in data or 'password' not in data:
            logger.warning("Login attempt with missing credentials")
            return jsonify({"error": "Missing credentials"}), 400

        session = Session()
        try:
            user = session.query(User).filter_by(username=data['username']).first()
            if user and check_password_hash(user.password, data['password']):
                user.last_login = datetime.utcnow()
                session.commit()

                access_token = create_access_token(
                    identity=str(user.id),
                    additional_claims={
                        'username': user.username,
                        'role': user.role
                    }
                )

                response = jsonify({
                    "message": "Login successful",
                    "redirect": "/dashboard"
                })

                response.set_cookie(
                    'access_token',
                    value=access_token,
                    max_age=int(timedelta(hours=2).total_seconds()),
                    path='/',
                    httponly=True,
                    secure=app.config['JWT_COOKIE_SECURE'],
                    samesite=app.config['JWT_COOKIE_SAMESITE']
                )
                logger.info(f"Successful login for user: {user.username}")
                return response

            logger.warning(f"Failed login attempt for username: {data['username']}")
            return jsonify({"error": "Invalid credentials"}), 401
        except Exception as e:
            session.rollback()
            logger.error(f"Login error: {str(e)}")
            return jsonify({"error": "Internal server error"}), 500
        finally:
            session.close()

    @app.route('/api/validate', methods=['GET'])
    @jwt_required()
    def validate_token():
        user_id = get_jwt_identity()
        return jsonify(logged_in_as=user_id), 200

    return jwt