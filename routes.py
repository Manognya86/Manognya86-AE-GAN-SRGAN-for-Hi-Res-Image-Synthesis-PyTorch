# routes.py
import os
import logging
import threading
from functools import wraps
from datetime import datetime
import shutil
import torch
import torchvision.transforms as transforms
import torchvision.datasets as dset
from torch.utils.data import DataLoader
from flask_jwt_extended import JWTManager, jwt_required, get_jwt_identity, unset_jwt_cookies, get_jwt, verify_jwt_in_request
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash
# from database import Session, User, ImageRecord, Feedback, Deployment, ModelEvaluation


from flask import (
    render_template, jsonify, request, send_from_directory, 
    redirect, url_for, current_app, make_response
)
from flask_jwt_extended import (
    jwt_required, 
    get_jwt_identity, 
    unset_jwt_cookies, 
    # verify_jwt_in_request_optional  # Make sure this import is included
)
from flask_socketio import SocketIO, emit

from database import Session, User, ImageRecord, Feedback, Deployment, ModelEvaluation
from train import AEGANTrainer
from inference_api import AEGANInference
from deploy import app, socketio, limiter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize rate limiter
limiter.init_app(app)

# Helper functions
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in current_app.config['ALLOWED_EXTENSIONS']

def calculate_psnr(original, enhanced):
    """Calculate PSNR between original and enhanced images"""
    mse = torch.mean((original - enhanced) ** 2)
    return 20 * torch.log10(1.0 / torch.sqrt(mse)).item()

def calculate_ssim(original, enhanced):
    """Calculate SSIM between original and enhanced images"""
    # Simplified SSIM calculation - in practice use a proper implementation
    c1 = (0.01 * 1) ** 2
    c2 = (0.03 * 1) ** 2
    
    mu_x = torch.mean(original)
    mu_y = torch.mean(enhanced)
    
    sigma_x = torch.std(original)
    sigma_y = torch.std(enhanced)
    sigma_xy = torch.mean((original - mu_x) * (enhanced - mu_y))
    
    numerator = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
    denominator = (mu_x**2 + mu_y**2 + c1) * (sigma_x**2 + sigma_y**2 + c2)
    
    return (numerator / denominator).item()

def role_required(roles):
    """Role-Based Access Control Decorator"""
    def wrapper(fn):
        @wraps(fn)
        @jwt_required()
        def decorator(*args, **kwargs):
            current_user = get_jwt_identity()
            if current_user.get('role') not in roles:
                logger.warning(f"Unauthorized access attempt by {current_user['username']}")
                return jsonify({"error": "Unauthorized"}), 403
            return fn(*args, **kwargs)
        return decorator
    return wrapper

def create_new_version(model_dir):
    """Create a new version of the trained model"""
    try:
        version = datetime.now().strftime("%Y%m%d-%H%M%S")
        new_version_dir = os.path.join("models", version)
        os.makedirs(new_version_dir, exist_ok=True)
        
        # Copy model files
        for file in os.listdir(model_dir):
            if file.endswith(".pth"):
                shutil.copy(os.path.join(model_dir, file), new_version_dir)
        
        # Update latest symlink
        latest_path = os.path.join("models", "latest")
        if os.path.exists(latest_path):
            os.remove(latest_path)
        os.symlink(version, latest_path)
        
        return True
    except Exception as e:
        logger.error(f"Failed to create new version: {str(e)}")
        return False

# ======================
# Web Interface Routes
# ======================

@app.route('/')
def start_page():
    """Landing page"""
    return render_template('start_page.html')

@app.route('/login')
def login_page():
    """Login page"""
    return render_template('login.html')

@app.route('/register')
def register_page():
    """Registration page"""
    return render_template('register.html')

@app.route('/dashboard')
@jwt_required()
def dashboard_page():
    try:
        user_id = get_jwt_identity()
        claims = get_jwt()
        username = claims.get("username")

        session = Session()
        recent_images = session.query(ImageRecord).filter_by(
            user_id=int(user_id)
        ).order_by(ImageRecord.created_at.desc()).limit(5).all()
        session.close()

        return render_template('dashboard.html',
                               username=username,
                               recent_images=recent_images)
    except Exception as e:
        logger.error(f"Dashboard error: {str(e)}")
        return redirect(url_for('login_page'))

@app.route('/api/register', methods=['POST'])
def handle_registration():
    """Handle user registration"""
    data = request.get_json()
    if not data or 'username' not in data or 'password' not in data:
        return jsonify({"error": "Missing registration data"}), 400

    session = Session()
    try:
        if session.query(User).filter_by(username=data['username']).first():
            return jsonify({"error": "Username already exists"}), 409

        new_user = User(
            username=data['username'],
            password=generate_password_hash(data['password']),
            role='user'
        )
        session.add(new_user)
        session.commit()
        
        return jsonify({
            "message": "Registration successful",
            "redirect": url_for('login_page')
        }), 201
    except Exception as e:
        session.rollback()
        logger.error(f"Registration error: {str(e)}")
        return jsonify({"error": "Registration failed"}), 500
    finally:
        session.close()

@app.route('/dashboard')
@jwt_required()
def dashboard():
    """User Dashboard"""
    try:
        current_user = get_jwt_identity()
        with Session() as session:
            user = session.query(User).get(int(current_user))
            recent_images = session.query(ImageRecord).filter_by(user_id=user.id)\
                              .order_by(ImageRecord.created_at.desc()).limit(5).all()
            return render_template('dashboard.html', 
                                 user=user,
                                 recent_images=recent_images)
    except Exception as e:
        logger.error(f"Dashboard error: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500

@app.route('/users')
@role_required(['admin'])
def manage_users():
    """User Management (Admin Only)"""
    try:
        with Session() as session:
            users = session.query(User).all()
            return render_template('user_management.html', users=users)
    except Exception as e:
        logger.error(f"User management error: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500

@app.route('/train')
@role_required(['data_scientist'])
def training_interface():
    """Training Interface"""
    return render_template('training_interface.html')

@app.route('/feedback')
@jwt_required()
@role_required(['data_scientist', 'admin'])
def feedback_analysis():
    """Feedback Analysis Page"""
    try:
        with Session() as session:
            # Get all feedbacks with associated user info
            feedbacks = session.query(
                Feedback, User.username
            ).join(
                User, Feedback.user_id == User.id
            ).order_by(
                Feedback.created_at.desc()
            ).limit(50).all()
            
            # Calculate rating distribution
            ratings = [0] * 5
            for feedback, _ in feedbacks:
                if 1 <= feedback.rating <= 5:
                    ratings[feedback.rating - 1] += 1
                    
            # Format the data for the template
            feedback_data = [{
                'id': f.id,
                'username': username,
                'rating': f.rating,
                'comment': f.comment,
                'created_at': f.created_at.strftime('%Y-%m-%d %H:%M:%S')
            } for f, username in feedbacks]
            
            return render_template(
                'feedback_analysis.html',
                feedbacks=feedback_data,
                ratings=ratings
            )
    except Exception as e:
        logger.error(f"Feedback analysis error: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500

@app.route('/monitor')
@role_required(['devops', 'admin'])
def system_monitoring():
    """System Monitoring"""
    try:
        with Session() as session:
            # Get system stats
            image_count = session.query(ImageRecord).count()
            user_count = session.query(User).count()
            recent_deployments = session.query(Deployment).order_by(
                Deployment.timestamp.desc()).limit(5).all()
            
            return render_template('system_monitoring.html',
                                 image_count=image_count,
                                 user_count=user_count,
                                 deployments=recent_deployments)
    except Exception as e:
        logger.error(f"Monitoring error: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500

@app.route('/deploy')
@role_required(['devops', 'admin'])
def model_deployment():
    """Model Deployment Interface"""
    try:
        with Session() as session:
            deployments = session.query(Deployment).order_by(Deployment.timestamp.desc()).all()
            return render_template('model_deployment.html',
                                 deployments=deployments)
    except Exception as e:
        logger.error(f"Model deployment error: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500

# ======================
# API Endpoints
# ======================

@app.route('/api/enhance', methods=['POST'])
@jwt_required()
def enhance_image():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "Empty filename"}), 400

    user_id = get_jwt_identity()

    try:
        filename = file.filename
        upload_path = os.path.join('static/uploads', filename)
        file.save(upload_path)

        enhancer = AEGANInference('models/latest', upscale_factor=4)
        with open(upload_path, 'rb') as f:
            image_bytes = f.read()

        output_path, msg = enhancer.enhance_image(image_bytes)
        if not output_path:
            return jsonify({"error": msg}), 500

        processed_filename = f"enhanced_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{filename}"
        final_output_path = os.path.join('static/processed', processed_filename)
        shutil.move(output_path, final_output_path)

        with Session() as session:
            record = ImageRecord(
                user_id=int(user_id),
                original_path=filename,
                enhanced_path=processed_filename,
                created_at=datetime.utcnow()
            )
            session.add(record)
            session.commit()
            record_id = record.id # Capture the record ID for the response

        return jsonify({
            "enhanced_url": f"/static/processed/{processed_filename}",
            "record_id": record_id
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/debug/feedbacks')
def list_feedbacks():
    with Session() as session:
        rows = session.query(Feedback).all()
        return jsonify([
            {
                "id": f.id,
                "user_id": f.user_id,
                "rating": f.rating,
                "comment": f.comment,
                "created_at": f.created_at.isoformat()
            } for f in rows
        ])


@app.route('/api/logout', methods=['POST'])
def logout():
    """Debug version of logout"""
    try:
        logger.info("Logout endpoint called")
        response = jsonify({"message": "Logout successful"})
        
        # Try to capture any possible errors with the JWT unset
        try:
            logger.info("Attempting to unset JWT cookies")
            unset_jwt_cookies(response)
            logger.info("JWT cookies unset successfully")
        except Exception as jwt_error:
            logger.error(f"Error unsetting JWT cookies: {str(jwt_error)}")
        
        # Manually delete the cookie as a fallback
        try:
            logger.info("Manually deleting access_token cookie")
            response.delete_cookie('access_token', path='/')
            logger.info("Cookie deleted successfully")
        except Exception as cookie_error:
            logger.error(f"Error deleting cookie: {str(cookie_error)}")
        
        logger.info("Returning successful logout response")
        return response
    except Exception as e:
        logger.error(f"Logout error: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/save-image', methods=['POST'])
@jwt_required()
def save_image():
    """Save enhanced image API"""
    try:
        data = request.json
        if not data or 'record_id' not in data:
            return jsonify({"error": "Invalid request"}), 400
        
        current_user = get_jwt_identity()
        
        with Session() as session:
            record = session.query(ImageRecord).filter_by(
                id=data['record_id'],
                user_id=int(current_user)
            ).first()
            
            if not record:
                return jsonify({"error": "Image not found"}), 404
                
            record.saved = True
            session.commit()
            
            return jsonify({"message": "Image saved successfully"}), 200
            
    except Exception as e:
        logger.error(f"Save image error: {str(e)}")
        return jsonify({"error": str(e)}), 500
    
@app.route('/api/feedback', methods=['POST'])
@jwt_required()
def submit_feedback():
    """Submit Feedback API"""
    try:
        # Get data from request
        data = request.get_json()
        if not data:
            logging.error("No JSON data in request")
            return jsonify({"error": "No data provided"}), 400

        if 'rating' not in data:
            logging.error("Missing rating in feedback data")
            return jsonify({"error": "Rating is required"}), 400

        # Validate rating
        try:
            rating = int(data['rating'])
            if rating < 1 or rating > 5:
                return jsonify({"error": "Rating must be between 1 and 5"}), 400
        except (ValueError, TypeError):
            return jsonify({"error": "Rating must be a number"}), 400

        # Get user ID from JWT token
        current_user = get_jwt_identity()
        if not current_user:
            return jsonify({"error": "User not authenticated"}), 401

        # Save feedback to database
        with Session() as session:
            new_feedback = Feedback(
                user_id=int(current_user),
                rating=rating,
                comment=data.get('comment', ''),
                created_at=datetime.utcnow()
            )
            session.add(new_feedback)
            session.commit()

            
            logging.info(f"Feedback submitted by user {current_user}: rating={rating}")
            return jsonify({"message": "Feedback submitted successfully", "id": new_feedback.id}), 201
            
    except Exception as e:
        logging.error(f"Feedback submission error: {str(e)}")
        return jsonify({"error": f"Failed to save feedback: {str(e)}"}), 500

from flask import make_response  # Make sure this import is present

@app.route('/api/feedback', methods=['GET'])
# For now this access token is allowed for all users will be restricted in future to admins and data scientist only
# @jwt_required()
# @role_required(['data_scientist', 'admin'])
def get_feedback():
    
    """Get Feedback Data API with pagination, search, and CSV export"""
    try:
        page = int(request.args.get('page', 1))
        per_page = int(request.args.get('per_page', 10))
        search = request.args.get('search', '').strip().lower()
        export_csv = request.args.get('export', '') == 'csv'

        with Session() as session:
            query = session.query(Feedback, User.username).join(User, Feedback.user_id == User.id)

            if search:
                query = query.filter(
                    (User.username.ilike(f'%{search}%')) |
                    (Feedback.comment.ilike(f'%{search}%'))
                )

            total = query.count()

            if export_csv:
                feedbacks = query.order_by(Feedback.created_at.desc()).all()
                csv_data = "ID,UserID,Username,Rating,Comment,Created At\n"
                for f, username in feedbacks:
                    csv_data += f'{f.id},{f.user_id},"{username}",{f.rating},"{f.comment}","{f.created_at.isoformat()}"\n'

                response = make_response(csv_data)
                response.headers["Content-Disposition"] = "attachment; filename=feedback.csv"
                response.headers["Content-Type"] = "text/csv"
                return response

            feedbacks = query.order_by(Feedback.created_at.desc())\
                             .offset((page - 1) * per_page)\
                             .limit(per_page).all()

            return jsonify({
                "page": page,
                "per_page": per_page,
                "total": total,
                "feedbacks": [{
                    "id": f.id,
                    "user_id": f.user_id,
                    "username": username,
                    "rating": f.rating,
                    "comment": f.comment,
                    "created_at": f.created_at.isoformat()
                } for f, username in feedbacks]
            })

    except Exception as e:
        logger.error(f"Feedback retrieval error: {str(e)}")
        return jsonify({"error": f"Failed to load feedback: {str(e)}"}), 500



@app.route('/api/evaluate', methods=['POST'])
@role_required(['data_scientist'])
def evaluate_model():
    """Model Evaluation API"""
    try:
        data = request.json
        test_dataset_path = data.get('test_dataset')
        
        if not test_dataset_path or not os.path.exists(test_dataset_path):
            return jsonify({"error": "Invalid test dataset path"}), 400
        
        # Load test dataset
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(256),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        dataset = dset.ImageFolder(root=test_dataset_path, transform=transform)
        dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
        
        # Initialize metrics
        total_psnr = 0
        total_ssim = 0
        count = 0
        
        enhancer = AEGANInference(current_app.config['MODEL_DIR'])
        
        with torch.no_grad():
            for images, _ in dataloader:
                images = images.to(enhancer.device)
                enhanced = enhancer.model(images)
                
                # Calculate metrics
                batch_psnr = calculate_psnr(images, enhanced)
                batch_ssim = calculate_ssim(images, enhanced)
                
                total_psnr += batch_psnr
                total_ssim += batch_ssim
                count += 1
        
        avg_psnr = total_psnr / count
        avg_ssim = total_ssim / count
        
        # Store evaluation results
        with Session() as session:
            evaluation = ModelEvaluation(
                model_version=current_app.config['MODEL_VERSION'],
                psnr=avg_psnr,
                ssim=avg_ssim,
                evaluated_at=datetime.utcnow()
            )
            session.add(evaluation)
            session.commit()
        
        return jsonify({
            "psnr": avg_psnr,
            "ssim": avg_ssim,
            "model_version": current_app.config['MODEL_VERSION']
        })
        
    except Exception as e:
        logger.error(f"Model evaluation failed: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/deploy', methods=['POST'])
@role_required(['devops'])
def deploy_model():
    """Model Deployment API"""
    try:
        model_path = request.json.get('model_path')
        if not model_path or not os.path.exists(model_path):
            return jsonify({"error": "Invalid model path"}), 400
        
        current_user = get_jwt_identity()
        
        # Create new version
        version = datetime.now().strftime("%Y%m%d-%H%M%S")
        model_dir = os.path.join("models", version)
        os.makedirs(model_dir, exist_ok=True)
        
        # Copy model files
        shutil.copy(os.path.join(model_path, "RefinerG_final.pth"), model_dir)
        
        # Update latest symlink
        latest_path = os.path.join("models", "latest")
        if os.path.exists(latest_path):
            os.remove(latest_path)
        os.symlink(version, latest_path)
        
        # Record deployment
        with Session() as session:
            deployment = Deployment(
                user_id=int(current_user),
                model_version=version,
                timestamp=datetime.utcnow(),
                status='success'
            )
            session.add(deployment)
            session.commit()
        
        return jsonify({
            "message": "Model deployed successfully",
            "version": version
        })
        
    except Exception as e:
        logger.error(f"Model deployment failed: {str(e)}")
        
        # Record failed deployment
        with Session() as session:
            deployment = Deployment(
                user_id=int(current_user),
                model_version="unknown",
                timestamp=datetime.utcnow(),
                status='failed'
            )
            session.add(deployment)
            session.commit()
        
        return jsonify({"error": str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """System Health Check"""
    try:
        # Check database connection
        with Session() as session:
            session.query(User).limit(1).all()
        
        # Check model loading
        AEGANInference(current_app.config['MODEL_DIR'])
        
        return jsonify({
            "status": "ok",
            "database": "connected",
            "model": "loaded"
        })
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500

# ======================
# WebSocket Handlers
# ======================

@socketio.on('connect')
def handle_connect():
    """WebSocket Connection Handler"""
    try:
        emit('status', {'message': 'Connected to training updates'})
        logger.info("New WebSocket connection established")
    except Exception as e:
        logger.error(f"WebSocket connection error: {str(e)}")

@socketio.on('start_training')
def handle_training(config):
    """WebSocket Training Start Handler"""
    def training_thread(config):
        try:
            logger.info(f"Starting training with config: {config}")
            trainer = AEGANTrainer(config)
            
            # Stage 1 training
            emit('training_update', {
                'stage': 1,
                'message': 'Starting Stage 1 training'
            })
            trainer.train_stage1()
            
            # Stage 2 training
            emit('training_update', {
                'stage': 2,
                'message': 'Starting Stage 2 training'
            })
            trainer.train_stage2()
            
            # Package models
            emit('training_update', {
                'stage': 3,
                'message': 'Packaging trained models'
            })
            if create_new_version(config.outf):
                emit('training_complete', {
                    'message': 'Training completed successfully',
                    'version': datetime.now().strftime("%Y%m%d-%H%M%S")
                })
            else:
                emit('training_error', {
                    'error': 'Failed to package models'
                })
            
        except Exception as e:
            logger.error(f"Training error: {str(e)}")
            emit('training_error', {'error': str(e)})
    
    try:
        thread = threading.Thread(target=training_thread, args=(config,))
        thread.start()
    except Exception as e:
        logger.error(f"Training thread error: {str(e)}")
        emit('training_error', {'error': 'Failed to start training'})

# ======================
# Utility Routes
# ======================

@app.route('/static/uploads/<filename>')
def uploaded_file(filename):
    """Serve uploaded files"""
    return send_from_directory(current_app.config['UPLOAD_FOLDER'], filename)

@app.route('/static/processed/<filename>')
def processed_file(filename):
    """Serve processed files"""
    return send_from_directory(current_app.config['PROCESSED_FOLDER'], filename)

@app.route('/dashboard/auth-check', methods=['GET'])
@jwt_required()
def dashboard_auth_check():
    """Verify authentication status for dashboard"""
    return jsonify({"status": "authenticated"}), 200

# Error Handlers
@app.errorhandler(404)
def page_not_found(e):
    """Handle 404 errors"""
    return render_template('404.html'), 404

@app.errorhandler(401)
def unauthorized(e):
    """Handle 401 errors"""
    if request.path.startswith('/api/'):
        return jsonify({"error": "Unauthorized"}), 401
    return redirect(url_for('login_page'))

@app.errorhandler(429)
def ratelimit_handler(e):
    """Handle rate limiting"""
    return jsonify({"error": "Rate limit exceeded"}), 429