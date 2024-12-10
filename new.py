# Import necessary libraries
from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import os
import smtplib
import PyPDF2
import re
import spacy
import joblib
from datetime import datetime
from sqlalchemy.exc import SQLAlchemyError
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Replace with your actual secret key
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///candidates.db'
app.config['UPLOAD_FOLDER'] = 'uploads'
db = SQLAlchemy(app)

ALLOWED_EXTENSIONS = {'pdf'}

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Define the path to the models
desktop_path = os.path.expanduser("~/Desktop")
model_folder = os.path.join(desktop_path, "final sy")

# Load Naive Bayes model and vectorizer
naive_bayes_model = joblib.load(os.path.join(model_folder, 'naive_bayes_model.pkl'))
tfidf_vectorizer = joblib.load(os.path.join(model_folder, 'tfidf_vectorizer.pkl'))

# User model
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(128))

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

# Candidate model with education requirements
class Candidate(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(150), nullable=False)
    email = db.Column(db.String(150), nullable=False)
    resume_text = db.Column(db.Text, nullable=False)
    rank_score = db.Column(db.Float, nullable=False)
    reviewed = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    experience = db.Column(db.String(200))
    education = db.Column(db.String(200))
    has_o_levels = db.Column(db.Boolean, default=False)
    has_a_levels = db.Column(db.Boolean, default=False)
    has_degree = db.Column(db.Boolean, default=False)
    meets_education_criteria = db.Column(db.Boolean, default=False)

# SelectedCandidate model
class SelectedCandidate(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    candidate_id = db.Column(db.Integer, db.ForeignKey('candidate.id'), nullable=False)
    selected_at = db.Column(db.DateTime, default=datetime.utcnow)
    candidate = db.relationship('Candidate', backref=db.backref('selected', lazy=True))

# Education requirements analyzer
def analyze_education_requirements(education_text):
    education_text = education_text.lower()
    
    requirements = {
        'o_levels': False,
        'a_levels': False,
        'degree': False
    }
    
    # Check for O Levels
    o_level_patterns = [
        r'5\s+o[\s-]*levels?',
        r"5\s+'o'[\s-]*levels?",
        r'five\s+o[\s-]*levels?',
        r'5\s+ordinary[\s-]*levels?'
    ]
    requirements['o_levels'] = any(re.search(pattern, education_text) for pattern in o_level_patterns)
    
    # Check for A Levels
    a_level_patterns = [
        r'3\s+a[\s-]*levels?',
        r"3\s+'a'[\s-]*levels?",
        r'three\s+a[\s-]*levels?',
        r'3\s+advanced[\s-]*levels?'
    ]
    requirements['a_levels'] = any(re.search(pattern, education_text) for pattern in a_level_patterns)
    
    # Check for degree
    degree_patterns = [
        r'bachelor[\'s]*\s+degree',
        r'b\.?s\.?c\.?',
        r'b\.?a\.?',
        r'degree\s+in'
    ]
    requirements['degree'] = any(re.search(pattern, education_text) for pattern in degree_patterns)
    
    return requirements

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Create tables
with app.app_context():
    try:
        db.create_all()
    except SQLAlchemyError as e:
        print(f"Database error: {e}")

# Create tables
with app.app_context():
    db.create_all()

# Check if file is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Rank scores logic
rank_keywords = {
    'responsible': 5, 'hardworking': 5, 'dedicated': 5, 'motivated': 6,
    'reliable': 7, 'detail-oriented': 5, 'self-starter': 6, 'team player': 6,
    'proactive': 7, 'passionate': 8, 'proficient in': 8, 'experienced with': 9,
    'bachelor\'s degree': 5, 'certified in': 7, 'managed': 6, 'developed': 8,
    'implemented': 8, 'achieved': 9, 'trained': 7, 'coordinated': 6,
    'awarded': 7, 'recognized': 8, 'published': 7, 'led': 6, 'expert in': 9,
    'innovated': 9, 'honored': 5, 'directed': 7, 'outstanding': 8,
    'specialized in': 8, 'proven track record': 9, 'leadership experience': 9,
    'award-winning': 9, 'industry leader': 8, 'strategic thinker': 6,
    'exceptional': 8, 'visionary': 6, 'accomplished': 7, 'influential': 8,
    'distinguished': 5
}

def calculate_rank_score(resume_text):
    # Use both keyword-based scoring and Naive Bayes prediction
    keyword_score = sum(rank * len(re.findall(rf'\b{re.escape(word)}\b', resume_text.lower()))
                        for word, rank in rank_keywords.items())
    
    # Vectorize the input text using the TF-IDF vectorizer
    vectorized_text = tfidf_vectorizer.transform([resume_text])
    
    # Naive Bayes prediction (assuming it returns a probability)
    nb_score = naive_bayes_model.predict_proba(vectorized_text)[0][1]  # Assuming binary classification
    
    # Combine scores (you may want to adjust the weights)
    combined_score = 0.7 * keyword_score + 0.3 * (nb_score * 100)
    
    return combined_score

# Extract information from resume text using spaCy
def extract_info(resume_text):
    doc = nlp(resume_text)
    
    # Extract name (assuming the first person entity is the candidate's name)
    name = next((ent.text for ent in doc.ents if ent.label_ == "PERSON"), "Unknown")
    
    # Extract email
    email = next((token.text for token in doc if token.like_email), "unknown@example.com")
    
    # Extract experience (you may need to refine this based on your resume format)
    experience = " ".join([sent.text for sent in doc.sents if "experience" in sent.text.lower()])
    
    # Extract education (you may need to refine this based on your resume format)
    education = " ".join([sent.text for sent in doc.sents if "education" in sent.text.lower()])
    
    return name, email, experience, education

@app.route('/', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']

        if User.query.filter_by(username=username).first() or User.query.filter_by(email=email).first():
            flash('Username or email already exists.', 'danger')
        else:
            new_user = User(username=username, email=email)
            new_user.set_password(password)
            db.session.add(new_user)
            db.session.commit()
            flash('Registration successful. Please log in.', 'success')
            return redirect(url_for('login'))

    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = User.query.filter_by(username=username).first()

        if user and user.check_password(password):
            session['user_id'] = user.id
            flash('Logged in successfully.', 'success')
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid username or password.', 'danger')

    return render_template('login.html')

@app.route('/dashboard')
def dashboard():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return render_template('dashboard.html')

@app.route('/logout')
def logout():
    session.pop('user_id', None)
    flash('You have been logged out.', 'info')
    return redirect(url_for('login'))

@app.route('/api/mark_reviewed/<int:candidate_id>', methods=['POST'])
def mark_reviewed(candidate_id):
    candidate = Candidate.query.get(candidate_id)
    
    if not candidate:
        return jsonify({'error': 'Candidate not found'}), 404

    # Mark the candidate as reviewed
    candidate.reviewed = True
    db.session.commit()

    return jsonify({'message': 'Candidate marked as reviewed successfully'})

@app.route('/api/filter_candidates')
def filter_candidates():
    min_score = request.args.get('min_score', type=float)
    max_score = request.args.get('max_score', type=float)
    keyword = request.args.get('keyword')

    query = Candidate.query

    if min_score is not None:
        query = query.filter(Candidate.rank_score >= min_score)
    if max_score is not None:
        query = query.filter(Candidate.rank_score <= max_score)
    if keyword:
        query = query.filter(Candidate.resume_text.ilike(f'%{keyword}%'))

    candidates = query.order_by(Candidate.rank_score.desc()).all()
    return jsonify([{
        'id': c.id,
        'name': c.name,
        'email': c.email,
        'rank_score': c.rank_score
    } for c in candidates])
    
@app.route('/notifications/<int:candidate_id>/<string:email>', methods=['GET', 'POST'])
def notifications(candidate_id, email):
    if 'user_id' not in session:
        return redirect(url_for('login'))

    candidate = Candidate.query.get_or_404(candidate_id)

    if request.method == 'POST':
        message = request.form['message']

        try:
            message = Mail(
                from_email='nheperatakunda612@gmail.com',
                to_emails=email,
                subject='Application Update',
                plain_text_content=message)
            
            sg = SendGridAPIClient('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')  # Replace with your API key
            response = sg.send(message)
            
            if response.status_code == 202:
                flash('Email sent successfully', 'success')
            else:
                flash('Error sending email', 'danger')
        except Exception as e:
            flash(f'Error sending email: {e}', 'danger')
            print(f"Detailed error: {str(e)}")  # For debugging

    return render_template('notifications.html', candidate=candidate, email=email)

@app.route('/applications')
def applications():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    candidates = Candidate.query.order_by(Candidate.rank_score.desc()).all()
    return render_template('applications.html', candidates=candidates)

# Modified upload route to include education analysis
@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part', 'danger')
            return redirect(request.url)

        file = request.files['file']
        
        if file.filename == '':
            flash('No selected file', 'danger')
            return redirect(request.url)

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            with open(filepath, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                resume_text = ""
                for page in reader.pages:
                    text = page.extract_text()
                    if text:
                        resume_text += text

            score = calculate_rank_score(resume_text)
            name, email, experience, education = extract_info(resume_text)
            
            # Analyze education requirements
            education_reqs = analyze_education_requirements(education)
            meets_all_criteria = all(education_reqs.values())
            
            new_candidate = Candidate(
                name=name,
                email=email,
                resume_text=resume_text,
                rank_score=score,
                experience=experience,
                education=education,
                has_o_levels=education_reqs['o_levels'],
                has_a_levels=education_reqs['a_levels'],
                has_degree=education_reqs['degree'],
                meets_education_criteria=meets_all_criteria
            )
            
            db.session.add(new_candidate)
            db.session.commit()

            # If candidate meets all criteria, automatically add to final panel
            if meets_all_criteria:
                selected = SelectedCandidate(candidate_id=new_candidate.id)
                db.session.add(selected)
                db.session.commit()

            return redirect(url_for('analysis', candidate_id=new_candidate.id))
    
    return render_template('upload.html')

# Modified analysis route
@app.route('/analysis', methods=['GET', 'POST'])
def analysis():
    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({'error': 'No file part in the request'}), 400
        
        file = request.files['file']

        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Extract resume text from the file
            with open(filepath, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                resume_text = ""
                for page in reader.pages:
                    text = page.extract_text()
                    if text:
                        resume_text += text

            # Calculate rank score and extract info
            rank_score = calculate_rank_score(resume_text)
            name, email, experience, education = extract_info(resume_text)

            # Analyze education requirements (O Levels, A Levels, Degree)
            education_reqs = analyze_education_requirements(education)
            meets_all_criteria = all(education_reqs.values())

            # Save the result to the database
            candidate = Candidate(
                name=name,
                email=email,
                resume_text=resume_text,
                rank_score=rank_score,
                experience=experience,
                education=education,
                has_o_levels=education_reqs['o_levels'],
                has_a_levels=education_reqs['a_levels'],
                has_degree=education_reqs['degree'],
                meets_education_criteria=meets_all_criteria,
                created_at=datetime.utcnow()
            )
            db.session.add(candidate)
            db.session.commit()

            # Render result page with the education status
            education_status = {
                'o_levels': candidate.has_o_levels,
                'a_levels': candidate.has_a_levels,
                'degree': candidate.has_degree,
                'meets_all': candidate.meets_education_criteria
            }
            
            return render_template('result.html', candidate=candidate, education_status=education_status)
    
    elif request.method == 'GET':
        candidate_id = request.args.get('candidate_id')
        if candidate_id:
            candidate = Candidate.query.get(candidate_id)
            if candidate:
                # Prepare education status for display
                education_status = {
                    'o_levels': candidate.has_o_levels,
                    'a_levels': candidate.has_a_levels,
                    'degree': candidate.has_degree,
                    'meets_all': candidate.meets_education_criteria
                }
                
                return render_template('result.html', candidate=candidate, education_status=education_status)
            else:
                return jsonify({'error': 'Candidate not found'}), 404
        else:
            return jsonify({'error': 'No candidate ID provided'}), 400
    
    return jsonify({'error': 'Invalid request'}), 400


# Modified final panel route to include education criteria
@app.route('/shortlist', methods=['GET', 'POST'])
def shortlist():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    if request.method == 'POST':
        data = request.get_json()
        if not data or 'candidates' not in data:
            return jsonify({'success': False, 'message': 'No candidates selected'}), 400

        try:
            # Clear existing selections
            SelectedCandidate.query.delete()
            
            # Get candidates with high scores (≥ 70) or meeting education criteria
            qualified_candidates = (
                Candidate.query
                .filter(
                    db.or_(
                        Candidate.rank_score >= 70,
                        Candidate.meets_education_criteria == True
                    )
                )
                .with_entities(Candidate.id)
                .all()
            )
            
            qualified_ids = [c.id for c in qualified_candidates]
            
            # Combine manually selected candidates with qualified candidates
            all_candidate_ids = list(set(data['candidates'] + qualified_ids))
            
            # Add all selections to the database
            for candidate_id in all_candidate_ids:
                selected = SelectedCandidate(candidate_id=candidate_id)
                db.session.add(selected)
            
            db.session.commit()
            return jsonify({
                'success': True, 
                'message': 'Candidates selected successfully',
                'auto_selected': len(qualified_ids)
            })
        
        except Exception as e:
            db.session.rollback()
            return jsonify({'success': False, 'message': str(e)}), 500

    # GET request - display final panel
    qualified_candidates = (
        Candidate.query
        .outerjoin(SelectedCandidate)
        .filter(
            db.or_(
                SelectedCandidate.candidate_id.isnot(None),
                Candidate.rank_score >= 70,
                Candidate.meets_education_criteria == True
            )
        )
        .order_by(Candidate.rank_score.desc(), Candidate.name)
        .all()
    )
    
    return render_template(
        'shortlist.html', 
        candidates=qualified_candidates,
        high_score_threshold=70
    )
@app.route('/api/remove_from_final/<int:candidate_id>', methods=['POST'])
def remove_from_final(candidate_id):
    if 'user_id' not in session:
        return jsonify({'success': False, 'message': 'Not authenticated'}), 401

    try:
        selection = SelectedCandidate.query.filter_by(candidate_id=candidate_id).first()
        if selection:
            db.session.delete(selection)
            db.session.commit()
            return jsonify({'success': True, 'message': 'Candidate removed successfully'})
        return jsonify({'success': False, 'message': 'Candidate not found'}), 404
    
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'message': str(e)}), 500
        
if __name__ == '__main__':
    app.run(debug=True)