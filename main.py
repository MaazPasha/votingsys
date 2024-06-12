from flask import Flask, render_template, url_for, request, session, flash, redirect
from flask_mail import *
from flask import current_app
from email.mime.multipart import MIMEMultipart
from sklearn.preprocessing import LabelEncoder
import face_recognition
import smtplib
import pymysql
import pandas as pd
import numpy as np
import os
import cv2
from PIL import Image
import shutil
import datetime
import time
import requests
from sklearn.preprocessing import LabelEncoder
import pickle
from werkzeug.utils import secure_filename
import pymysql.err

haarcascades_dir = cv2.data.haarcascades

# Load the face cascade classifier
face_cascade_path = os.path.join(haarcascades_dir, 'haarcascade_frontalface_default.xml')
face_cascade = cv2.CascadeClassifier(face_cascade_path)
le = LabelEncoder()
facedata = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
cascade = cv2.CascadeClassifier(facedata)


# Establish a connection to the MySQL database
mydb = pymysql.connect(host='localhost', user='user_name', password='password', port=port_no, database='data_base_name')

# Sender's email credentials
sender_address = 'example@gmail.com'
sender_pass = 'enter your sender pass'

# Initialize Flask app
app = Flask(__name__, template_folder='templates')
app.config['SECRET_KEY'] = 'enter secret key'
app.config['UPLOAD_FOLDER'] = "put the folder path"
STORED_IMAGES_FOLDER = "enter images stored folder path"  # Folder with stored images for verification
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}


# Initialize session variables
def initialize():
    session['IsAdmin'] = False
    session['User'] = None

# Define route for the home page
@app.route('/')
@app.route('/home')
def home():
    return render_template('index.html')


# Define route for the admin login page
@app.route('/admin', methods=['POST','GET'])
def admin():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        
        # Connect to the database
        conn = pymysql.connect(host='localhost', user='user_name', password='password', database='data_base_name')
        cursor = conn.cursor()

        # Execute a query to retrieve the user's email based on the provided email
        query = f"SELECT email FROM voters WHERE email = '{email}'"
        cursor.execute(query)
        user_email = cursor.fetchone()

        # Check if the user_email is not None and password is 'admin'
        if user_email and password == 'admin':
            session['IsAdmin'] = True
            session['User'] = user_email[0]  # Store the user's email in the session
            flash('Admin login successful', 'success')
            return redirect(url_for('admin_controls'))  # Redirect to the administrator controls route
        else:
            flash('Invalid email or password', 'danger')
            return  redirect(url_for('home'))  # Render admin login page with IsAdmin set to False
    else:
        # Set IsAdmin to False by default
        session['IsAdmin'] = False
        session['User'] = None
        return render_template('admin.html', IsAdmin=False)
    
    
@app.route('/admin_controls')
def admin_controls():
    if session.get('IsAdmin'):
        return render_template('admin_controls.html')
    else:
        flash('You are not authorized to access this page', 'danger')
        return redirect(url_for('admin'))


# Define route for adding a nominee
@app.route('/add_nominee', methods=['POST','GET'])
def add_nominee():
    if request.method == 'POST':
        member = request.form['member_name']
        party = request.form['party_name']
        logo = request.form['test']
        nominee = pd.read_sql_query('SELECT * FROM nominee', mydb)
        all_members = nominee.member_name.values
        all_parties = nominee.party_name.values
        all_symbols = nominee.symbol_name.values
        if member in all_members:
            flash("The member already exists", 'info')
        elif party in all_parties:
            flash("The party already exists", 'info')
        elif logo in all_symbols:
            flash("The logo is already taken", 'info')
        else:
            sql = "INSERT INTO nominee (member_name, party_name, symbol_name) VALUES (%s, %s, %s)"
            cur = mydb.cursor()
            cur.execute(sql, (member, party, logo))
            mydb.commit()
            cur.close()
            flash("Successfully registered a new nominee", 'primary')
    return render_template('nominee.html', admin=session['IsAdmin'])

# Define route for voter registration
@app.route('/registration', methods=['POST','GET'])
def registration():
    if request.method == 'POST':
        first_name = request.form['first_name']
        last_name = request.form['last_name']
        state = request.form['state']
        d_name = request.form['d_name']
        middle_name = request.form['middle_name']
        aadhar_id = request.form['aadhar_id']
        voter_id = request.form['voter_id']
        pno = request.form['pno']
        age = int(request.form['age'])
        email = request.form['email']
        voters = pd.read_sql_query('SELECT * FROM voters', mydb)
        all_aadhar_ids = voters.aadhar_id.values
        all_voter_ids = voters.voter_id.values
        if age >= 18:
            if (aadhar_id in all_aadhar_ids) or (voter_id in all_voter_ids):
                flash("Already Registered as a Voter")
            else:
                sql = 'INSERT INTO voters (first_name, middle_name, last_name, aadhar_id, voter_id, email,pno,state,d_name, verified) VALUES (%s,%s,%s, %s, %s, %s, %s, %s, %s, %s)'
                cur = mydb.cursor()
                cur.execute(sql, (first_name, middle_name, last_name, aadhar_id, voter_id, email, pno, state, d_name, 'no'))
                mydb.commit()
                cur.close()
                session['aadhar'] = aadhar_id
                session['status'] = 'no'
                session['email'] = email
                return redirect(url_for('verify'))
        else:
            flash("if age less than 18 than not eligible for voting", "info")
    return render_template('voter_reg.html')

# Define route for email verification
@app.route('/verify', methods=['POST','GET'])
def verify():
    if session.get('status') == 'no':
        if request.method == 'POST':
            otp_check = request.form['otp_check']
            if otp_check == session.get('otp'):
                session['status'] = 'yes'
                sql = "UPDATE voters SET verified=%s WHERE aadhar_id=%s"
                cur = mydb.cursor()
                cur.execute(sql, ('yes', session.get('aadhar')))
                mydb.commit()
                cur.close()
                flash("Email verified successfully", 'primary')
                return render_template('capture.html')
            else:
                flash("Wrong OTP. Please try again.", "info")
                return redirect(url_for('verify'))
        else:
            # Sending OTP
            message = MIMEMultipart()
            receiver_address = session.get('email')
            message['From'] = sender_address
            message['To'] = receiver_address
            Otp = str(np.random.randint(100000, 999999))
            session['otp'] = Otp
            message.attach(MIMEText(session.get('otp'), 'plain'))
            try:
                with smtplib.SMTP('smtp.gmail.com', 587) as abc:
                    abc.starttls()
                    abc.login(sender_address, sender_pass)
                    text = message.as_string()
                    abc.sendmail(sender_address, receiver_address, text)
            except smtplib.SMTPAuthenticationError as e:
                flash("SMTP Authentication Error. Please check your email credentials and try again.", 'danger')
                return redirect(url_for('home'))  # Redirect to home page or appropriate error page
            except Exception as e:
                flash("An error occurred while sending the OTP. Please try again later.", 'danger')
                return redirect(url_for('home'))  # Redirect to home page or appropriate error page
    else:
        flash("Your email is already verified", 'warning')
    return render_template('verify.html')


# Define route for capturing images
@app.route('/capture_images', methods=['POST','GET'])
def capture_images():
    if request.method == 'POST':
        # Initialize webcam
        cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        sampleNum = 0
        path_to_store = os.path.join(os.getcwd(), "C:/tahirpro/static/user_images" + session['aadhar'])
        os.makedirs(path_to_store, exist_ok=True)
        
        # Capture and store grayscale images
        while True:
            ret, img = cam.read()
            try:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            except:
                continue
            
            # Detect faces
            faces = cascade.detectMultiScale(gray, 1.3, 5)
            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                # Increment sample number
                sampleNum += 1
                # Save grayscale image in the specified folder
                cv2.imwrite(os.path.join(path_to_store, f"{sampleNum}.jpg"), gray[y:y + h, x:x + w])
            
            cv2.imshow('frame', img)
            cv2.setWindowProperty('frame', cv2.WND_PROP_TOPMOST, 1)
            
            # Break loop if 'q' is pressed or sample number exceeds 200
            if cv2.waitKey(100) & 0xFF == ord('q') or sampleNum >= 200:
                break
        
        # Release webcam and close OpenCV windows
        cam.release()
        cv2.destroyAllWindows()
        
        flash("Images captured and stored successfully", "success")
        return redirect(url_for('home'))  # Redirect to home page after capturing images
    
    return render_template('capture.html')

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/verify_image', methods=['GET', 'POST'])
def verify_image():
    if request.method == 'POST':
        cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        ret, frame = cam.read()
        if not ret:
            flash('Unable to access the camera', 'danger')
            return redirect(request.url)
        
        captured_image_path = 'captured_image.jpg'
        cv2.imwrite(captured_image_path, frame)
        cam.release()

        stored_image_paths = [os.path.join(STORED_IMAGES_FOLDER, f) for f in os.listdir(STORED_IMAGES_FOLDER) if allowed_file(f)]
        verified, message = verify_image(captured_image_path, stored_image_paths)

        if verified:
            # Assuming the Aadhar ID is part of the filename or can be derived from it
            # For demonstration, we're setting a fixed Aadhar ID
            session['select_aadhar'] = 'some_aadhar_id'
            flash('Image verified successfully! ' + message, 'success')
            return redirect(url_for('select_candidate'))  # Redirect to select_candidate page after successful verification
        else:
            flash('Image verification failed. ' + message, 'danger')
            return redirect(url_for('verify_image'))

    return render_template('verify_image.html')


def verify_image(captured_image_path, stored_image_paths):
    captured_image = cv2.imread(captured_image_path)
    gray = cv2.cvtColor(captured_image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) == 0:
        return False, "No faces detected in the captured image."

    for stored_image_path in stored_image_paths:
        stored_image = cv2.imread(stored_image_path)
        stored_gray = cv2.cvtColor(stored_image, cv2.COLOR_BGR2GRAY)
        result = cv2.matchTemplate(gray, stored_gray, cv2.TM_CCOEFF_NORMED)
        if cv2.minMaxLoc(result)[1] > 0.8:
            return True, "Face matched successfully."

    return False, "No matching faces found."

def detect_faces_and_recognize(cam):
    if not cam.isOpened():
        flash("Unable to open camera", "error")
        return None

    detected_persons = []
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    flag = 0

    while True:
        ret, im = cam.read()
        if not ret:
            flash("Unable to capture image from camera", "error")
            return None

        flag += 1
        if flag == 200:
            flash("Unable to detect person. Contact help desk for manual voting", "info")
            cv2.destroyAllWindows()
            return None

        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

        for (x, y, w, h) in faces:
            cv2.rectangle(im, (x, y), (x + w, y + h), (225, 0, 0), 2)
            captured_face = cv2.resize(gray[y:y + h, x:x + w], (100, 100))
            recognized_id = recognize_face(captured_face)

            if recognized_id:
                detected_persons.append(recognized_id)
                cv2.putText(im, f"Aadhar: {recognized_id}", (x, y + h), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            else:
                cv2.putText(im, "Unknown", (x, y + h), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        cv2.imshow('im', im)
        cv2.setWindowProperty('im', cv2.WND_PROP_TOPMOST, 1)

        if cv2.waitKey(1) == ord('q'):
            cv2.destroyAllWindows()
            return detected_persons

def recognize_face(captured_face):
    stored_image_paths = [os.path.join(STORED_IMAGES_FOLDER, f) for f in os.listdir(STORED_IMAGES_FOLDER) if allowed_file(f)]
    for stored_image_path in stored_image_paths:
        stored_image = cv2.imread(stored_image_path, cv2.IMREAD_GRAYSCALE)
        result = cv2.matchTemplate(captured_face, stored_image, cv2.TM_CCOEFF_NORMED)
        if cv2.minMaxLoc(result)[1] > 0.8:
            return stored_image_path.split("\\")[-1]  # Extract Aadhar ID from file path
    return None



# Define a function to get images and their labels
def getImagesAndLabels(path):
    folderPaths = [os.path.join(path, f) for f in os.listdir(path)]
    faces = []
    Ids = []
    global le
    for folder in folderPaths:
        imagePaths = [os.path.join(folder, f) for f in os.listdir(folder)]
        aadhar_id = folder.split("\\")[-1]  # Get the Aadhar ID from the folder name
        for imagePath in imagePaths:
            # loading the image and converting it to gray scale
            pilImage = Image.open(imagePath).convert('L')
            # Now we are converting the PIL image into numpy array
            imageNp = np.array(pilImage, 'uint8')
            # extract the face from the training image sample
            faces.append(imageNp)
            Ids.append(aadhar_id)
    
    # Fit and transform the labels using LabelEncoder
    Ids_encoded = le.fit_transform(Ids)
    
    # Save the LabelEncoder for later use
    with open('encoder.pkl', 'wb') as output:
        pickle.dump(le, output)
    
    return faces, Ids_encoded


# Define route for training the face recognition model
@app.route('/train', methods=['POST','GET'])
def train():
    if request.method == 'POST':
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        faces, Ids = getImagesAndLabels(r"C:\tahirpro\code\user_images")
        
        # Convert Aadhar IDs to integer labels
        labels = [int(aadhar_id) for aadhar_id in Ids]
        
        recognizer.train(faces, np.array(labels))
        recognizer.save("Trained.yml")
        flash("Model Trained Successfully", 'primary')
        return redirect(url_for('home'))
    return render_template('train.html')


# Define route for updating voter information
@app.route('/update')
def update():
    return render_template('update.html')

# Define route for updating voter information after submission
@app.route('/updateback', methods=['POST','GET'])
def updateback():
    if request.method == 'POST':
        first_name = request.form['first_name']
        last_name = request.form['last_name']
        middle_name = request.form['middle_name']
        aadhar_id = request.form['aadhar_id']
        voter_id = request.form['voter_id']
        email = request.form['email']
        pno = request.form['pno']
        age = int(request.form['age'])
        voters = pd.read_sql_query('SELECT * FROM voters', mydb)
        all_aadhar_ids = voters.aadhar_id.values
        if age >= 18:
            if aadhar_id in all_aadhar_ids:
                sql = "UPDATE VOTERS SET first_name=%s, middle_name=%s, last_name=%s, voter_id=%s, email=%s,pno=%s, verified=%s where aadhar_id=%s"
                cur = mydb.cursor()
                cur.execute(sql, (first_name, middle_name, last_name, voter_id, email,pno, 'no', aadhar_id))
                mydb.commit()
                cur.close()
                session['aadhar'] = aadhar_id
                session['status'] = 'no'
                session['email'] = email
                flash("Database Updated Successfully", 'Primary')
                return redirect(url_for('verify'))
            else:
                flash(f"Aadhar: {aadhar_id} doesn't exist in the database for updation", 'warning')
        else:
            flash("Age should be 18 or greater than 18 to be eligible", "info")
    return render_template('update.html')


@app.route('/vote', methods=['GET'])
def vote():
    return redirect(url_for('verify_image'))


# Define route for voting
@app.route('/voting', methods=['POST', 'GET'])
def voting():
    if 'select_aadhar' not in session:
        return redirect(url_for('vote'))

    if request.method == 'POST':
        aadhar = session['select_aadhar']
        voted_candidate = request.form.get('candidate')  # Ensure the correct name of the input field in the form
        
        # Check if the user has already voted
        cur = mydb.cursor()
        try:
            cur.execute("SELECT * FROM vote WHERE aadhar = %s", (aadhar,))
            if cur.fetchone():
                flash("You have already voted", "warning")
            else:
                # Insert the vote along with the Aadhar ID into the database
                sql = "INSERT INTO vote (vote, aadhar) VALUES (%s, %s)"
                values = (voted_candidate, aadhar)
                cur.execute(sql, values)

                # Update the 'voted' status in the voters table
                update_sql = "UPDATE voters SET voted = TRUE WHERE aadhar_id = %s"
                cur.execute(update_sql, (aadhar,))
                
                mydb.commit()
                flash("Voted Successfully", 'primary')
        except pymysql.err.IntegrityError as e:
            mydb.rollback()
            flash("An error occurred while voting", "danger")
            print("Error:", e)  # Log the error for debugging
        finally:
            cur.close()
            return redirect(url_for('home'))

    return render_template('voting.html')


# Define route for selecting a candidate to vote for
# Define route for selecting a candidate to vote for
@app.route('/select_candidate', methods=['POST', 'GET'])
def select_candidate():
    aadhar = session.get('aadhar')  # Corrected from 'select_aadhar'
    if not aadhar:
        flash("No Aadhar ID found. Please verify your identity first.", "danger")
        return redirect(url_for('vote'))

    df_nom = pd.read_sql_query('SELECT * FROM nominee', mydb)
    all_nom = df_nom['symbol_name'].values

    if request.method == 'POST':
        vote = request.form['test']
        session['vote'] = vote

        # Check if the user has already voted
        cur = mydb.cursor()
        try:
            cur.execute("SELECT * FROM vote WHERE aadhar = %s", (aadhar,))
            if cur.fetchone():
                flash("You have already voted", "warning")
            else:
                sql = "INSERT INTO vote (vote, aadhar) VALUES (%s, %s)"
                values = (vote, aadhar)
                cur.execute(sql, values)
                mydb.commit()
                flash("Voted Successfully", 'primary')
        except pymysql.err.IntegrityError as e:
            mydb.rollback()
            flash("An error occurred while voting", "danger")
        finally:
            cur.close()
            return redirect(url_for('home'))

    return render_template('select_candidate.html', noms=sorted(all_nom))

# Define route for viewing voting results
@app.route('/voting_res', methods=['POST', 'GET'])
def voting_res():
    df_vote = pd.read_sql_query('SELECT * FROM vote', mydb)
    
    # Ensure all possible images are considered
    all_possible_imgs = ['1.png', '2.png', '3.jpg', '4.png', '5.png', '6.png']
    
    # Count votes for each image
    counts = df_vote['vote'].value_counts().to_dict()

    # Prepare frequencies for all possible images, including those with 0 votes
    all_freqs = [counts.get(img, 0) for img in all_possible_imgs]

    return render_template('voting_res.html', all_freqs=all_freqs, all_possible_imgs=all_possible_imgs)


# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)