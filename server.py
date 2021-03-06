from flask import Flask, render_template, request, Response, flash, redirect, url_for
from werkzeug.utils import secure_filename
import os
import tensorflow as tf
import cv2
import numpy as np
import main as m
import psycopg2
import credentials as cr
import extract_features as ef
from datetime import date, timedelta

UPLOAD_FOLDER = './uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
model_val_acc = tf.keras.models.load_model('./nudityModel/max_val_accuracy_model.epoch02-val_accuracy0.98')
model_val_loss = tf.keras.models.load_model('./nudityModel/min_val_loss_model.epoch01-val_loss0.08')
model_acc = tf.keras.models.load_model('./nudityModel/max_accuracy_model.epoch02-accuracy0.93')
model_loss = tf.keras.models.load_model('./nudityModel/min_loss_model.epoch09-loss0.14')
size = (50, 50)
app.secret_key = cr.secret_key
app.permanent_session_lifetime = timedelta(hours=cr.time)
con = psycopg2.connect(host=cr.host, database=cr.db_name, user=cr.username, password=cr.password)
cur = con.cursor()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def home():
    return "<h1>Welcome to the MOLECULE BACKEND API</h1>"


@app.route('/check', methods=['GET', 'POST'])
def imgCheck():
    if request.method == 'POST':
        if 'img' not in request.files:
            flash('No img part')
            print('No img part')
            return redirect(request.url)
        
        file = request.files['img']
        
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            
            image = cv2.imread(f'./uploads/{filename}')
            image = cv2.resize(image, size)
            image = np.array([image])
            image = image/255.0
            
            pred1 = list(model_val_loss.predict(image)[0])
            pred2 = list(model_val_acc.predict(image)[0])
            pred3 = list(model_acc.predict(image)[0])
            pred4 = list(model_loss.predict(image)[0])
            
            p = [pred1, pred2, pred3, pred4]
            zero = 0
            for i in p:
                if i.index(max(i)) == 0:
                    zero += 1
            
            if zero > 2:
                return 'SFW'
            elif zero < 2:
                return 'NSFW'
            else:
                return 'NSFW'
    

@app.route('/query', methods=['GET', 'POST'])
def query():
    if request.method == 'POST':
        data = request.get_json()
        try:
            query = data['query']
            print(query)
            result = m.run(query)
            return result
        except:
            return Response(response='POST fail')
        
    if request.method == 'GET':
        try:
            query = request.args.get('q')
            query = query.strip()
            print(query)
            result = m.run(query)
            return result
        except:
            return Response(response='GET fail')
        
    return Response(response='QUERY Request fail', status=404)


@app.route('/geturl')
def geturl():
    url = request.args.get('url')
    print('URL: ', url)
    query = f"select * from phishing_urls where url='{url}';"
    cur.execute(query)
    for i in cur.fetchall():
        if i:
            print('Found in DATABASE')
            flash('PHISHING URL')
            return redirect(url_for('phishing'))
    result = ef.go(url)
    ef.data = []
    if result == 'PHISHING URL':
        try:
            query = "insert into phishing_urls(url,date) values('" + url + "',DATE '" + str(date.today()) + "');"
            cur.execute(query)
            con.commit()
        except:
            print('Some problem in entering the URL into database')
    flash(result)
    if result[0] == 'P':
        return redirect(url_for('phishing'))
    else:
        return redirect(url_for('safe'))

@app.route('/phishing')
def phishing():
    return '<h1>Phishing URL detected</h1>'

@app.route('/safe')
def safe():
    return '<h1>Legitimate URL</h1>'

if __name__ == '__main__':
    app.run()



