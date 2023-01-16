import os.path
from flask import Flask, request, redirect, url_for, render_template, flash, send_file, Response
from werkzeug.utils import secure_filename
import camera_rendering

app = Flask(__name__)

app.secret_key = "secret key"
app.config['HOST'] = '127.0.0.9'
app.config['PORT'] = 8080
app.config['UPLOAD_FOLDER'] = '/static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

def is_image_file(filename):
   return os.path.splitext(filename)[1].lower() in {'.jpg', '.jpeg', '.png', '.gif'}

@app.route('/choice')
def choice():
    return render_template('camera_import.html')

@app.route('/camera')
def camera_display():
    return render_template('camera.html')

@app.route('/camera_import')
def camera_import():
    return Response(camera_rendering.launch_camera(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/cameratest')
def camera_picture():
    return send_file(camera_rendering.capture(app.config['UPLOAD_FOLDER']), mimetype='image/jpeg')

@app.route('/')
def image_interface():
        return render_template('index1.html'), 404

@app.route('/', methods=['GET', 'POST'])
def import_image():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        if  request.files['file'].filename == '':
            flash('No image selected for uploading')
            return redirect(request.url)
        if request.files['file'] and is_image_file(request.files['file'].filename):
            request.files['file'].save(os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(request.files['file'].filename)))
            flash('Image successfully uploaded and displayed below')
            return render_template('index1.html', filename=secure_filename(request.files['file'].filename))
        else:
            flash('Allowed image types are - png, jpg, jpeg, gif')
            return redirect(request.url)
 
@app.route('/display/<filename>')
def display_image(filename):
    return redirect(url_for('static', filename=f'uploads/{filename}'), code=301)
        

if __name__ == '__main__':
    app.run(host=app.config['HOST'], port=app.config['PORT'], debug=True)