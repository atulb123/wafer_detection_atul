import os
import shutil
from wsgiref import simple_server
from flask import Flask, request, render_template
from flask_cors import CORS, cross_origin
from os_level_operations.create_delete_folder import CreateDeleteFolder
from test.test_flow import TestFlow
from train.train_flow import TrainFLow

app = Flask(__name__)
CORS(app)

@app.route("/", methods=['GET'])
@cross_origin()
def home():
    return render_template('index.html')


@app.route("/train", methods=['POST'])
@cross_origin()
def train():
    try:
        folder = CreateDeleteFolder()
        folder.perform_folder_clean_up()
        files = request.files.getlist("file")
        if "octet-stream" in str(files[0]):
            folder.copy_files_to_folder(os.getcwd() + "/train_test_files/default_training_files",
                                        os.getcwd() + "/train_test_files/train_raw_dataset")
        else:
            for file in files:
                file.save(os.getcwd() + "/train_test_files/train_raw_dataset/" + file.filename)
        TrainFLow().train_flow()
        return render_template("train.html", message="Train Flow Successfully Completed")
    except BaseException as msg:
        return render_template("train.html", message=msg)


@app.route("/predict", methods=['POST'])
@cross_origin()
def predict():
    try:
        folder = CreateDeleteFolder()
        files = request.files.getlist("file")
        if "octet-stream" in str(files[0]):
            folder.copy_files_to_folder(os.getcwd() + "/train_test_files/default_test_files",
                                        os.getcwd() + "/train_test_files/test_raw_dataset")
        else:
            for file in files:
                file.save(os.getcwd() + "/train_test_files/test_raw_dataset/" + file.filename)
        TestFlow().test_flow()
        shutil.move(os.getcwd() + "/train_test_files/test_files/prediction_output.csv",
                    os.getcwd() + "/static/prediction_output.csv")
        return render_template("predict.html", message="Test Flow Successfully Completed",
                               url="/static/prediction_output.csv")
    except BaseException as msg:
        return render_template("predict.html", message=msg, url="")


# @app.route("/testing", methods=['GET'])
# @cross_origin()
# def testing():
#     return render_template("predict.html", message="hello", url="/static/prediction_output.csv")

port = int(os.getenv("PORT", 5001))
if __name__ == "__main__":
    host = '0.0.0.0'
    httpd = simple_server.make_server(host=host, port=port, app=app)
    httpd.serve_forever()

