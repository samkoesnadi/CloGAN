# ANALYSIS OF DEEP DOMAIN SHIFT FOR MEDICAL CLASSIFICATION TASKS ON CHEST X-RAY DATASETS

The project directory consists of files, programmed in Python. The description of each file is commented in the beginning of the corresponding file.

>To have the pre-trained model, download https://github.com/samuelmat19/CloGAN/releases/download/pretrained-model/model.hdf5 and copy it to "resources" directory.

## Tutorial

### Predict abnormalities/symptoms in chest X-ray image with Xception network, trained on classification loss and CloGAN

1. Make sure that Python 3 is installed in your system.
2. Open terminal or command line that has access to the Python binary and change the directory to this project.
3. Install Python virtual environment : `pip3 install --user virtualenv`, in case you do not have the virtual environment installed.
4. Create new virtual environment : `python3 -m venv CloGAN`
5. Activate the virtual environment : In Linux and macOS, run `source CloGAN/bin/activate` and in Windows, `.\CloGAN\Scripts\activate`
6. Install the required libraries in the virtual environment : `pip install -r requirements.txt`. Note that the tensorflow in the requirements.txt is CPU version.
7. Run the CloGAN_predict.py with : `python CloGAN_predict.py`

8. Input the file path of the chest X-ray image to be predicted. (There are two examples that you could use, taken from ChestXray-14: **resources/cardiomegaly.png** and **resources/effusion.png**)
9. The predictions with saliency maps will be generated.
10. Enter `exit` if you want to leave the app.

11. You can leave the virtual environment : `deactivate`
