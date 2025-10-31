SAR
=====
This is the code of the implementation of Patch-Agnostic Defense against Adversarial Patch Attacks

Setup
-----
conda create --name pad python=3.8<br>
conda activate sar<br>
pip install -r requirements.txt<br>

Usage
----
python run-SAR.py --invert-mask <br> 
Remember to replace the file path with your own <br>
input_path: the dir path of the attacked images <br>
save_path: the dir path where you want to save the defended images
