NOTES
python version min 3.7.7

- install => python3 -m venv yolo8_env
- copy this code => source yolo8_env/bin/activate
- install => python3 -m pip install --upgrade pip
- install => pip3 install ultralytics roboflow
- download dataSet di sini => https://universe.roboflow.com/nike-prototype/nike-prototype-q3j7o
- extract, and change name folder "dataSet"
- copy main.yaml on folder dataSet to your path project

E X A M P L E
my-project/

├── dataSet/

│   ├── README.dataset.txt

│   ├──README.roboflow.txt

│   ├── test/

│   ├── train/

│   ├── valid/

├── yolo8_env/

├── .gitignore

├── data.yaml

├── main.py

└── trainDataSet

- if your first clone this project, you must run file trainDataset.py => "python3 trainDataset"
- to test via webcam for object detection, please run main.py => "python3 main.py"

