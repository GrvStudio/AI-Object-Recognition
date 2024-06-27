NOTES
python version 3.12.4

- install => python3 -m venv yolo8_env
- copy this code => source yolo8_env/bin/activate
- install => python3 -m pip install --upgrade pip
- install => pip3 install ultralytics roboflow
- download dataSet di sini => https://universe.roboflow.com/rahmad-iqbal-guf2m/nike-prototype
- extract, and change name folder "dataSet"
- copy main.yaml on folder dataSet to your path project
example

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

