# MNIST Digit Recognizer (PyTorch)

Простой CNN-классификатор для распознавания рукописных цифр (0–9) на базе PyTorch.

##  Возможности
- Обучает собственную сверточную сеть на MNIST
- Поддерживает предсказание любых PNG/JPG (любой размер)
- Автоматическая нормализация в формат MNIST (инвертирование, центрирование, 20×20 → 28×28)
- Сохранение модели в `models/mnist_cnn.pth`

##  Установка

pip install -r requirements.txt

## Обучение модели 

python train.py

## После обучения появится файл:
- models/mnist_cnn.pth 

## Использование для предсказаний
python predict.py images/one.png


## Архитектура модели

- Conv2d(1 → 32, kernel=3)
- Conv2d(32 → 64, kernel=3)
- MaxPool2d(2×2)
- Linear(64×12×12 → 128)
- Linear(128 → 10)

Активация: ReLU  
Функция потерь: CrossEntropyLoss  
Оптимизатор: Adam


## Структура проекта

lox1/
│── model.py
│── train.py
│── predict.py
│── requirements.txt
│── README.md
│
├── models/
│     └── mnist_cnn.pth
├── images/
│     ├── one.png
│     ├── two.jpg
│     └── ...
└── data/   # скачивается автоматически, коммитить не нужно


## Структура проекта

lox1/
│── model.py
│── train.py
│── predict.py
│── requirements.txt
│── README.md
│
├── models/
│     └── mnist_cnn.pth
├── images/
│     ├── one.png
│     ├── two.jpg
│     └── ...
└── data/   # скачивается автоматически, коммитить не нужно


## Лицензия

MIT License — можно использовать свободно.
