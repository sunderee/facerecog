# facerecog

`facerecog` is a facial recognition system developed in Python, heavily based on work
by [Ars Futura](https://github.com/arsfutura/face-recognition).

**Usage:** you need to provide folders with PNG images of at least two people inside `assets` directory

```
|-- assets
|   |-- first_person
|   |-- second_person
```

enumerating them. For instance, `first_person` would contain images with names `first_person_1.png`
, `first_person_2.png`, etc. You will be comparing the results against `test.png`, which should located directly
inside `assets` folder.

Next, create a virtual environment, install dependencies and run the program.

```bash
# Create virtual environment and install dependencies
python3 -m venv venv
source venv/bin/activate
pip3 install -r requirements.txt

# Run the training and/or classification task
sh train-task.sh
sh classify-task.sh <directory-of-test-image>

# ...or just call from main.py
python3 main.py --only-train
python3 main.py --dir=<directory-of-test-image>

# Deactivate virtual environment when you're done
deactivate
```

For the first run, you need the `train-task.sh` in order to build the model.
