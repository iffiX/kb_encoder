sudo apt install -y python3.8-venv libhdf5-dev libpython3-dev graphviz ninja
python3 -m venv venv
venv/bin/pip install torch==1.10.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
venv/bin/pip install wheel
venv/bin/pip install -r requirements.txt
venv/bin/python -m spacy download en_core_web_sm
venv/bin/python -m spacy download en_core_web_md
venv/bin/python -c 'import nltk; nltk.download("punkt");nltk.download("averaged_perceptron_tagger");nltk.download("wordnet");nltk.download("omw-1.4")'