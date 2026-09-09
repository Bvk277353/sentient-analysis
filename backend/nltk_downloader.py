import os
import nltk

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

NLTK_DATA_DIR = os.path.join(
    BASE_DIR,
    "nltk_data"
)

os.makedirs(
    NLTK_DATA_DIR,
    exist_ok=True
)

resources = [
    "stopwords",
    "punkt",
    "punkt_tab",
    "wordnet"
]

for resource in resources:

    print(f"Downloading {resource}...")

    nltk.download(
        resource,
        download_dir=NLTK_DATA_DIR
    )

print("NLTK resources downloaded successfully.")
