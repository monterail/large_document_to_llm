import os
import sys
import json

from pypdf import PdfReader
from llama_index import Document, GPTListIndex


def extract_pdf_text(pdf_path):
    text_chunks = []
    pdf = PdfReader(pdf_path)
    for page in pdf.pages:
        text_chunks.append(page.extract_text())
    return text_chunks

def make_text_documents(text_chunks):
    documents = []
    for text in text_chunks:
        documents.append(Document(text))
    return documents


def make_gpt_list_index(documents):

    if os.path.exists('index.json'):
        with open('index.json', 'r') as f:
            dict = json.load(f)
        index = GPTListIndex.load_from_dict(dict)

    else:

        index = GPTListIndex.from_documents(documents)
        dict = index.save_to_dict()
        with open('index.json', 'w') as f:
            json.dump(dict, f)

    return index


def main(source, prompt):
    if not os.path.exists(source):
        print("File not found")
        return

    text_chunks = extract_pdf_text(source)

    documents = make_text_documents(text_chunks)

    index = make_gpt_list_index(documents)

    print(index)

    print("\n \n \n")

    response = index.query(prompt, mode="embedding")
    print(response)


if __name__ == '__main__':
    source = sys.argv[1]
    prompt = sys.argv[2]
    main(source, prompt)
