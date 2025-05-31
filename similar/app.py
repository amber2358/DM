from data_loader import PlainDataLoader
from shingling import Shingles
from typing import Dict, List, Union
import pickle

def recommend(dataset, text: Union[str,List[str]], num:int = 3) -> List[List[str]]:
    recommended = []
    loader = PlainDataLoader()
    poems = loader.extract_full_poem(dataset)
    shingler = Shingles()
    text = shingler.shingling_sentence(text)
    with open(f"./models/{dataset}_model.pkl", "rb") as file:
        similarer = pickle.load(file)
    for id in similarer.find_similar_poem(text,num):
        recommended.append(poems[id])

    return recommended

if __name__ == '__main__':
    text = ...   # words or sentences already written
    num = ...    # number of poems you want to recommend for you
    dataset = ...   # style you want
    recommended = recommend(dataset, text, num)
