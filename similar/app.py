from data_loader import PlainDataLoader
from shingling import Shingles
from typing import Dict, List, Union
import pickle
import argparse
from get_similarity import Get_Similarity

def recommend(dataset, text: Union[str,List[str]], num:int = 3) -> List[List[str]]:
    recommended = []
    loader = PlainDataLoader()
    poems = loader.extract_full_poem(dataset)
    shingler = Shingles()
    text = text.split()
    text = shingler.shingling_sentence(text)
    if dataset not in ['songci', 'yuanqu', 'yudingquantangshi', 'chuci']:
        raise TypeError()
    with open(f"./models/{dataset}_model.pkl", "rb") as file:
        similarer = pickle.load(file)
    for id in similarer.find_similar_poem(text,num):
        recommended.append(''.join(poems[id]))

    return '\n'.join(recommended)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", type=str, help='words or sentences already written')
    parser.add_argument('--num', type=int, help='number of poems you want to recommend for you')
    parser.add_argument('--dataset', type=str, help='style you want')
    args = parser.parse_args()
    recommended = recommend(args.dataset, args.text, args.num)
    print(recommended)