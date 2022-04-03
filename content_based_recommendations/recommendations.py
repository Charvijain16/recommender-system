import ast
import torch
from sklearn.metrics.pairwise import cosine_similarity


def get_key(val, dict):
    for key, value in dict.items():
        if val == value:
            return key
    return "key doesn't exist"

def similarity(iri='http://www.gra.fo/schema/untitled-ekg#topio.another-company.183.VECTOR', model='RotatE', number_of_recommendations=3):
    if model == 'RotatE':
        path = "content_based_recommendations/EmbeddingModels/results_official/resultsRotatE/"
    elif model == 'TransH':
        path = "content_based_recommendations/EmbeddingModels/results_official/resultsTransH/"
    # IRI = extractIRI(dataset)   # remove this

    entity_ids = open(path + "entities_ids.txt", "r")
    contents1 = entity_ids.read()
    entity = ast.literal_eval(contents1)
    id = entity[iri]

    # ToDO : Softcode these values
    all_datasets_ids = [144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157]

    model = torch.load(path + 'trained_model.pkl')
    entity_embeddings = model.entity_representations[0]
    original = entity_embeddings(torch.as_tensor(id)).detach().numpy()
    d = dict.fromkeys(all_datasets_ids)
    for i in range(len(all_datasets_ids)):
        embdding = entity_embeddings(torch.as_tensor(all_datasets_ids[i])).detach().numpy()
        # print("Is embedding complex(real and imaginary) in nature?", np.iscomplexobj(embdding))  # -> False
        cos_sim = cosine_similarity(original.reshape(1, -1), embdding.reshape(1, -1))
        d[all_datasets_ids[i]] = cos_sim
    # print(d)
    recommended_ids = sorted(d, key=d.get, reverse=True)[1:number_of_recommendations+1]

    recommended_iris = []
    for i in range(len(recommended_ids)):
        id = recommended_ids[i]
        iri = get_key(id, entity)
        recommended_iris.append(iri)

    # return recommended_iris
    return recommended_ids


# def extractIRI(name):               # if dataset name is given instead of iri
#     str = ".csv"
#     file_name = name.__add__(str)
#     path = "/home/cjain/PycharmProjects/RS_2021/data/Datasets/"
#     file_path = path.__add__(file_name)
#     df = pd.read_csv(file_path)
#     IRI = df.iat[0, 0]
#     return IRI

if __name__ == "__main__":
    # # dataset = "Grundwasserkörper, NGP 2009, Österreich"
    # iri = 'https://data.inspire.gv.at/e76c1db4-69ee-4252-aa0e-c7a65cf069f9'
    similarity(iri='http://www.gra.fo/schema/untitled-ekg#topio.another-company.183.VECTOR', model='RotatE',
               number_of_recommendations=4)

