import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import math
from operator import itemgetter
import sys
import time
import itertools
import xml.etree.ElementTree as etree
import os
import pandas as pd
from sklearn.cluster import KMeans
from scipy.spatial.distance import euclidean
import heapq


WEIGHT_MODEL_MAP = {'TF':0,'DF':1,'TF-IDF':2}

USER_TXT_DESCRIPTOR_FILENAME = '../data/desctxt/devset_textTermsPerUser.txt'
IMAGE_TXT_DESCRIPTOR_FILENAME = '../data/desctxt/devset_textTermsPerImage.txt'
LOCATION_TXT_DESCRIPTOR_FILENAME = '../data/desctxt/devset_textTermsPerPOI.txt'
IMAGE_PATH = '../data/descvis/img/'

IMAGE_MODELS = ['CM','CM3X3','CN','CN3X3','CSD','GLRM','GLRM3X3','HOG','LBP','LBP3X3']

def get_entity_data_from_file(filename):
    """
    Takes the textual descriptor filename and returns the list of entities for file
    """
    f = open(filename)
    entity_list = f.readlines()
    return entity_list

def get_locations_from_metadata(filename):
    """
    Get the locations list from the location metadata file
    """
    location_map = {}
    location_ids = []
    location_text = []

    tree = etree.parse(filename)
    for node in tree.iter('topic'):
        topic_children = node.getchildren()
        for child in topic_children:
            #number = child.tag,
            if 'number' in child.tag:
                location_id = int(child.text)
                location_ids.append(location_id)
            if 'title' in child.tag:
                location_text.append(child.text)

    location_map = dict(zip(location_ids,location_text))
    return location_map

def get_entity_term_vector_map(entity_list,weight_model_index,is_location_query=False,location_map = {}):
    """
    Returns the list of (term,weight) for a given entity 
    """
    entity_ids = []
    entity_term_vector_dict = {}
    for entity_obj in entity_list:#TODO change name
        term_index = entity_obj.find('"')

        #Get the entity id and term_weights list from each entity
        entity_id = entity_obj[:term_index].strip()
        entity_term_weights_list = entity_obj[entity_obj.find('"'):].split(" ")

        if is_location_query:
            #In case of location entity, id will need to be fetched from the location_map
            for location_id,location_text in location_map.items():
                if location_text.startswith(entity_id.split(" ")[0]):
                    entity_id = location_id
                    break
        
        entity_ids.append(entity_id)
        terms = []
        weights = []
        for i,x in enumerate(entity_term_weights_list):
            if i % 4:
                weights.append(float(x))
            else:
                #remove the quotes from the string or call function
                y = x.replace('"','')
                terms.append(y)

        #Construct the term-weight vector for each entity for given weight_model(TF/DF/IDF)

        #print("Terms",terms)
        #print("weights",weights)


        selected_weights = []
        i = weight_model_index
        while i < len(weights):
            selected_weights.append(weights[i])
            i = i+3

        #print ("selected_weights",selected_weights)

        term_weight_vector = list(map(lambda X: (X[0],X[1]), list(zip(terms,selected_weights))))

        #print("TERM_WEIGHT",term_weight_vector)

        entity_term_vector_dict[entity_id] = term_weight_vector
        #break

    return entity_term_vector_dict

def get_cosine_similarity_score(vector_1,vector_2):
    """
    Return the cosine similarity score for the 2 vectors
    """
    vector_1_weights = [t[1] for t in vector_1]
    vector_2_weights = [t[1] for t in vector_2]

    vector_1_weights = np.array(vector_1_weights).reshape(1,-1)
    vector_2_weights = np.array(vector_2_weights).reshape(1,-1)

    return 1 - cosine_similarity(np.array(vector_1_weights),np.array(vector_2_weights)).item()

def get_vector_norm(vector):
    #vector_norm =  math.sqrt(sum(map(lambda x:x*x,vector)))
    return np.linalg.norm(vector)

def get_normalized_term_vector(term_vector):
    term_weights = [t[1] for t in term_vector]

    norm = get_vector_norm(term_weights)

    if norm:
        normalized_term_vector = [(a[0],a[1]/norm) for a in term_vector]
    else:
        normalized_term_vector = term_vector

    return normalized_term_vector

def updated_term_vector(input_vector,reference_vector):
    '''
    considering the input_vector 
    check the terms in the given entity_vector,
    if match keep it
    else add the term,weight as (term,0) and remove the terms from the vector not in given entity vector/
    '''
    dict1 = dict(input_vector)
    dict2 = dict(reference_vector)

    updated_entity_term_vector = []
    for k,v in dict2.items():
        if k in dict1:
            updated_entity_term_vector.append((k,v))

    dict3 = dict(updated_entity_term_vector)
    for k in dict1.keys():
        if k not in dict3:
            updated_entity_term_vector.append((k,0))

    return updated_entity_term_vector

def get_vector_distances(vector_a,vector_b):
    #Get the absolute difference of scores between 2 vectors
    vector_distances = []
    for a,b in zip(vector_a,vector_b):
        vector_distances.append((a[0],abs(a[1] - b[1])))
    return vector_distances

def get_entity_match_data(entity_term_vector_dict,given_entity_id,k):
    entity_match_data = []

    given_entity_term_vector = entity_term_vector_dict[given_entity_id]

    normalized_given_entity_term_vector = get_normalized_term_vector(given_entity_term_vector)

    for entity_id,entity_term_vector in entity_term_vector_dict.items():
        if entity_id == given_entity_id:
            continue

        updated_entity_term_vector = updated_term_vector(given_entity_term_vector,entity_term_vector)
        similarity_score = get_cosine_similarity_score(given_entity_term_vector,updated_entity_term_vector)

        normalized_entity_term_vector = get_normalized_term_vector(updated_entity_term_vector)

        term_value_differences = get_vector_distances(normalized_given_entity_term_vector,normalized_entity_term_vector)
        sorted(term_value_differences,key=itemgetter(1))

        if len(term_value_differences) > 3:
            highest_contributors = [term_value_differences[0][0],term_value_differences[1][0],term_value_differences[2][0]]
        else:
            highest_contributors = []

        entity_match_data.append((entity_id,similarity_score,highest_contributors))

    sorted(entity_match_data,key=itemgetter(1))

    return entity_match_data[0:k]

def get_centroid_of_location(location_dataset):
    # Get the centroid vector for given location using K Means clustering algorithm
    kmeans = KMeans(n_clusters=1)
    kmeans.fit(location_dataset)
    return kmeans.cluster_centers_.flatten()
    #TODO reducing dimensions such as to speedup the operation and select more relevant features.(mandatory for task5)
    pass

def get_location_data(filename):
    #Returns the dataframe for a given location file
    location_df = pd.read_csv(filename,header=None)
    return location_df

def get_all_location_datasets_per_model(image_model,input_location,location_map):
    #Returns the map of (location id -> location_dataset) for each location


    reverse_location_map = dict((v,k) for k,v in location_map.items())
    location_dataset_map = {}

    #iterate over descvis/img folder,  get the dataset for each location-model file
    location_model_files = os.listdir(IMAGE_PATH)
    location_model_files.sort()
    for l in location_model_files:
        if model in l and input_location not in l:
            location_dataset = get_location_data(os.path.join(IMAGE_PATH,l))
            location_name = l[:l.find(model)-1]
            location_id = reverse_location_map[location_name]
            location_dataset_map[location_id]=location_dataset

    return location_dataset_map

def get_euclidean_similarity_score(vector_a,vector_b):
    #Returns the euclidean distance between 2 vectors
    x = min(len(vector_a), len(vector_b))
    if len(vector_a) > len(vector_b):
        vector_a = vector_a[:x]
    else:
        vector_b = vector_b[:x]

    return euclidean(vector_a,vector_b)

def get_similar_image_pairs(location1_id,location2_id,location_1_dataset,location_2_dataset,model):
    '''
    Returns the image pairs which are similar between location1 and location2
    '''
    location_image_vector_map = {location1_id:{},location2_id:{}}

    #Get the images vectors for location using values of pandas
    location1_images_vectors = location_1_dataset.values.tolist()
    location2_images_vectors = location_2_dataset.values.tolist()

    for l1i in location1_images_vectors:
        image_id = l1i[0]
        location_image_vector_map[location1_id][image_id] = l1i[1:]

    for l2i in location2_images_vectors:
        image_id = l2i[0]
        location_image_vector_map[location2_id][image_id] = l2i[1:]

    img_to_img_score_map = {}

    #Iterate over the location1 images and location2 images and store in the map with ((image_1,image_2)->euclidean siimilarity score between 2 image vectors)
    for i in location_image_vector_map[location1_id].keys():
        for j in location_image_vector_map[location2_id].keys():
            img_to_img_score_map[(i,j)] = get_euclidean_similarity_score(location_image_vector_map[location1_id][i],location_image_vector_map[location2_id][j])

    #sort the obtained image to image score vector in desc order
    img_to_img_score_vector = sorted(img_to_img_score_map.items(),key = lambda kv:kv[1])

    image_pairs = [x[0] for x in img_to_img_score_vector[0:3]]

    return image_pairs

    #return [img_to_img_score_vector[0][0],img_to_img_score_vector[1][0],img_to_img_score_vector[2][0]]

def get_similar_locations_given_model(location_datasets,input_location_dataset,input_location_id,k,model,location_map):
    '''
    Returns the k top most similar locations for a given visual descriptor model for images
    '''

    location_match_data = []
    input_location_centroid_arry = get_centroid_of_location(input_location_dataset)

    reverse_location_map = dict((value,key) for key,value in location_map.items())

    location_dataset_map = {}

    for location_id,dataset in location_datasets.items():
        location_dataset_map[location_id] = dataset
        location_centroid_arry = get_centroid_of_location(dataset)

        #Apply Euclidean distance similarity between input location centroid and the other obtained centroid vectors
        index = min(len(input_location_centroid_arry), len(location_centroid_arry))
        if len(input_location_centroid_arry) > len(location_centroid_arry):
            input_location_centroid_arry = input_location_centroid_arry[:index]
        else:
            location_centroid_arry = location_centroid_arry[:index]

        similarity_score = get_euclidean_similarity_score(input_location_centroid_arry,location_centroid_arry)

        image_pairs = []

        location_match_data.append((location_id,similarity_score,image_pairs))
        #break
    #print ("location_match_data",location_match_data)

    sorted(location_match_data,key=itemgetter(1))

    location_match_data = location_match_data[0:k]

    #For the given location match data, find 3 similar images
    location_image_pair_map = {}

    for l in location_match_data:
        similar_image_pairs = get_similar_image_pairs(input_location_id,reverse_location_map[location_map[l[0]]],location_dataset_map[reverse_location_map[location_map[l[0]]]],input_location_dataset,model)
        #similar_image_pairs = get_similar_image_pairs(l1_l2_img_similarity_matrix)
        location_image_pair_map[l[0]] = similar_image_pairs
    
    for i,l in enumerate(location_match_data):
        location_match_data[i] = (l[0],l[1],location_image_pair_map[l[0]])

    return location_match_data



def get_location_model_vectors(location_map):
    #Each key is a location and value is a list of list of computed centroids for each location
    location_model_vector_map = {}

    reverse_location_map = dict((v,k) for k,v in location_map.items())

    for location in location_map.values():
        #Iterate over the descvis img and for each model of the each location get the location dataset and compute the Kmeans centroid.
        location_model_vector_map[reverse_location_map[location]] = []
        location_model_files = os.listdir(IMAGE_PATH)
        location_model_files.sort()
        for l in location_model_files:
            if location in l:
                location_dataset = get_location_data(os.path.join(IMAGE_PATH,l))
                location_centroid_arry = get_centroid_of_location(location_dataset)
                location_model_vector_map[reverse_location_map[location]].append(location_centroid_arry)

    return location_model_vector_map


def get_l1model_l2_model_similarity(location1_model_vector,location2_model_vector):
    '''
    Returns the similarities list for location1 and location2. Similarity measure used is Euclidean.
    '''
    l1model_l2model_similarities = []

    for l1,l2 in zip(location1_model_vector,location2_model_vector):
        similarity_score = get_euclidean_similarity_score(l1,l2)
        l1model_l2model_similarities.append(similarity_score)
    return l1model_l2model_similarities

def get_top_k_similar_locations(input_location_id,location_model_vector_map,k):
    '''
    Returns the top k similar locations based on each visual descriptor model contributions and similarity scores
    '''
    input_location_model_vector = location_model_vector_map[input_location_id]

    location_similarity_vector_map = {}

    top_k_similar_location_match_data = []

    for location_id,location_model_vector in location_model_vector_map.items():
        if input_location_id == location_id:
            continue
        else:
            #Get the location to location similarity based on model vector(computed using K Means centroid)
            location_similarity_vector_map[(input_location_id,location_id)] = get_l1model_l2_model_similarity(input_location_model_vector,location_model_vector)

    location_similarity_score_map = {}
    #Normalize the similarity vector for each l1,l2 pair.
    for l1_l2_id,sim_scores in location_similarity_vector_map.items():
        location_similarity_score_map[l1_l2_id] = get_vector_norm(sim_scores)

    location_similarities = sorted(location_similarity_score_map.items(),key = lambda kv:kv[1])

    # only_similarities = [x[1] for x in location_similarities]
    # print("Min similarity",min(only_similarities))

    k_most_similar_location_similarities = location_similarities[0:k]
    '''
    Iterate over the k most similar location similarities and compute model contribution using normalization for each model for given match
    Append the location-location pair(ids),similarity score and model contribution list to final returnable list
    '''
    for l in k_most_similar_location_similarities:
        model_contribution = [(x/get_vector_norm(location_similarity_score_map[l[0]])) for x in location_similarity_vector_map[l[0]]]
        top_k_similar_location_match_data.append((l[0][1],location_similarity_score_map[l[0]],model_contribution))

    return top_k_similar_location_match_data

if __name__ == '__main__':
    start = time.time()
    task_id = int(sys.argv[1])
    entity_id = sys.argv[2]
    model = sys.argv[3]
    if task_id == 5:
        k = int(sys.argv[3])
    else:
        k = int(sys.argv[4])

    location_map = get_locations_from_metadata('../data/devset_topics.xml')

    if task_id == 1:
        users = get_entity_data_from_file(USER_TXT_DESCRIPTOR_FILENAME)
        user_match_data = get_entity_match_data(get_entity_term_vector_map(users,WEIGHT_MODEL_MAP[model]),entity_id,k)
        print("Top k similar users based on textual descriptors")
        print (user_match_data)
    if task_id == 2:
        images = get_entity_data_from_file(IMAGE_TXT_DESCRIPTOR_FILENAME)
        image_match_data = get_entity_match_data(get_entity_term_vector_map(images,WEIGHT_MODEL_MAP[model]),entity_id,k)
        image_match_data = [(int(x[0]),x[1],x[2]) for x in image_match_data]
        print("Top k similar images based on textual descriptors")
        print (image_match_data)
    if task_id == 3:
        locations = get_entity_data_from_file(LOCATION_TXT_DESCRIPTOR_FILENAME)
        location_match_data = get_entity_match_data(get_entity_term_vector_map(locations,WEIGHT_MODEL_MAP[model],is_location_query=True,location_map=location_map),int(entity_id),k)
        location_match_data = [(location_map[x[0]],x[1],x[2]) for x in location_match_data]
        print("Top k similar locations based on textual descriptors")
        print(location_match_data)

    if task_id == 4:
        input_location_id = int(entity_id)
        input_location = location_map[input_location_id]
        input_location_model_filename = os.path.join(IMAGE_PATH,input_location + " " + model + ".csv")

        input_location_dataset = get_location_data(input_location_model_filename)

        other_location_datasets = get_all_location_datasets_per_model(model,input_location,location_map)
        
        similar_locations = get_similar_locations_given_model(other_location_datasets,input_location_dataset,input_location_id,k,model,location_map)

        similar_locations = [(location_map[x[0]],x[1],[(int(x[0]),int(x[1])) for x in x[2]]) for x in similar_locations]

        print("K similar locations based on visual descriptors given the model %s"%model)
        print(similar_locations)

    if task_id == 5:
        input_location_id = int(entity_id)
        input_location = location_map[input_location_id]

        location_model_vector_map = get_location_model_vectors(location_map)

        top_k_similar_locations = get_top_k_similar_locations(input_location_id,location_model_vector_map,k)
        top_k_similar_locations = [(location_map[x[0]],x[1],x[2]) for x in top_k_similar_locations]
        print ("Top k_similar_locations based on each visual descriptor model")
        print(top_k_similar_locations)
