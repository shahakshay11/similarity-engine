Project Title - Similarity Engine

Project Description
This project helps to fetch similar entities namely users,locations,images based on the textual descriptor models like TF,DF,TF-IDF
and visual descriptor models of images like CM,CN,CN3x3 to name a few. In this project we basically deal with fundamental concept of vectors
and vector spaces.

Structure & Setup
    A)Files - main.py, requirements.txt
        main.py - Consists of entire code for returning the k similar locations/images/users for task1 to task3 and similarity computations depending
        on the visual descriptor model for task 4 and task 5 respectively

        requirements.txt - Consists of all the libraries used in the code with respective versions
        scikit-learn, numpy, pandas the significant of all.


    B)It's recommended to setup a virtualenv with python 3.6
        Steps to setup for execution
        a) Create virtualenv with python 3.6
        b) Activate virtualenv
        c) Run pip install -r requirements.txt

Running instructions
    Since we have 5 tasks to implement, we use task_id in the comammnd line argument as starting followed by task specific input which
    is explained as follows and task specific parameters will be followed by the task id

    Commandline arguments (task specific)
    a)Task1
        task_id,user_id,model,k
        Example - 1 39052554@N00 TF 5
    b)Task2
        task_id,image_id,model,k
        Example - 2 9067738157 TF 5
    c)Task3
        task_id,location_id,model,k
        Example - 3 27 TF 5
    d)Task4
        task_id,location_id,model,k
        Example - 4 10 CN3x3 7
    e)Task5
        task_id,location_id,k
        Example - 5 4 5

Libraries & Dependencies
    1) scikit-learn  version- 0.19.2
        a) Used for computing cosine similarity between 2 given vectors
        b) Used for computing the centroid using KMeans clustering algorithm
    2) numpy version-1.15.1
        Used for handling vectors and norm of vectors
    3) pandas version-0.23.4
        Used for reading the data into the dataframe
    4) scipy version-1.1.0
        Used for calculating the Euclidean distance
