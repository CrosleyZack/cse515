#! /bin/usr/python3.6
from loader import Loader
from distance import Similarity
from graph import Graph
from os.path import abspath, isdir, isfile
import argparse
from util import timed
import numpy as np
import numpy.linalg as la
import scipy.cluster.vq as vq
from scipy.sparse import csc_matrix
import util

class Interface():

    def __init__(self):
        self.__database__ = None
        self.__graph__ = None 
        self.__valid_types__ = ['photo', 'user', 'poi']
        self.__vis_models__ = ['CM', 'CM3x3', 'CN', 'CN3x3',
                'CSD', 'GLRLM', 'GLRLM3x3', 'HOG', 'LBP', 'LBP3x3']
        self.__io__()
    

    def __io__(self):
        print("Welcome to the CSE 515 data software. Please enter a command.\
              \nEnter \"help\" for a list of commands.")

        def __filepath__(parser, arg):
            if not isdir(arg):
                parser.error("The directory %s doesn't exist." %arg)
            else:
                return True

        parser = argparse.ArgumentParser()
        parser.add_argument('-task', type=int, choices=range(1,7), required=True, metavar='#')
        parser.add_argument('--k', type=int, metavar='#')
        parser.add_argument('--alg', type=str, metavar="algorithm_to_use")
        parser.add_argument('--imgs', type=int, nargs='+', metavar='img1 img2 ...')
        parser.add_argument('--load', type=str, metavar='filepath')
        parser.add_argument('--graph', type=str, metavar='filename')
        parser.add_argument('--layers', type=int, metavar='L')
        parser.add_argument('--hashes', type=int, metavar='k')
        # parser.add_argument('--cluster', type=int, metavar='c')
        parser.add_argument('--vectors', type=str) #Assuming this is a file locaiton 
        while True:
            user_input = input("\nEnter a Command:$ ")
            user_input = user_input.split(' ')
            try:
                args = parser.parse_args(user_input)
            except Exception as e:
                print("The command line input could not be parsed.")
                print(e)

            # load the database from the folder.
            if args.load:
                try:
                    self.load(args)
                except Exception as e:
                    print('Something went wrong during database load.')
                    print(e)

            # load the graph from the file.
            if args.graph:
                try:
                    self.graph(args)
                except Exception as e:
                    print('Something went wrong loading the graph.')
                    print(e)

            # Get method with this name - makes it easy to create new interface methods 
            #   without having to edit this method.
            try:
                method = getattr(self, 'task' + str(args.task))
            except AttributeError:
                print('The command specified was not a valid command.\n')
            
            try:
                method(args)
            except Exception as e:
                print('Something went wrong while executing ' + str(method.__name__))
                print(str(e))



    def help(self, args):
        """
        Command:\thelp
        Description:\tPrints the interface information about the program.
        """

        print("The following are valid commands to the program.")
        for item in dir(self):
            if not item.startswith('__'):
                method = getattr(self, item)
                print(method.__doc__)
        

    def load(self, args):
        """
        Command:\t--load <filepath>
        Description:\tLoads the database at that file path. If the file does not exist, will prompt to create a new database using the folder of a users choice.
        Arguments:
        \t<filepath> - A valid file path in the system.
        """
        
        folder = abspath(args.load)

        if not isdir(folder):
            print("[ERROR] The provided path was not a folder, and therefore not a valid data directory.")
            return
        
        self.__database__ = Loader.make_database(folder)
        print("Database loaded successfully.")
    


    def graph(self, args):
        """
        Command:\t--graph <filepath>
        Description:\tLoads the graph from the binary (pickle file) specified.
        Arguments:
        \t<filepath> a valid file path in the system.
        """
        f = abspath(args.graph)

        if not isfile(f):
            print("[ERROR] The provided path was not a valid file.")
            return
        
        self.__graph__ = Graph.load(f)
        print('Graph loaded successfully.')



    @timed
    def task1(self, args):
        if args.k == None:
            raise ValueError('Parameter K must be defined for task 1.')
        k = int(args.k)
        if self.__graph__ == None:
            self.__graph__ = Loader.make_graphs(self.__database__, k)
        # visualize graph.
        self.__graph__.display()
    


    def task2(self, args):
        if args.k == None:
            raise ValueError('K must be defined for task 2.')
        c = int(args.k)
        # alg = args.alg

        #YOUR CODE HERE.
        clusters = {}
        clusters1 = {}
        images = self.__graph__.get_images()
        list_of_clusters = []
        list_of_clusters1 = []
        lengOfA = 0
        lengOfB = 0
        lengOfClusters = {}
        A = self.__graph__.get_adjacency()
        D = np.diag(np.ravel(np.sum(A,axis=1)))
        L = D - A
        l, U = la.eigh(L)
        f = U[:, 1]
        labels = np.ravel(np.sign(f))
        # Clustering function
        for image in images:

            if(labels[images.index(image)] == -1):
                cluster = 'A'
                clusters[image] = cluster
                lengOfA += 1

                if not cluster in list_of_clusters:
                    list_of_clusters.append(cluster)
            else:
                cluster = 'B'
                clusters[image] = cluster
                lengOfB += 1
                if not cluster in list_of_clusters:
                    list_of_clusters.append(cluster)

        # display
        for image in images:
            self.__graph__.add_to_cluster(image, clusters[image])
        self.__graph__.display(clusters=list_of_clusters, filename='task2.png')
        print("Clusters in A:", lengOfA)
        print("Clusters in B:", lengOfB)

        xx = c+1
        means, labels1 = vq.kmeans2(U[:, 1:xx], c, iter= 500)
        for j in range(c):
            indices = [i for i, x in enumerate(labels1) if x == j]
            lengOfClusters[j] = len(indices)
            for everyIndice in indices:
                cluster = j
                clusters1[images[everyIndice]] = cluster
            if not j in list_of_clusters1:
                list_of_clusters1.append(j)
        for image in images:
            self.__graph__.add_to_cluster(image, clusters1[image])
        self.__graph__.display(clusters=list_of_clusters1, filename='task2_kspectral.png')
        print(lengOfClusters)




    def task3(self, args):
        if args.k == None:
            raise ValueError('K must be defined for task 3.')
        k = int(args.k)

        # YOUR CODE HERE.
        G = self.__graph__.get_adjacency()
        n = G.shape[0]
        s = 0.86
        maxerr = 0.0001

        # transform G into markov matrix A
        A = csc_matrix(G, dtype=np.float)
        rsums = np.array(A.sum(1))[:, 0]
        ri, ci = A.nonzero()
        A.data /= rsums[ri]

        # bool array of sink states
        sink = rsums == 0

        # Compute pagerank r until we converge
        ro, r = np.zeros(n), np.ones(n)
        while np.sum(np.abs(r - ro)) > maxerr:
            ro = r.copy()
            # calculate each pagerank at a time
            for i in range(0, n):
                # in-links of state i
                Ai = np.array(A[:, i].todense())[:, 0]
                # account for sink states
                Di = sink / float(n)
                # account for teleportation to state i
                Ei = np.ones(n) / float(n)

                r[i] = ro.dot(Ai * s + Di * s + Ei * (1 - s))

        # weights = r / float(sum(r))
        # weightDict = {}
        # for xx in range(len(weights)):
        #     weightDict[xx] = weights[xx]
        pass

    def task4(self, args):
        if args.k == None or args.imgs == None:
            raise ValueError('K and Imgs must be defined for task 4.')
        k = int(args.k)
        imgs = list(args.imgs)
    
        # YOUR CODE HERE.


    
    def task5(self, args):
        if args.layers == None or args.hashes == None \
            or args.vectors == None or args.k == None \
            or args.imgs == None:
            raise ValueError('Layers, Hashes, Vectors, K, and IMG must all be defined for task 5.')
        layers = int(args.layers)
        hashes = int(args.hashes)
        vectors = str(args.vectors)
        t = int(args.k)
        imgs = list(args.imgs)
    
        # YOUR CODE HERE
    


    def task6(self, args):
        if args.alg == None:
            raise ValueError('Alg must be defined for task 6.')
        
        alg = str(args.alg)

        # YOUR CODE HERE



    def quit(self, *args):
        """
        Command:\tquit
        Description:\tExits the program and performs necessary cleanup.
        """
        exit(1)

if __name__ == '__main__':
    Interface()