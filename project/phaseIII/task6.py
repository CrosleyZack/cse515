import operator
from distance import Similarity
from distance import Distance
import pandas as pd
from util import timed, show_images


class KNN:

    def get_neighbors(self, labelled_set, labels, imageInstance, k):
        distances = []
        j = 0
        for x in range(len(labelled_set)):
            # dist = Similarity.cos_similarity(imageInstance, labelled_set[x])
            dist = Distance.E2_distance(imageInstance, labelled_set[x])
            distances.append((labelled_set[x], labels[j], dist))
            j += 1
        distances.sort(key=operator.itemgetter(2), reverse=False)
        neighbors = []
        for x in range(k):
            neighbors.append(distances[x][1])
        # print("neighbors:" + str(neighbors))
        return neighbors

    def get_response(self, neighbors):
        class_votes = {}
        for x in range(len(neighbors)):
            response = neighbors[x]
            if response in class_votes:
                class_votes[response] += 1
            else:
                class_votes[response] = 1
        sorted_votes = sorted(class_votes.items(), key=operator.itemgetter(1), reverse=True)
        return sorted_votes[0][0]

    def get_labelled_set(self, imageIDs, image_dict):
        labelled_set = []
        for imageID in imageIDs:
            labelled_set.append(image_dict[int(imageID)])
        return labelled_set

    @timed
    def knn_algorithm(self, imageIds, labels, k, database):
        image_dict = pd.DataFrame(database.get_vis_table())
        image_dict = image_dict.T
        image_dict = image_dict.to_dict('list')

        labelled = {}
        for j, imageId in enumerate(imageIds):
            labelled[imageId] = labels[j]
            # labelled.append({imageId: labels[j]})
        # print(labelled)

        labelled_set = self.get_labelled_set(imageIds, image_dict)
        print("Working")
        for v, image in enumerate(image_dict):
            image = int(image)
            if image not in labelled:
                neighbors = self.get_neighbors(labelled_set, labels, image_dict[image], k)
                result = self.get_response(neighbors)
                # print(str(v) + " labelled as :" + str(result))
                labels.append(result)
                imageIds.append(image)
                labelled[image] = result
                # labelled.append({image: result})
                # labelled_set.append(image_dict[image])

        return labelled

    def main(self):
        k = 3
        imageIDs = ['3298433827', '299114458', '948633075', '4815295122', '5898734700', '4027646409', '1806444675',
                    '4501766904', '6669397377', '3630226176', '3630226176', '3779303606', '4017014699']
        labels = ['fort', 'sculpture', 'sculpture', 'sculpture', 'sculpture', 'fort', 'fort', 'fort', 'sculpture',
                  'sculpture', 'sculpture', 'sculpture', 'sculpture']
        '''
        j = 0
        for i in args:
            if j % 2 == 0:
                imageIDs.append([i])
            else:
                labels.append([i])
            j = j + 1
        '''
        result = self.knn_algorithm(imageIDs, labels, k, database=())
        print("result: " + str(result))


if __name__ == '__main__':
    knn = KNN()
    knn.main()
