import torch
import torch.nn.functional as F
import numpy as np


def get_similarity_matrix(embeddings_tensor: torch.Tensor):
    num_of_speaker = embeddings_tensor.shape[0]
    num_utterance_for_a_speaker = embeddings_tensor.shape[1]
    # dim = 1 because  we assume , dim = 1 is the uttance_number axis
    centroids_matrix = torch.mean(embeddings_tensor, dim=1)

    # cloning is needed for reverse differenciation
    centroids_matrix = centroids_matrix.clone() / \
        (torch.norm(centroids_matrix, dim=2, keepdim=True) + 0.0000001)
    similarity_matrix = torch.zeros(
        num_of_speaker, num_utterance_for_a_speaker, num_of_speaker)

    for x1 in range(num_of_speaker):
        for x2 in range(num_utterance_for_a_speaker):
            for y in range(num_of_speaker):
                similarity_matrix[x1][x2][y] = np.dot(
                    centroids_matrix[y], embeddings_tensor[x1][x2])

    return similarity_matrix


def softmax_loss(embeddings_tensor: torch.Tensor):
    num_of_speaker = embeddings_tensor.shape[0]
    num_utterance_for_a_speaker = embeddings_tensor.shape[1]
    similarity_matrix = get_similarity_matrix(embeddings_tensor)

    loss_matrix = torch.zeros(
        num_of_speaker, num_utterance_for_a_speaker)
    # dim=1 refers to utterence number
    # dim=0 refers to speaker number

    sum_of_exps = torch.zeros(
        num_of_speaker, num_utterance_for_a_speaker)
    for k in range(num_of_speaker):
        sum_of_exps += np.exp(similarity_matrix[:][:][k])
    np.log(sum_of_exps)

    loss_matrix = torch.zeros(
        num_of_speaker, num_utterance_for_a_speaker)
    for j in range(num_of_speaker):
        for i in range(num_utterance_for_a_speaker):
            loss_matrix[j][i] = -similarity_matrix[j][i][j] + \
                np.log(sum_of_exps[j][i])

    return loss_matrix


def contrast_loss(embeddings_tensor: torch.Tensor):
    num_of_speaker = embeddings_tensor.shape[0]
    num_utterance_for_a_speaker = embeddings_tensor.shape[1]
    similarity_matrix = get_similarity_matrix(embeddings_tensor)

    loss_matrix = torch.zeros(
        num_of_speaker, num_utterance_for_a_speaker)
    # dim=1 refers to utterence number
    # dim=0 refers to speaker number

    loss_matrix = torch.zeros(
        num_of_speaker, num_utterance_for_a_speaker)
    for j in range(num_of_speaker):
        for i in range(num_utterance_for_a_speaker):
            max_cheker = F.sigmoid(similarity_matrix[j][i]).clone()
            max_cheker = torch.cat([max_cheker[:j], max_cheker[j+1:]])
            #  reference  https://discuss.pytorch.org/t/how-to-remove-an-element-from-a-1-d-tensor-by-index/23109/2

            loss_matrix[j][i] = 1 - \
                F.sigmoid(similarity_matrix[j][i][j]
                          ) + torch.max(max_cheker)

    return loss_matrix
