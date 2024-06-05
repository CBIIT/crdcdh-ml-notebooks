from sklearn.manifold import TSNE
from matplotlib import pylab

def plot_word_vector_tsne(word_vectors, word_labels, perplexity, filename):
    """
    Plot 2D word vectors TSNE.
    @param word_vectors: word vectors
    @param word_labels: word labels
    @param perplexity: 
    """
    tsne = TSNE(perplexity=perplexity, n_components=2, init="pca", n_iter=10000)
    two_d_embeddings = tsne.fit_transform(word_vectors)
    pylab.figure(figsize=(20, 20))
    for i, label in enumerate(word_labels):
        x, y = two_d_embeddings[i, :]
        pylab.scatter(x, y)
        pylab.annotate(
            label, xy=(x, y), xytext=(5, 2), textcoords="offset points", ha="right", va="bottom"
        )
   
    pylab.savefig(filename)
    pylab.show()