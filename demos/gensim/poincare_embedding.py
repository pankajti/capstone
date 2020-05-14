from gensim.models.poincare import PoincareModel
relations = [('kangaroo', 'marsupial'), ('kangaroo', 'mammal'), ('gib', 'cat')]
model = PoincareModel(relations, negative=2)
model.train(epochs=50)

print("model")