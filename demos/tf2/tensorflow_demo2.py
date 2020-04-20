from tensorflow import keras
import tensorflow as tf


class MyDenseLayer(tf.keras.layers.Layer):
  def __init__(self, num_outputs):
    super(MyDenseLayer, self).__init__(num_outputs)
    self.num_outputs = num_outputs

  def build(self, input_shape):
    self.kernel = self.add_weight("kernel",
                                  shape=[int(input_shape[-1]),
                                         self.num_outputs])

  @tf.function
  def call(self, input):
    return tf.matmul(input, self.kernel)

layer1 = MyDenseLayer(10)
layer2 = MyDenseLayer(20)


model = keras.Sequential(layer1)
model.add(layer2)

#model.add(keras.layers.Dense(4))

model.compile(optimizer="adam", loss="binary_crossentropy")
#model.compile()
res  = model.fit(tf.ones(shape=(2,4)),tf.zeros(shape=(2)))

print(res)

