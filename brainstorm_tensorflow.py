from karoo.modules.plagih_sympy_extras import plagih_sympify
import tensorflow as tf

print(plagih_sympify('Min(0, 1)'))

const_a = tf.constant(2, dtype=tf.float32)
const_b = tf.constant(5, dtype=tf.float32)

min_1 = tf.math.reduce_min([const_a, const_b, const_b])

print(min_1)

