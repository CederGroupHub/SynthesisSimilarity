import tensorflow as tf
import numpy as np

class CircleLoss(tf.keras.losses.Loss):

    def __init__(self,
                 gamma: int = 64,
                 margin: float = 0.25,
                 reduction='auto',
                 name=None):
        super().__init__(reduction=reduction, name=name)
        self.gamma = gamma
        self.margin = margin
        self.O_p = 1 + self.margin
        self.O_n = -self.margin
        self.Delta_p = 1 - self.margin
        self.Delta_n = self.margin

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """ NOTE : y_pred must be cos similarity/dot similarity/logit

        Args:
            y_true (tf.Tensor): shape [None,num_total_labels] one-hot matrix (0, 1)
            y_pred (tf.Tensor): shape [None,num_total_labels] float logits

        Returns:
            tf.Tensor: loss
        """

        alpha_p = tf.nn.relu(self.O_p - tf.stop_gradient(y_pred))
        alpha_n = tf.nn.relu(tf.stop_gradient(y_pred) - self.O_n)
        # yapf: disable
        y_true = tf.cast(y_true, tf.float32)

        # (None, num_total_labels)
        logit_p = - y_true * (alpha_p * (y_pred - self.Delta_p)) * self.gamma
        # minus 10000.0 to make contribution from mask is zero
        logit_p = logit_p - (1.0 - y_true) * 10000.0
        # (None, )
        loss_p = tf.reduce_logsumexp(logit_p, axis=-1)

        # (None, num_total_labels)
        logit_n = (1.0 - y_true) * (alpha_n * (y_pred - self.Delta_n)) * self.gamma
        logit_n = logit_n - y_true * 10000.0
        # (None, )
        loss_n = tf.reduce_logsumexp(logit_n, axis=-1)

        loss = tf.math.softplus(loss_p+loss_n)

        return loss


if __name__ == "__main__":
    batch_size = 2
    nclass = 5
    y_pred = tf.random.uniform((batch_size, nclass), -1, 1, dtype=tf.float32)

    # y_true = tf.random.uniform((batch_size,), 0, nclass, dtype=tf.int32)
    # y_true = tf.one_hot(y_true, nclass, dtype=tf.float32)
    y_true = tf.constant([[1,0,0,1,1], [0,0,1,0,1]])

    mycircleloss = CircleLoss()

    print(
        'mycircleloss:\n',
        mycircleloss.call(y_true, y_pred).numpy(),
        mycircleloss(y_true, y_pred)
    )