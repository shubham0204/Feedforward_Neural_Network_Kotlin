package neural_network.loss

import neural_network.Matrix

abstract class Loss {

    abstract fun computeLoss( y_true : Matrix , y_pred : Matrix ) : Matrix
    abstract fun computeLossGradient( y_true : Matrix , y_pred : Matrix ) : Matrix

}