package neural_network.loss

import neural_network.Matrix

class MeanSquaredError  : Loss() {

    override fun computeLoss(y_true: Matrix, y_pred: Matrix): Matrix {
        TODO("Not yet implemented")
    }

    override fun computeLossGradient(y_true: Matrix, y_pred: Matrix): Matrix {
        return y_pred - y_true
    }
}