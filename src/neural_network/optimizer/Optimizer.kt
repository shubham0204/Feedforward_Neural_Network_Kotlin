package neural_network.optimizer

import neural_network.Dense
import neural_network.Matrix
import neural_network.loss.Loss

abstract class Optimizer {

    abstract fun optimize( layers : Array<Dense> , loss : Loss , y_pred : Matrix , y_true : Matrix ) : Array<Dense>
    abstract fun resetState()

}