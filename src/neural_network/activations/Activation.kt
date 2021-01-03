package neural_network.activations

import neural_network.Matrix

abstract class Activation {

    abstract fun call( x : Matrix) : Matrix
    abstract fun gradient( x : Matrix): Matrix

}