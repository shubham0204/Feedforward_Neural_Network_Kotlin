package NeuralNetwork

import kotlin.math.exp

abstract class Activation {

    abstract fun call( x : Matrix ) : Matrix
    abstract fun gradient( x : Matrix ): Matrix

}