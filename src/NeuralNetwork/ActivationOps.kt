package NeuralNetwork

import kotlin.math.exp

class ActivationOps {

    class Sigmoid : Activation() {

        override fun call(x: Matrix) : Matrix {
            val activation = Matrix( x.m , x.n )
            for ( i in 0 until x.m ) {
                for ( j in 0 until x.n ) {
                    activation.set( i , j , sigmoid_( x.get( i , j ) ) )
                }
            }
            return activation
        }

        override fun gradient(x: Matrix): Matrix {
            val gradient = Matrix( x.m , x.n )
            for ( i in 0 until x.m ) {
                for ( j in 0 until x.n ) {
                    gradient.set( i , j , sigmoidGradient_( x.get( i , j ) ) )
                }
            }
            return gradient
        }

        private fun sigmoid_( x : Double ) : Double {
            return 1.0 / ( 1.0 + exp( -x ) )
        }
        private fun sigmoidGradient_( x : Double ) : Double {
            return sigmoid_(x) * ( 1.0 - sigmoid_(x) )
        }

    }

    class ReLU : Activation() {

        override fun call(x: Matrix): Matrix {
            val activation = Matrix( x.m , x.n )
            for ( i in 0 until x.m ) {
                for ( j in 0 until x.n ) {
                    activation.set( i , j , relu_( x.get( i , j ) ) )
                }
            }
            return activation
        }

        override fun gradient(x: Matrix): Matrix {
            val gradient = Matrix( x.m , x.n )
            for ( i in 0 until x.m ) {
                for ( j in 0 until x.n ) {
                    gradient.set( i , j , reluGradient_( x.get( i , j ) ) )
                }
            }
            return gradient
        }

        private fun relu_( x : Double ) : Double {
            return if ( x > 0.0 ){ x } else { 0.0 }
        }
        private fun reluGradient_( x : Double ) : Double {
            return if ( x > 0.0 ){ 1.0 } else { 0.0 }
        }

    }

}