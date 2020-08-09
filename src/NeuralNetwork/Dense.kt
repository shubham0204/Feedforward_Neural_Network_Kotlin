
package NeuralNetwork

class Dense(
    var units : Int ,
    var activation : Activation,
    var requiresBias : Boolean = false
) {

    var X : Matrix? = null
    var W : Matrix? = null
    var B : Matrix = MatrixOps.onesLike( 1 , units )
    var y : Matrix? = null
    var theta : Matrix? = null
    var dy_dtheta : Matrix? = null
    var dtheta_dW : Matrix? = null
    var dtheta_dX : Matrix? = null

    fun initWeights( inputDims : Int ){
        W = MatrixOps.uniform( inputDims , units )
    }

    fun forward( inputs : Matrix ) : Matrix {
        X = inputs
        if ( requiresBias ) {
            theta = MatrixOps.dot( inputs , W!! ) + B
        }
        else {
            theta = MatrixOps.dot( inputs , W!! )
        }
        y = activation.call( theta!! )
        return y!!
    }


    fun computeGradients() {
        dy_dtheta = activation.gradient( theta!! )
        dtheta_dW = X
        dtheta_dX = W
    }

}

