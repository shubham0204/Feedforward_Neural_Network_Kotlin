package NeuralNetwork

class Model( private var inputDims : Int , var layers : Array<Dense> ) {

    private var y : Matrix? = null
    private var y_hat : Matrix? = null
    private val alpha : Double = 0.1

    fun compile() {
        var inputDimForLayer = inputDims
        for( i in layers.indices ){
            layers[ i ].initWeights( inputDimForLayer )
            inputDimForLayer = layers[ i ].units
        }
    }

    fun forward( inputs : Matrix , labels : Matrix ) : Matrix {
        var layerInput = inputs
        //println( layers[2].W )
        for ( i in layers.indices ){
            //println( "Model $i")
            //println( "Model $i layerInput $layerInput")
            val theta = layers[ i ].forward( layerInput )
            //println( "THETA $theta")
            layerInput = theta
        }
        //println( "output $layerInput")
        y = layerInput
        y_hat = labels
        return y!!
    }

    fun backward() {
        layers.reverse()
        val dJ_dyN = lossGradient( y!! , y_hat!! )
        //println( dJ_dyN )
        layers.forEach{ layer -> layer.computeGradients() }

        var dJ_dtheta : Matrix?
        var dtheta_dX : Matrix?

        val dyi_dthetai = layers[ 0 ].dy_dtheta
        val dthetai_dwi = layers[ 0 ].dtheta_dW

        val dJ_dwi = MatrixOps.dot( dthetai_dwi!!.transpose() , ( dJ_dyN * dyi_dthetai ) )

        layers[ 0 ].W = optimize( dJ_dwi , layers[ 0 ].W!! )
        layers[ 0 ].B = optimize( dJ_dyN * dyi_dthetai , layers[ 0 ].B )

        dJ_dtheta = dJ_dyN * dyi_dthetai
        dtheta_dX = layers[ 0 ].dtheta_dX
        for ( i in 1 until layers.size ) {
            val dJ_dyi = MatrixOps.dot( dJ_dtheta!! , dtheta_dX!!.transpose() )
            val dJ_dthetai = dJ_dyi * layers[ i ].dy_dtheta
            val dJ_dwi = MatrixOps.dot( layers[ i ].dtheta_dW!!.transpose() , dJ_dthetai )
            layers[ i ].W = optimize( dJ_dwi , layers[ i ].W!! )
            layers[ i ].B = optimize( dJ_dthetai , layers[ i ].B )
            dJ_dtheta = dJ_dthetai
            dtheta_dX = layers[ i ].dtheta_dX
        }

        layers.reverse()
    }

    private fun optimize(grad : Matrix, param : Matrix ) : Matrix {
        return param - ( grad * alpha )
    }

    private fun lossGradient( y : Matrix , y_hat : Matrix ) : Matrix {
        return ( y - y_hat )
    }

}