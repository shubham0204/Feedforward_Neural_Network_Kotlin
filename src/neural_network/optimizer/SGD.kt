package neural_network.optimizer

import neural_network.Dense
import neural_network.Matrix
import neural_network.MatrixOps
import neural_network.loss.Loss

class SGD( private var withMomentum : Boolean = false ) : Optimizer() {

    var learningRate = 0.001
    var beta = 0.99
    var V_w : Matrix? = null
    var V_b : Matrix? = null

    override fun optimize(layers: Array<Dense>, loss: Loss, y_pred: Matrix, y_true: Matrix): Array<Dense> {
        layers.reverse()
        val dJ_dyN = loss.computeLossGradient( y_true , y_pred )

        layers.forEach{ layer -> layer.computeGradients() }

        var dJ_dtheta : Matrix?
        var dtheta_dX : Matrix?

        val dyN_dthetaN = layers[ 0 ].dy_dtheta
        val dthetai_dwi = layers[ 0 ].dtheta_dW

        val dJ_dwN = MatrixOps.dot(dthetai_dwi!!.transpose(), (dJ_dyN * dyN_dthetaN))

        layers[ 0 ].W = optimize( dJ_dwN , layers[ 0 ].W!! )
        layers[ 0 ].B = optimize( dJ_dyN * dyN_dthetaN , layers[ 0 ].B )

        dJ_dtheta = dJ_dyN * dyN_dthetaN
        dtheta_dX = layers[ 0 ].dtheta_dX
        for ( i in 1 until layers.size ) {
            val dJ_dyi = MatrixOps.dot(dJ_dtheta!!, dtheta_dX!!.transpose())
            val dJ_dthetai = dJ_dyi * layers[ i ].dy_dtheta
            val dJ_dwi = MatrixOps.dot(layers[i].dtheta_dW!!.transpose(), dJ_dthetai)
            if ( withMomentum ) {
                if ( V_b == null && V_w == null ) {
                    V_w = MatrixOps.zerosLike( dJ_dwi )
                    V_b = MatrixOps.zerosLike( dJ_dthetai )
                }
                V_w = ( V_w!! * beta ) + ( dJ_dwi * ( 1.0 - beta ) )
                V_b = ( V_b!! * beta ) + ( dJ_dthetai * ( 1.0 - beta ) )
                layers[ i ].W = optimize( V_w!! , layers[ i ].W!! )
                layers[ i ].B = optimize( V_b!! , layers[ i ].B )
            }
            else {
                layers[ i ].W = optimize( dJ_dwi , layers[ i ].W!! )
                layers[ i ].B = optimize( dJ_dthetai , layers[ i ].B )
            }
            dJ_dtheta = dJ_dthetai
            dtheta_dX = layers[ i ].dtheta_dX
        }
        layers.reverse()
        return layers
    }

    private fun optimize(grad : Matrix, param : Matrix) : Matrix {
        return param - ( grad * learningRate )
    }

    override fun resetState() {
        // TODO : NOT IMPLEMENTED
    }


}