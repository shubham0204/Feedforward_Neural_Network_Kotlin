package neural_network

import neural_network.activations.ActivationOps
import neural_network.loss.MeanSquaredError
import neural_network.optimizer.RMSProp
import neural_network.optimizer.SGD

fun main(args : Array<String>) {

    val model = Model(
            inputDims = 6 ,
            layers = arrayOf(
                    Dense( 12 , ActivationOps.ReLU() ),
                    Dense( 6 , ActivationOps.ReLU() ),
                    Dense( 2 , ActivationOps.Softmax() )
            )
    )
    val loss = MeanSquaredError()
    val optimizer = RMSProp().apply { learningRate = 0.01 }
    model.compile( loss , optimizer )

    val x = MatrixOps.uniform( 1 , 6 )
    val y = MatrixOps.uniform( 1 , 2 )
    model.forward( x , y )
    model.backward()


}
