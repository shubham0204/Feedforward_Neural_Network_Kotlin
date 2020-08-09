package NeuralNetwork

fun main(args : Array<String>) {

    val model = Model(
            inputDims = 6 ,
            layers = arrayOf(
                    Dense( 12 , ActivationOps.ReLU() ),
                    Dense( 6 , ActivationOps.ReLU() ),
                    Dense( 2 , ActivationOps.Sigmoid() )
            )
    )
    model.compile()
    val inputs = MatrixOps.uniform( 1 , 6 )
    val labels = MatrixOps.zerosLike( 1 , 2 )
    labels.set( 0 , 1 , 1.0 )
    for( i in 0 until 100 ) {
        model.forward( inputs , labels )
        model.backward()
    }


}
