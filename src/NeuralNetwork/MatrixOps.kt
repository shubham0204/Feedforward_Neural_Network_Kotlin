package NeuralNetwork

import java.lang.RuntimeException
import java.util.*
import kotlin.math.exp

class MatrixOps {

    companion object {

        private fun dot( x1 : DoubleArray , x2 : DoubleArray ) : Double  {
            assert( x1.size == x2.size )
            return DoubleArray( x1.size ).mapIndexed{ index, d -> x1[ index ] * x2[ index ] }.sum()
        }

        fun zerosLike(m : Int, n : Int ) : Matrix {
            return Matrix( m , n )
        }

        fun onesLike(m : Int, n : Int ) : Matrix {
            val out = Matrix( m , n )
            for ( i in 0 until m ) {
                for ( j in 0 until n ) {
                    out.set( i , j , 1.0 )
                }
            }
            return out
        }

        fun sigmoid( mat : Matrix ) : Matrix {
            for ( i in 0 until mat.m ) {
                for ( j in 0 until mat.n ) {
                    mat.set( i , j , sigmoid_( mat.get( i , j ) ) )
                }
            }
            return mat
        }

        fun uniform( m : Int , n : Int ) : Matrix {
            val out = Matrix( m , n )
            for ( i in 0 until m ) {
                for ( j in 0 until n ) {
                    out.set( i , j , Random().nextDouble() )
                }
            }
            return out
        }

        fun dot(mat1 : Matrix, mat2 : Matrix ) : Matrix {
            if ( mat1.n != mat2.m ) {
                throw RuntimeException( "Shapes incompatible $mat1 and $mat2")
            }
            val product = Matrix( mat1.m , mat2.n )
            for ( i in 0 until mat1.m ){
                for ( j in 0 until mat2.n ) {
                    product.set( i , j , dot( mat1.getRow( i ) , mat2.getColumn( j ) ))
                }
            }
            return product
        }

        private fun sigmoid_( x : Double ) : Double {
            return 1.0 / ( 1.0 + exp( -x ) )
        }

    }



}