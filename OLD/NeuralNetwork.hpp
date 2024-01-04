#include <eigen3/Eigen/Eigen>
#include <iostream>
#include <vector>

using std namespace;

typedef float Scalar;
typedef Eigen::MatrixXf matrix;
typedef Eigen::RowVectorXf RowVector;
typedef Eigen::VectorXf ColVector;

class NeuralNetwork {
    public:
        NeuralNetwork(vector<int> topology, Scalar learningRate = Scalar(0.005));
        void propogateForward(RowVector& input);
        void propogateBackward(RowVector& output);
        void calcErrors(RowVector& output);
        void updateWeights();
        void train(vector<RowVector*> data);

        vector<RowVector*> neuronLayers;
        vector<RowVector*> cacheLayers;
        vector<RowVector*> deltas;
        vector<Matrix*> weights;
        Scalar learning rate;
}
