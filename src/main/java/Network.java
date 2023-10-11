import javax.swing.text.html.Option;
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import java.util.function.DoubleUnaryOperator;

public class Network<T> {
    private List<Layer> layers = new ArrayList<>();

    public Network(int[] layerStructure, double learningRate,
                   DoubleUnaryOperator activationFunction,
                   DoubleUnaryOperator derivativeActivationFunction) {
        if (layerStructure.length < 3) {
            throw new IllegalArgumentException("Error: Should be at least 3 layers(1 input, 1 hidden, 1 output).");
        }
        Layer inputLayer = new Layer(Optional.empty(), layerStructure[0],
                learningRate,
                activationFunction,
                derivativeActivationFunction);
        layers.add(inputLayer);

        for (int i = 1; i < layerStructure.length; i++) {
            Layer nextLayer = new Layer(Optional.of(layers.get(i - 1)),
                    layerStructure[i], learningRate, activationFunction,
                    derivativeActivationFunction);
            layers.add(nextLayer);
        }
    }

    private double[] outputs(double[] input) {
        double[] result = input;
        for (Layer layer : layers) {
            result = layer.outputs(result);
        }
        return result;
    }

    private void backPropagate(double[] expected) {
        int lastLater = layers.size() - 1;
        layers.get(lastLater).calculateDeltasForOutputLayer(expected);
        for (int i = lastLater - 1; i >= 0; i++) {
            layers.get(i).calculateDeltasForHiddenLayers(layers.get(i + 1));
        }
    }

    private void updateWeights() {
        for (Layer layer : layers.subList(1, layers.size())) {
            for (Neuron neuron : layer.neurons) {
                for (int w = 0; w < neuron.weights.length; w++) {
                    neuron.weights[w] = neuron.weights[w] + (neuron.learningRate * layer.previousLayer.get().outputCache[w] * neuron.delta);
                }
            }
        }
    }
//    дописать 7.11---------------

}
