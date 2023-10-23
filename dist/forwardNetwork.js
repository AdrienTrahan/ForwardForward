import Matrix from "matrix-js";
export var MaskingType;
(function (MaskingType) {
    MaskingType[MaskingType["NoMasking"] = 0] = "NoMasking";
    MaskingType[MaskingType["StraightMasking"] = 1] = "StraightMasking";
})(MaskingType || (MaskingType = {}));
class LossFunction {
    useNormalization = true;
    threshold;
    activationFunction;
    derivativeFunction;
}
export class StaticFunction extends LossFunction {
    useNormalization = true;
    threshold = 0.5;
    activationFunction = (x) => x;
    derivativeFunction = (activation, weight, overallActivation, completedActivation, sleeping) => 0;
}
export class JeffFunction extends LossFunction {
    useNormalization = true;
    threshold = 0.5;
    activationFunction = (x) => x > 0 ? x : 0;
    derivativeFunction = (activation, weight, overallActivation, completedActivation, sleeping, goodness) => {
        return ((1 / (1 + Math.exp(-(goodness - this.threshold)))) - (sleeping ? 0 : 1)) * 2 * (activation * weight) * (activation * weight >= 0 ? 1 : 0) * activation;
    };
}
export class ForwardNetwork {
    static Matrix = Matrix;
    layers = [];
    connections = {};
    masks = {};
    lossFunction = {};
    sleeping = false;
    assignId(layer) {
        layer.id = this.layers.length;
        this.layers.push(layer);
    }
    createConnection(sendingLayer, receivingLayer, lossFunction, connectionInitializer = (x, y) => Math.random(), masking = MaskingType.NoMasking) {
        let connectionId = `${sendingLayer.id}-${receivingLayer.id}`;
        this.connections[connectionId] = Matrix(Array(receivingLayer.size).fill(0).map((_, x) => Array(sendingLayer.size).fill(0).map((_, y) => connectionInitializer(x, y))));
        this.masks[connectionId] = Matrix(Array(receivingLayer.size).fill(0).map(() => Array(sendingLayer.size).fill(0).map(() => 1)));
        this.lossFunction[connectionId] = lossFunction;
        switch (masking) {
            case MaskingType.StraightMasking: {
                if (sendingLayer.size != receivingLayer.size) {
                    console.error(`layers must have the same size to use neuron to neuron masking. got sizes: ${sendingLayer.size} -> ${receivingLayer.size}`);
                }
                else {
                    this.masks[connectionId] = Matrix(Matrix(Array(sendingLayer.size).fill(Array(sendingLayer.size).fill(0))).map((_, [y, x]) => x == y ? 1 : 0));
                }
                break;
            }
            default: {
                break;
            }
        }
        receivingLayer.incommingConnections.push(sendingLayer.id);
        sendingLayer.outcommingConnections.push(receivingLayer.id);
    }
    forward(sleeping = false) {
        this.sleeping = sleeping;
        for (var layer of this.layers) {
            layer.forwardPass();
        }
        for (var layer of this.layers) {
            layer.adjusteWeights();
        }
        for (var layer of this.layers) {
            layer.sync();
        }
    }
    clear() {
        for (var layer of this.layers) {
            layer.setActivations(Array(layer.size).fill(0));
        }
    }
    compactPurkinjeWeights(connections, decodeWeights) {
        return Matrix(Matrix(this.connections[decodeWeights].mul(this.masks[decodeWeights])).prod(Matrix(this.connections[connections].mul(this.masks[connections]))));
    }
}
export class ForwardLayer {
    activations;
    tempActivations;
    size;
    network;
    id;
    learningRate = 0.001;
    shape = [];
    activationThreshold = 2;
    incommingConnections = [];
    outcommingConnections = [];
    lossFunction;
    painLayer = false;
    constructor(network, size, lossFunction, shape) {
        this.network = network;
        this.network.assignId(this);
        this.size = size;
        this.lossFunction = lossFunction;
        this.activations = Matrix(Array(size).fill(0).map(() => [0]));
        this.tempActivations = Matrix(Array(size).fill(0).map(() => [0]));
        this.shape = shape ?? [size];
    }
    connectToNextLayer(nextLayer, lossFunction, connectionInitializer = (x, y) => Math.random(), masking = MaskingType.NoMasking) {
        this.network.createConnection(this, nextLayer, lossFunction, connectionInitializer, masking);
    }
    forwardPass() {
        this.tempActivations = Matrix(Array(this.size).fill(0).map(() => [0]));
        for (var previousLayer of this.incommingConnections.map(id => this.network.layers[id])) {
            if (!previousLayer.painLayer) {
                let normalizedPreActivations = previousLayer.activations;
                if (this.lossFunction.useNormalization) {
                    let sum = normalizedPreActivations().reduce((a, b) => a + Math.pow(b[0], 2), 0);
                    sum = (sum == 0) ? 1 : sum;
                    normalizedPreActivations = Matrix(normalizedPreActivations.map((x) => Math.pow(x, 2) / sum));
                }
                this.tempActivations = Matrix(this.tempActivations.add(Matrix(Matrix(this.network.connections[`${previousLayer.id}-${this.id}`].mul(this.network.masks[`${previousLayer.id}-${this.id}`])).prod(normalizedPreActivations))));
            }
        }
    }
    adjusteWeights() {
        let error = 0;
        for (var i = 0; i < this.incommingConnections.length; i++) {
            let previousLayer = this.network.layers[this.incommingConnections[i]];
            if (!previousLayer.painLayer) {
                let lossFunction = this.network.lossFunction[`${previousLayer.id}-${this.id}`] ?? previousLayer.lossFunction;
                let weightsActivity = Matrix(Matrix(Array(this.size).fill((lossFunction.useNormalization ? previousLayer.getNormalizedActivations() : previousLayer.activations)().map(x => x[0]))).map((element, [x, y]) => { return [element, this.network.connections[`${previousLayer.id}-${this.id}`](x, y)]; }));
                let gradient = Matrix(weightsActivity.map(([activation, weight], index) => this.learningRate * lossFunction.derivativeFunction(activation, weight, this.tempActivations()[index[0]], lossFunction.activationFunction(this.tempActivations()[index[0]]), this.network.sleeping, this.goodness(), error)));
                this.network.connections[`${previousLayer.id}-${this.id}`] = Matrix(this.network.connections[`${previousLayer.id}-${this.id}`].sub(gradient));
            }
            else {
                error = previousLayer.activations()[0][0];
            }
        }
    }
    getNormalizedActivations() {
        let normalizedPreActivations = this.activations;
        let sum = normalizedPreActivations().reduce((a, b) => a + Math.pow(b[0], 2), 0);
        sum = (sum == 0) ? 1 : sum;
        return (Matrix(normalizedPreActivations.map((x) => Math.pow(x, 2) / sum)));
    }
    sync() {
        this.tempActivations = Matrix(this.tempActivations.map(this.lossFunction.activationFunction));
        this.activations = Matrix([...this.tempActivations()].map(x => [...x]));
    }
    setActivations(activations) {
        this.activations = Matrix(activations.map(x => [x]));
    }
    goodness() {
        return this.activations().reduce((a, b) => a + Math.pow(b[0], 2), 0);
    }
}
//# sourceMappingURL=forwardNetwork.js.map