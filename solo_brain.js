import Matrix from "matrix-js";
export class Neuron{
    threshold = 2;
    positivity = Matrix([[1]]);
    tempPositivity = Matrix([[1]]);
    loss = 0
    learningRate = 0.0001;
    connectedNeurons = [];
    weights = Matrix([[]]);
    bias = Matrix([[Math.random()]]);
    activation = Matrix([[0]]);
    tempActivation = Matrix([[0]]);
    tempPreviousActivations;
    activationFunction = (x) => Math.max(0, x);
    activationFunctionDerivative = (x) => x > 0 ? 1 : 0;
    activationFunctionDerivativeFromResult = (x) => x == 0 ? 0 : 1;
    logisticFunction = (x) => Math.max(Math.min(1/(1+Math.exp(-x)), 0.9999), 1e-4);
    logisticFunctionDerivative = (x) => this.logisticFunction(x) * (1 - this.logisticFunction(x));
    clamp = (x) => Math.min(Math.max(x, 1e-8), 0.99999999);
    costFunction = (positivity, prediction) => -this.clamp(positivity) * Math.log(this.clamp(prediction)) - (1-this.clamp(positivity)) * Math.log(1 - this.clamp(prediction));
    costFunctionDerivative = (positivity, prediction) => - (this.clamp(prediction) - this.clamp(positivity)) / (this.clamp(prediction) - 1) / this.clamp(prediction);
    constructor(){
    }
    connectNeurons(neurons){
        for (var i = 0; i < neurons.length; i++){
            if (this.connectedNeurons.length == 0){
                this.weights = Matrix([[Math.random()]]);
            }else{
                this.weights = Matrix(this.weights.merge.bottom([[Math.random()]]));
            }
            
            this.connectedNeurons.push(neurons[i]);
        }
    }
    previousPositivities(){
        let positivities = undefined;
        for (var i = 0; i < this.connectedNeurons.length; i++){
            if (positivities == undefined){
                positivities = Matrix([...this.connectedNeurons[i].positivity()])
            }else{
                positivities = Matrix(positivities.merge.bottom([...this.connectedNeurons[i].positivity()]));
            }
        }
        return positivities;
    }
    previousActivations(){
        let activations = undefined;
        for (var i = 0; i < this.connectedNeurons.length; i++){
            if (activations == undefined){
                activations = Matrix([...this.connectedNeurons[i].activation()])
            }else{
                activations = Matrix(activations.merge.bottom([...this.connectedNeurons[i].activation()]));
            }
        }
        this.tempPreviousActivations = Matrix([...activations()]);
        return activations;
    }
    weightSum(){
        return this.weights().reduce((a, b) => a+b[0], 0);
    }
    setActivation(activation, positivity){
        this.activation = Matrix([[activation]]);
        this.positivity = Matrix([[positivity]]);
    }
    // computes activation for current neuron
    forward(){
        if (this.connectedNeurons.length == 0){return};
        this.tempActivation = Matrix(Matrix(Matrix(Matrix(this.weights.trans())
            .prod(this.previousActivations()))
            .add(this.bias))
            .map(x => this.activationFunction(x)));
        let weightSum = this.weightSum();
        this.tempPositivity = Matrix(Matrix(Matrix(this.weights.trans())
            .prod(this.previousPositivities()))
            .map(x => x/weightSum));
    }
    sync(){
        this.activation = Matrix([...this.tempActivation()]);
        this.positivity = Matrix([...this.tempPositivity()]);
    }
    goodness(){
        return this.activation.map(x => x * x).reduce((a, b) => a + b[0], 0);
    }
    probability(){
        return this.logisticFunction(this.goodness() - this.threshold)
    }
    cost(){
        let flattenedPositivity = this.positivity().map(x => x[0]);
        return flattenedPositivity.map((positivity) => this.costFunction(positivity, this.probability()))[0];
    }
    gradient(){
        let flattenedPositivity = this.positivity().map(x => x[0]);
        let costFunctionDifferentiated = flattenedPositivity.map((positivity) => this.costFunctionDerivative(positivity, this.probability()))[0];
        let logisticFunctionDifferentiated = this.logisticFunctionDerivative(this.goodness() - this.threshold);
        let goodnessDifferentiated = Matrix(this.activation.map(x => x * 2));
        let activationDifferentiated = Matrix(this.activation.map(this.activationFunctionDerivativeFromResult));
        let gradient = Matrix(Matrix(activationDifferentiated.mul(goodnessDifferentiated)).map(x => x*logisticFunctionDifferentiated*costFunctionDifferentiated))
        return gradient;
    }
    learn(){
        if (this.connectedNeurons.length == 0){return};
        let stepSize = Matrix(this.gradient().map(x => x*this.learningRate));
        this.bias = Matrix(this.bias.sub(stepSize));
        let weightsStep = Matrix(this.tempPreviousActivations.prod(stepSize));
        this.weights = Matrix(this.weights.sub(weightsStep));
    }
    verdict(){
        return this.probability() > 0.5;
    }
    binaryPositivity(){
        return this.positivity()[0][0] > 0.5;
    }
    
}
export class Layer{
    neurons = [];

    constructor(size = 1){
        for (var i = 0; i < size; i++){
            this.neurons.push(new Neuron());
        }
    }
    connectLayer(previousLayer){
        for (var i = 0; i < this.neurons.length; i++){
            for (var j = 0; j < previousLayer.neurons.length; j++){
                this.neurons[i].connectNeurons([previousLayer.neurons[j]]);
            }
        }
    }
    connectNeuron(previousNeuron){
        for (var i = 0; i < this.neurons.length; i++){
            this.neurons[i].connectNeurons(previousNeuron);
        }
    }
    setActivations(activations, positivities){
        for (var i = 0; i < this.neurons.length; i++){
            if (typeof positivities == "number"){
                this.neurons[i].setActivation(activations[i], positivities);
            }else{
                this.neurons[i].setActivation(activations[i], positivities[i]);
            }
        }
    }
    sync(){
        for (var i = 0; i < this.neurons.length; i++){
            this.neurons[i].sync();
        }
    }
    forward(){
        for (var i = 0; i < this.neurons.length; i++){
            this.neurons[i].forward();
        }
    }
    learn(){
        for (var i = 0; i < this.neurons.length; i++){
            this.neurons[i].learn();
        }
    }
    verdict(){
        return this.neurons.map(neuron => neuron.probability() > 0.5);
    }
    binaryPositivity(){
        return this.neurons.map(neuron => neuron.positivity()[0][0] > 0.5);
    }
    timestep(learn = false){
        this.forward();
        this.sync();
        if (learn){
            this.learn();
        }
    }
}
export class Cluster{
    layers = [];
    constructor(layers){
        this.layers = layers;
    }
    forward(){
        for (var i = 0; i < this.layers.length; i++){
            this.layers[i].forward();
        }
    }
    sync(){
        for (var i = 0; i < this.layers.length; i++){
            this.layers[i].sync();
        }
    }
    learn(){
        for (var i = 0; i < this.layers.length; i++){
            this.layers[i].learn();
        }
    }
    timestep(learn = false){
        this.forward();
        this.sync();
        if (learn){
            this.learn();
        }
    }
}