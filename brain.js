import Matrix from "matrix-js";

export class Layer{
    threshold = 2;
    dataType = true;
    activationFunction = (x) => Math.max(0, x);
    activationFunctionDerivative = (x) => x > 0 ? 1 : 0;
    activationFunctionDerivativeFromResult = (x) => x == 0 ? 0 : 1;
    logisticFunction = (x) => 1/(1+Math.exp(-x));
    logisticFunctionDerivative = (x) => this.logisticFunction(x) * (1 - this.logisticFunction(x));
    costFunction = (x) => -Math.log(x);
    costFunctionDerivative = (x) => -1/x;
    constructor(size, previous){
        this.size = size;
        this.activations = Matrix(Array.from({length: this.size}, () => [0]));
        this.bias = Matrix(Array.from({length: this.size}, () => [Math.random()]));
        this.isFirst = previous == undefined;
        if (!this.isFirst){
            this.previousLayer = previous;
            this.weights = Matrix(Array.from({length: this.size}, () => Array.from({length: this.previousLayer.size}, () => Math.random())));
        }
    }
    setActivations(input){
        this.activations = Matrix(input.map(x => [x]));
    }
    fetchActivations(){
        if (this.isFirst){return};
        this.activations = Matrix(Matrix(Matrix(this.weights.prod(this.normalizeActivations(this.previousLayer.activations))).add(this.bias)).map(this.activationFunction));
    }
    normalizeActivations(activations){
        let moduleOfActivations = Math.sqrt(activations.map(x => x*x).reduce((a, b) => a + b[0], 0));
        if (moduleOfActivations == 0){moduleOfActivations = 1;}
        return Matrix(activations.map(x=> x/moduleOfActivations));
    }
    getLearningRate(){
        return 0.1;
    }
    getGoodness(){
        return this.activations.map(x => x*x).reduce((a, b) => a + b[0], 0);
    }
    forward(isDataPositive, learn){
        let gradient, loss;
        this.fetchActivations();
        if (learn){
            let goodnessScale = this.getGoodness() - this.threshold;
            let probability = this.logisticFunction(goodnessScale);
            if (isDataPositive){
                loss = this.costFunction(probability);
                gradient = this.computePositiveGradient(probability, goodnessScale, this.activations);
            }else{
                loss = this.costFunction(1 - probability);
                gradient = this.computeNegativeGradient(probability, goodnessScale, this.activations);
            }
            this.bias = Matrix(this.bias.add(gradient));
            if (!this.isFirst){
                let weightsGradient = Matrix(gradient.prod(Matrix(this.normalizeActivations(this.previousLayer.activations).trans())));
                this.weights = Matrix(this.weights.add(weightsGradient));
            }
        }
        return loss;
    }
    computePositiveGradient(probability, goodnessScale, activations){
        let gradient = - this.getLearningRate() * this.costFunctionDerivative(probability) * 
                       this.logisticFunctionDerivative(goodnessScale);
        gradient = Matrix(activations.map(activation => activation * 2 * gradient));
        gradient = Matrix(gradient.mul(Matrix(activations.map(this.activationFunctionDerivativeFromResult))));
        return gradient;
    }
    computeNegativeGradient(probability, goodnessScale, activations){
        let gradient = this.getLearningRate() * this.costFunctionDerivative(1 - probability) * 
                       this.logisticFunctionDerivative(goodnessScale);
        gradient = Matrix(activations.map(activation => activation * 2 * gradient));
        gradient = Matrix(gradient.mul(Matrix(activations.map(this.activationFunctionDerivativeFromResult))));
        return gradient;
    }
    getVerdict(){
        return (this.getGoodness() - this.threshold) > 0
    }
}

export class Network{
    shape = [];
    layers = [];
    constructor(shape){
        this.shape = shape;
        for (var i = 0; i < shape.length; i++){
            if (i == 0){
                this.layers.push(new Layer(shape[i]))
            }else{
                this.layers.push(new Layer(shape[i], this.layers[i - 1]))
            }
        }
    }
    firstLayer(){
        return this.layers[0];
    }
    lastLayer(){
        return this.layers[this.layers.length - 1];
    }
    setActivations(activations, dataType){
        this.firstLayer().setActivations(activations);
        if (dataType != undefined){
            this.firstLayer().dataType = dataType;
        }
    }
    timestep(learn = true) {
        let loss = 0;
        for (var i = this.layers.length - 1; i >= 0; i--){
            this.layers[i].dataType = this.layers[Math.max(i - 1, 0)].dataType
            loss = this.layers[i].forward(this.layers[i].dataType, learn);
        }
        return loss;
    }
}
