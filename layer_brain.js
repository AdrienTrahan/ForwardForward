import Matrix from "matrix-js";

export class Layer{
    threshold = 2;
    dataType = true;
    loss = 0
    learningRate = 0.1;
    // adamLearningRate = 0.001;
    // positiveOptimizer = new AdamOptimizer(this.adamLearningRate, 0.9, 0.999, false);
    // negativeOptimizer = new AdamOptimizer(this.adamLearningRate, 0.9, 0.999, false);
    // optimizer doesn't work yet
    useOptimizer = false;
    activationFunction = (x) => Math.max(0, x);
    activationFunctionDerivative = (x) => x > 0 ? 1 : 0;
    activationFunctionDerivativeFromResult = (x) => x == 0 ? 0 : 1;
    logisticFunction = (x) => Math.max(Math.min(1/(1+Math.exp(-x)), 0.9999), 1e-4);
    logisticFunctionDerivative = (x) => this.logisticFunction(x) * (1 - this.logisticFunction(x));
    costFunction = (x) => -Math.log(x);
    costFunctionDerivative = (x) => -1/x;
    constructor(size, previous, useOptimizer = false){
        this.size = size;
        this.activations = Matrix(Array.from({length: this.size}, () => [0]));
        this.bias = Matrix(Array.from({length: this.size}, () => [Math.random()]));
        this.isFirst = previous == undefined;
        this.useOptimizer = useOptimizer;
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
        return 0.01;
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
                if (this.useOptimizer){
                    gradient = Matrix(gradient.map(x => this.positiveOptimizer.getOptimizedGradient(x)));
                }else{
                    gradient = Matrix(gradient.map(x => x * this.learningRate));
                }
            }else{
                loss = this.costFunction(1 - probability);
                gradient = this.computeNegativeGradient(probability, goodnessScale, this.activations);
                if (this.useOptimizer){
                    gradient = Matrix(gradient.map(x => this.negativeOptimizer.getOptimizedGradient(x)));
                }else{
                    gradient = Matrix(gradient.map(x => x * this.learningRate));
                }
            }
            this.bias = Matrix(this.bias.sub(gradient));
            if (!this.isFirst){
                let weightsGradient = Matrix(gradient.prod(Matrix(this.normalizeActivations(this.previousLayer.activations).trans()))); 
                this.weights = Matrix(this.weights.sub(weightsGradient));
            }
            this.loss = loss;
    
        }
        return loss;
    }
    updateWeightsBias(){
        this.bias = Matrix(this.bias.sub(this.biasGradient));
        this.weights = Matrix(this.weights.sub(this.weightsGradient));
    }
    computePositiveGradient(probability, goodnessScale, activations){
        let gradient = this.costFunctionDerivative(probability) * 
                       this.logisticFunctionDerivative(goodnessScale);
        gradient = Matrix(activations.map(activation => activation * 2 * gradient));
        gradient = Matrix(gradient.mul(Matrix(activations.map(this.activationFunctionDerivativeFromResult))));
        return gradient;
    }
    computeNegativeGradient(probability, goodnessScale, activations){
        let gradient = - this.costFunctionDerivative(1 - probability) * 
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
    followstep(learn = true){
        let loss = 0;
        for (var i =0; i < this.layers.length; i++){
            this.layers[i].dataType = this.layers[Math.max(i - 1, 0)].dataType;
            loss = this.layers[i].forward(this.layers[i].dataType, learn);
        }
        return loss;
    }
    updateWeightsBias(){
        for (var i =1; i < this.layers.length; i++){
            this.layers[i].updateWeightsBias();
        }
    }
    print(){
        let equations = ["x", "y"];
        for (var i = 1; i < this.layers.length; i++){
            let newEquations = [];
            for (var j = 0; j < this.layers[i].size; j++){
                let currentCellEquation = [];
                let normalizedInput = `if(sqrt(${equations.map(x => x+"^2").join("+")}) == 0,1,sqrt(${equations.map(x => x+"^2").join("+")}))`;
                for (var k = 0; k < this.layers[i - 1].size; k++){
                    currentCellEquation.push(`(${this.layers[i].weights()[j][k]}*${equations[k]}/${normalizedInput})`);
                }
                newEquations.push(`if((${currentCellEquation.join("+")}+${this.layers[i].bias()[j][0]})>0,(${currentCellEquation.join("+")}+${this.layers[i].bias()[j][0]}), 0)`);
            }
            equations = newEquations;
        }
        return `-ln(1/(1+e^(-(${equations}^2 - ${this.lastLayer().threshold}))))`;
    }
}

class AdamOptimizer {
    constructor(learningRate, beta1, beta2) {
      this.learningRate = learningRate;
      this.beta1 = beta1;
      this.beta2 = beta2;
      this.m = 0;
      this.v = 0;
      this.t = 0;
    }
  
    getOptimizedGradient(grad) {
        this.t++;
        this.m = this.beta1 * this.m + (1 - this.beta1) * grad;
        this.v = this.beta2 * this.v + (1 - this.beta2) * (grad ** 2);
        const mHat = this.m / (1 - this.beta1 ** this.t);
        const vHat = this.v / (1 - this.beta2 ** this.t);
        const delta = -this.learningRate * mHat / (Math.sqrt(vHat) + 1e-8);
        return delta;
    }
}