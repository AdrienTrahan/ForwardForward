// import { Layer, Network } from "./brain.js";
import { Neuron, Layer, Cluster } from "./solo_brain.js";


let neuron1 = new Neuron();
let neuron2 = new Neuron();
let neuron3 = new Neuron();
let neuron4 = new Neuron();
let neuron5 = new Neuron();
neuron3.connectNeurons([neuron1, neuron2]);
neuron4.connectNeurons([neuron1, neuron2]);
neuron5.connectNeurons([neuron3, neuron4]);

let layer1 = new Layer(2);
let layer21 = new Layer(1)
let layer22 = new Layer(1);
let layer3 = new Layer(1);
layer21.connectLayer(layer1);
layer22.connectLayer(layer1);
layer3.connectLayer(layer21);
layer3.connectLayer(layer22);
layer1.connectLayer(layer3);
let cluster1 = new Cluster([layer1, layer21, layer22, layer3]);


let training = [
    {
        data:[0,0],
        type: false
    },
    {
        data:[1,0],
        type: true
    },
    {
        data:[0,1],
        type: true
    },
    {
        data:[0,0],
        type: false
    }
]
let accuracy = 0;
let accuracyCount = 0;
for (var i = 0; i < 1000000; i++){
    let randomData = training[Math.floor(Math.random() * training.length)];
    cluster1.layers[0].setActivations(randomData.data, randomData.type ? 1 : 0);
    cluster1.timestep();

    if (layer3.binaryPositivity()[0] == layer3.verdict()[0]){
        accuracy++;
    }
    accuracyCount++;
    console.log("acc", accuracy/accuracyCount);
}
