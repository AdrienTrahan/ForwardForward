import { ForwardNetwork, ForwardLayer, JeffFunction } from "./forwardNetwork.js";

let network = new ForwardNetwork();
let layer1 = new ForwardLayer(network, 2, new JeffFunction());
let layer2 = new ForwardLayer(network, 4, new JeffFunction());
let layer3 = new ForwardLayer(network, 1, new JeffFunction());

layer1.connectToNextLayer(layer2);
layer2.connectToNextLayer(layer3);

