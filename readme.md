# Forward Forward Algorithm
This is an implementation of Geoffrey Hinton's forward forward algorithm.

Go read about it <a href="https://www.cs.toronto.edu/~hinton/FFA13.pdf">here</a>
<br>
<br>

First import necessary classes:
```typescript
import { ForwardNetwork, ForwardLayer, JeffFunction, StaticFunction, MaskingType } from "./forwardNetwork.js";
```

You can initialize a network and create a layer like this:
```typescript
let network = new ForwardNetwork();
let layer1 = new ForwardLayer(network, 2, new JeffFunction());
```

If you don't want the layers to be affected each time ```forward``` is called, use a static function like so:
```typescript
let layer2 = new ForwardLayer(network, 1, new StaticFunction());
```

You can then create connections between layers:
```typescript
layer1.connectToNextLayer(layer2);
```
If you want to make apply a Loss function to a specific connection only, you can pass an instance of the function as a parameter.
You can also set a custom weight initialization function as a third parameter. (All weights are initialized randomly by default)
You can also apply a mask to a connection (The default is ```MaskingType.NoMasking```)
```typescript
layer1.connectToNextLayer(layer2, new CustomLossFunction(), () => 0.5, MaskingType.StraightMasking);
```


To propagate activations through the network synchronously, run the ```forward``` command.
Every forward iteration updates the network as if all data is positive.
```typescript
network.forward() //positive data
```

For negative data, pass ```true``` as an argument.
```typescript
network.forward(true) // negative data
```

To retrieve a layer's activation, call
```typescript
layer1.activations()
```
To retrieve a layer's normalized activations, call
```typescript
layer1.getNormalizedActivations()
```
Note: this returns a Matrix
