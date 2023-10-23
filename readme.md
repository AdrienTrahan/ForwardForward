#Forward Forward Algorithm
This is an implementation of Geoffrey Hinton's forward forward algorithm.

```typescript
import { ForwardNetwork, ForwardLayer, JeffFunction, StaticFunction } from "./forwardNetwork.js";
```

You can initialize a network and create layers like this:
```typescript
let network = new ForwardNetwork();
let layer1 = new ForwardLayer(network, 2, new JeffFunction());
let layer2 = new ForwardLayer(network, 4, new JeffFunction());
let layer3 = new ForwardLayer(network, 1, new JeffFunction());
```

If you don't want the layers to be affected each time forward is called, us a static function:
```typescript
let layer4 = new ForwardLayer(network, 1, new StaticFunction());
```


```typescript
layer1.connectToNextLayer(layer2);
layer2.connectToNextLayer(layer3);
```

```typescript
network.forward() //positive data
network.forward(true) // negative data
```
