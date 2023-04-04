import { Network, Layer } from "./brain.js";

let solutions = [
    {
        input: [0, 0],
        data: true
    },
    {
        input: [1, 1],
        data: true
    },
    {
        input: [1, 0],
        data: false
    },
    {
        input: [0, 1],
        data: false
    }
]


let brain = new Network([2, 2, 1]);
let latestIndex = 0;
let accuracy = 0;
function timestep(){
    latestIndex ++;
    let randomElement = solutions[Math.floor(Math.random() * solutions.length)];
    brain.setActivations(randomElement.input, randomElement.data);
    brain.timestep(true);
    if (brain.lastLayer().getVerdict() == brain.lastLayer().dataType){
        accuracy+=1;
    }
    console.log("accuracy: ", accuracy / latestIndex);
}
for (var i = 0; i < 1000000; i++){
    timestep();
}