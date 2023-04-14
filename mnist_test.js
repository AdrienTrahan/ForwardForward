import { Network, Layer } from "./brain.js";
import mnist from "mnist";
let {training: trainingData, test: testData} = mnist.set(8000, 10);
function createGoodData(source){
    return source.map(x => [...x.input].concat(x.output));
}
function createBadDataFromGoodData(source){
    return source.map(x => [...x.input].concat(randomShift(x.output)));
}
function randomShift(array){
    let shiftAmount = Math.floor((array.length - 1) * Math.random())+1;
    while (shiftAmount > 0){
        let element = array.shift();
        array.push(element);
        shiftAmount--;
    }
    return array;
}
let goodData = []
let badData = []
goodData = createGoodData(trainingData);
badData = createBadDataFromGoodData(trainingData);

let test_goodData = []
let test_badData = []
test_goodData = createGoodData(testData);
test_badData = createBadDataFromGoodData(testData);
let brain = new Network([goodData[0].length, 2000, 1000, 100, 10]);
function trainGood(){
    let randomElement = goodData[Math.floor(Math.random() * goodData.length)];
    brain.setActivations(randomElement, true);
    return brain.followstep(true);
}
function trainBad(){
    let randomElement = badData[Math.floor(Math.random() * badData.length)];
    brain.setActivations(randomElement, false);
    return brain.followstep(true);
}
let testPerPhase = 50;
for (var i = 0; i < 100; i++){
    for (var j = 0; j < testPerPhase; j++){
        trainGood()
        trainBad();
    }
    brain.updateWeightsBias();
    console.log(`accuracy: ${test() * 100}%`);
}


function test(){
    let correct = 0;
    for (var i = 0; i < test_goodData.length; i++){
        brain.setActivations(test_goodData[i]);
        brain.timestep(false);
        brain.timestep(false);
        brain.timestep(false);
        brain.timestep(false);
        if (brain.lastLayer().getVerdict()){
            correct++;
        }
    }
    let goodDataCorrect = correct;
    for (var i = 0; i < test_badData.length; i++){
        brain.setActivations(test_badData[i]);
        brain.timestep(false);
        brain.timestep(false);
        brain.timestep(false);
        brain.timestep(false);
        if (!brain.lastLayer().getVerdict()){
            correct++;
        }
    }
    console.log("ratio", goodDataCorrect/correct);
    return (correct/(test_badData.length + test_goodData.length));
}