const {RandomForestClassifier: RFClassifier} = require("ml-random-forest");
const FeedforwardNeuralNetwork = require("ml-fnn");

/*
const rawTrainingSet = [
    [
        1, 6
    ],
    [
        6, 2
    ],
    [
        2, 4
    ],
    [
        4, 3
    ],
    [
        3, 5
    ],
    [
        5, 1
    ]
];
*/
const generateFn = (count) => Array.apply(null, Array(count)).map(() => [
    Math.random() * 100,
    Math.random() * 100
]).map(([
    a, b
], index) => {
    const isEven = index % 2 === 0;
    const high = Math.max(a, b);
    const low = Math.min(a, b);
    return isEven
        ? [high, low]
        : [low, high];
});

// const rawTrainingSet = Array.apply(null, Array(10)).map(() => [
//     Math.random() * 100,
//     Math.random() * 100
// ]).map(([
//     a, b
// ], index) => {
//     const isEven = index % 2 === 0;
//     const high = Math.max(a, b);
//     const low = Math.min(a, b);
//     return isEven
//         ? [high, low]
//         : [low, high];
// });
const normalizeFn = rawSet => rawSet.map(([a, b]) => {
    const max = Math.max(a, b);
    return [
        a / max,
        b / max
    ];
});

// const rawTrainingSet = generateFn(10);
// const trainingSet = normalizeFn(rawTrainingSet);

// const trainingSet = rawTrainingSet;
// const trainingSet = rawTrainingSet.map(([a, b]) => {
//     const max = Math.max(a, b);
//     return [
//         a / max,
//         b / max
//     ];
// });
const predictFn = set => set.map(([a, b]) => (
    (a === b)
    ? 0
    : ((a < b)
        ? -1
        : 1)));

// const predictions = trainingSet.map(([a, b]) => (
//     a === b
//     ? 0
//     : a < b
//         ? -1
//         : 1));

// const predictions = trainingSet.map(([a, b]) => (
//     a <= b
//     ? 0
//     : 1));

// const classifier = new RFClassifier({
//     // seed: 3,
//     // maxFeatures: 0.8,
//     // replacement: true,
//     // nEstimators: 25
// });

const trainingSet = normalizeFn(generateFn(10));
const predictions = predictFn(trainingSet);

const classifier = new FeedforwardNeuralNetwork({hiddenLayers: [4], iterations: 1000, learningRate: 0.01});

console.log(`Training on ${trainingSet.length} records`);
classifier.train(trainingSet, predictions);

function predict(testSet) {

    const result = classifier.predict(testSet);
    const correctPredictions = predictFn(testSet);

    // console.log(result);
    
    let correct = 0;
    testSet.forEach((ts, index) => {
        const actual = result[index];
        const expected = correctPredictions[index];
        const isCorrect = actual === expected;
        if (isCorrect) {
            correct++;
        }
        // console.log(`${isCorrect}\t set: ${testSet[index]}\t actual: ${actual}\t expected: ${expected}`);
    });
    console.log(`Correct: ${correct} / ${testSet.length} tests (${correct / testSet.length * 100}%)`);
    
}
    
// predict(trainingSet);
predict(normalizeFn(generateFn(10000)));

const generateNums = (count) => Array.apply(null, Array(count)).map(() =>
    Math.random() * 100
);
const compareFn = (a, b) => {
    const [ result ] = classifier.predict([[a,b]]);
    return result;
}

console.log("Real world test: Sorting array of numbers");
const rawNums = generateNums(10);
const sortedNums = [...rawNums].sort(compareFn);
console.log("Unsorted:");
console.log(rawNums);
console.log(`Sorted by Neural Network Sort (only learned from ${trainingSet.length} records):`);
console.log(sortedNums);
