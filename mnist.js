const util = require("util");
const fs = require("fs");
const tf = require("@tensorflow/tfjs");
const model = require("./models");
const data_lib = require("./data/read_file");
const argvs = {
	epochs : 1,
	batch_size : 128,
	model_save_path : "./model"
}

async function main(argvs){
	var swc = {
		files : await data_lib.main("./data"),
		argvs : argvs
	}

	const train_datas = await data_lib.get_data(swc, true);
	// console.log(train_datas)
	model.summary();

	const validationSplit = 0.15;
	const numTrainExamplesPerEpoch =
		train_datas.images.shape[0] * (1 - validationSplit);
	const numTrainBatchesPerEpoch =
		Math.ceil(numTrainExamplesPerEpoch / swc.argvs.batchSize);
	console.log("training")

	console.log(train_datas.labels.get(1))

	// await model.fit(train_datas.images, train_datas.labels, {
	// 	epochs : swc.argvs.epochs,
	// 	batchSize : swc.argvs.batchSize,
	// 	validationSplit
	// });

	// console.log("evaluate");

	// const test_data = await data.get_data(swc, false);
	// const evalOutput = model.evaluate(test_data.images, test_data.labels);
	// console.log(
 //      `\nEvaluation result:\n` +
 //      `  Loss = ${evalOutput[0].dataSync()[0].toFixed(3)}; `+
 //      `Accuracy = ${evalOutput[1].dataSync()[0].toFixed(3)}`);

	// if (swc.argvs.model_save_path != null) {
	// 	await model.save(`${swc.argvs.model_save_path}`);
	// 	console.log(`Saved model to path: ${swc.argvs.model_save_path}`);
	// }
}

main(argvs);