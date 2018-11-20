const util = require("util");
const fs = require("fs");
const tf = require("@tensorflow/tfjs");
const assert = require("assert");
const TRAIN_IMAGES_FILE = '/train-images-idx3-ubyte';
const TRAIN_LABELS_FILE = '/train-labels-idx1-ubyte';
const TEST_IMAGES_FILE = '/t10k-images-idx3-ubyte';
const TEST_LABELS_FILE = '/t10k-labels-idx1-ubyte';
const IMAGE_HEADER_MAGIC_NUM = 2051;
const IMAGE_HEADER_BYTES = 16;
const IMAGE_HEIGHT = 28;
const IMAGE_WIDTH = 28;
const IMAGE_FLAT_SIZE = IMAGE_HEIGHT * IMAGE_WIDTH;
const LABEL_HEADER_MAGIC_NUM = 2049;
const LABEL_HEADER_BYTES = 8;
const LABEL_RECORD_BYTE = 1;
const LABEL_FLAT_SIZE = 10;

const BASE_URL = "./data";
function loadHeaderValues(buffer, headerLength) {
	const headerValues = [];
	for (let i = 0; i < headerLength / 4; i++) {
		headerValues[i] = buffer.readUInt32BE(i * 4);
	}
	return headerValues;
}
async function load_files(){
	var files = {
		train_images : fs.readFileSync(BASE_URL + TRAIN_IMAGES_FILE),
		train_labels : fs.readFileSync(BASE_URL + TRAIN_LABELS_FILE),
		test_images : fs.readFileSync(BASE_URL + TEST_IMAGES_FILE),
		test_labels : fs.readFileSync(BASE_URL + TEST_LABELS_FILE),
	}

	return files;
}

async function load_images(buffer){
	const headerBytes = IMAGE_HEADER_BYTES;
	const recordBytes = IMAGE_HEIGHT * IMAGE_WIDTH;
	const headerValues = loadHeaderValues(buffer, headerBytes);
	assert.equal(headerValues[0], IMAGE_HEADER_MAGIC_NUM);
	assert.equal(headerValues[2], IMAGE_HEIGHT);
	assert.equal(headerValues[3], IMAGE_WIDTH);

	const images = [];
	let index = headerBytes;
	while(index < buffer.byteLength){
		const array = new Float32Array(recordBytes);
		for(let i=0;i<recordBytes;i++){
			array[i] = buffer.readUInt8(index++) / 255;
		}
		images.push(array);
	}
	assert.equal(images.length, headerValues[1]);
	return images;
}

async function load_labels(buffer) {
	const headerBytes = LABEL_HEADER_BYTES;
	const recordBytes = LABEL_RECORD_BYTE;

	const headerValues = loadHeaderValues(buffer, headerBytes);
	assert.equal(headerValues[0], LABEL_HEADER_MAGIC_NUM);

	const labels = [];
	let index = headerBytes;
	while (index < buffer.byteLength) {
		const array = new Int32Array(recordBytes);
		for (let i = 0; i < recordBytes; i++) {
			array[i] = buffer.readUInt8(index++);
		}
		labels.push(array);
	}

	assert.equal(labels.length, headerValues[1]);
	return labels;
}

async function main(){
	var swc = {
		files : await load_files()
	}
	swc.files.train_images = await load_images(swc.files.train_images);
	swc.files.test_images = await load_images(swc.files.test_images);
	swc.files.train_labels = await load_labels(swc.files.train_labels);
	swc.files.test_labels = await load_labels(swc.files.test_labels);

	
}

main();