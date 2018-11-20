const tf = require("@tensorflow/tfjs");
// require('@tensorflow/tfjs-node');

const a = tf.variable(tf.scalar(Math.random()));
const b = tf.variable(tf.scalar(Math.random()));
const c = tf.variable(tf.scalar(Math.random()));
const d = tf.variable(tf.scalar(Math.random()));

const learningRate = 0.0001;
const optimizer = tf.train.sgd(learningRate);

function loss(predictions, labels) {
  // Subtract our labels (actual values) from predictions, square the results,
  // and take the mean.
  const meanSquareError = predictions.sub(labels).square().mean();
  return meanSquareError;
}

function predict(x){
	return tf.tidy(()=>{
		return a.mul(x.pow(tf.scalar(3)))
		.add(b.mul(x.square()))
		.add(c.mul(x))
		.add(d);
	})
}

function train(xs, ys, numIterations = 875) {
	for (let iter = 0; iter < numIterations; iter++) {
		// a.print()
		optimizer.minimize(() => {
			const predsYs = predict(xs);
			return loss(predsYs, ys);
		});
	}
}

const x = tf.scalar(3);

const ans_a = tf.scalar(1);
const ans_b = tf.scalar(1);
const ans_c = tf.scalar(1);
const ans_d = tf.scalar(1);

train(x, ans_a.mul(x.pow(tf.scalar(3)))
		.add(ans_b.mul(x.square()))
		.add(ans_c.mul(x))
		.add(ans_d));

const res = predict(x);
// res.print();
a.print()
// console.log(res);


// var a = [[1,2,3],
// 		 [3,4,5]];
// var b = [[2,3,4],
// 		 [3,1,4]];

// var v1 = tf.tensor2d(a);
// var v2 = tf.tensor2d(b);
// v1.print()
// v2.print()
// var res = v2.mul(v1);
// res.print()

// var variable = tf.zeros([2,3]);
// var biases = tf.variable(variable);
// biases.print()

// var t = [[1,2,3],
// 		 [4,5,6]];
// var pic1 = tf.tensor(t, [2,3])
// pic1.print()
// biases.assign(pic1.mul(biases));
// biases.square()
// biases.print()

// const a = tf.scalar(2);
// const b = tf.scalar(4);
// const c = tf.scalar(8);
// const input = 2;

// const res = tf.tidy(()=>{
// 	const x = tf.scalar(input);
// 	const ax2 = a.mul(x.square());
// 	const bx = b.mul(x);
// 	const y = ax2.add(bx).add(c);

// 	return y;
// })
// res.print()