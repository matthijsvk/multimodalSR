var test = require('tape');
var callNextTick = require('../index');

test('Async callin\'', function asyncTest(t) {
  t.plan(1);

	var resultsInOrder = [];

  function getResult(callback) {
  	var error = null;
  	var constantResult = 'This should be the second result.';
		callNextTick(callback, error, constantResult);
	}

	function callBackThatAddsResult(error, result) {
		resultsInOrder.push(result);
	}

	getResult(callBackThatAddsResult);
	resultsInOrder.push('This should be the first result.');

  setTimeout(function () {
  	t.deepEqual(
  		resultsInOrder, 
  		[
  			'This should be the first result.',
  			'This should be the second result.'
  		],
  		'Results were added in the right order.'
  	)
  },
  100);
});

test('More than one value param', function moreThanOneValueParamTest(t) {
	t.plan(1);

	var savedParams;

	function saveParams(error, one, two, three) {
		savedParams = [one, two, three];
	}

	callNextTick(saveParams, null, 'one', 'two', 'three');
  
	process.nextTick(function checkSavedParams() {
		t.deepEqual(
			savedParams, ['one', 'two', 'three'],
			'The callback received all of the parameters.'
		);
	});
});
