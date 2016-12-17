var sliceCall = Array.prototype.slice.call;

// Expecting params cb, error, result1, result2, ...
function makeCallbackCaller(cb) {
	var paramsForCallback = Array.prototype.slice.call(arguments, 1);

	return function callbackCall() {
		cb.apply(cb, paramsForCallback);
	};
}

// Expecting params cb, error, result1, result2, ...
function callNextTick() {
	var caller = makeCallbackCaller.apply(
		null, Array.prototype.slice.call(arguments, 0)
	);
	process.nextTick(caller);
}

module.exports = callNextTick;
