var assert = require('assert');
var idmaker = require('../idmaker');

suite('idmaker', function clashSuite() {
  test('There should be any clashes', function testClashes() {
    var ids = [];
    for (var i = 0; i < 10; ++i) {
      var id = idmaker.randomId(8);
      for (var j = i - 1; j >= 0; --j) {
        assert.notEqual(id, ids[j]);
      }
      ids.push(id);
    }
  });

  test('Lengths of ids should be as specified', function lengthSuite() {
    for (var i = 0; i < 10; ++i) {
      assert.equal(i, idmaker.randomId(i).length);
    }
  });
});
