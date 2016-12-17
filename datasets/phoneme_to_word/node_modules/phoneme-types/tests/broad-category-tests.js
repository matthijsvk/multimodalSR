var test = require('tape');
var phonemeTypes = require('../phoneme-types');

test('Basic tests', function basics(t) {
  t.plan(4);

  t.equal(phonemeTypes.isConsonantish('AA'), false);
  t.equal(phonemeTypes.isConsonantish('K'), true);

  t.equal(phonemeTypes.isVowelish('UH'), true);
  t.equal(phonemeTypes.isVowelish('Y'), false);
});

test('Syllable enders', function syllableEnders(t) {  
  var expectedResults = {
    AA: false,
    AE: false,
    AH: false,
    AO: false,
    AW: false,
    AY: false,
    EH: false,
    ER: false,
    EY: false,
    IH: false,
    IY: false,
    OW: false,
    OY: false,
    UH: false,
    UW: false,
    CH: true,
    JH: true,
    HH: false,
    DH: true,
    F: false,
    S: false,
    SH: true,
    TH: true,
    V: true,
    Z: true,
    ZH: true,
    L: false,
    R: false,
    M: true,
    N: false,
    NG: true,
    W: false,
    Y: false,
    B: true,
    D: true,
    G: true,
    K: true,
    P: false,
    T: true
  };

  t.plan(Object.keys(expectedResults).length);

  for (var phoneme in expectedResults) {
    t.equal(
      phonemeTypes.isSyllableEnder(phoneme), 
      expectedResults[phoneme],
      phoneme + (expectedResults[phoneme] ? ' is' : ' is not') + 
        ' classified as a syllable ender.'
    );
  }
});
