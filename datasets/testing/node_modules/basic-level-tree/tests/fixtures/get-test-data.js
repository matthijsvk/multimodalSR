var Immutable = require('immutable');

var testData = {
  'Wart': {
    value: {
      name: 'Wart',
      weakness: 'vegetables'
    },
    sessionKeysOfExpectedChildren: [
      'Tryclyde',
      'Fryguy'
    ]
  },
  'Tryclyde': {
    value: {
      name: 'Tryclyde',
      weakness: 'mushroom blocks'
    },
    sessionKeysOfExpectedChildren: [
      'Cobrat',
      'Pokey',
      'Panser'
    ]
  },
  'Fryguy': {
    value: {
      name: 'Fryguy',
      weakness: 'mushroom blocks'
    },
    sessionKeysOfExpectedChildren: [
      'Flurry',
      'Autobomb'
    ]
  },
  'Cobrat': {
    value: {
      name: 'Cobrat',
      weakness: 'turnips'
    },
    sessionKeysOfExpectedChildren: []
  },
  'Pokey': {
    value: {
      name: 'Pokey',
      weakness: 'Pokey heads'
    },
    sessionKeysOfExpectedChildren: []
  },
  'Panser': {
    value: {
      name: 'Panser',
      weakness: 'turtle shells'
    },
    sessionKeysOfExpectedChildren: []
  },
  'Flurry': {
    value: {
      name: 'Flurry',
      weakness: 'carrots'
    },
    sessionKeysOfExpectedChildren: []
  },
  'Autobomb': {
    value: {
      name: 'Autobomb',
      weakness: 'Flurry'
    },
    sessionKeysOfExpectedChildren: []
  }
};

module.exports = Immutable.Map(testData);
