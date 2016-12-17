var Immutable = require('immutable');

var testCases = [
  {
    parentKey: 'root',
    value: {
      name: 'Tryclyde',
      weakness: 'mushroom blocks'
    },
  },
  {
    parentKey: 'root',
    value: {
      name: 'Fryguy',
      weakness: 'mushroom blocks'
    }
  },

  {
    parentKey: 'Tryclyde',
    value: {
      name: 'Cobrat',
      weakness: 'turnips'
    }
  },
  {
    parentKey: 'Tryclyde',
    value: {
      name: 'Pokey',
      weakness: 'Pokey heads'
    }
  },
  {
    parentKey: 'Tryclyde',
    value: {
      name: 'Panser',
      weakness: 'turtle shells'
    }
  },

  {
    parentKey: 'Fryguy',
    value: {
      name: 'Flurry',
      weakness: 'carrots'
    }
  },
  {
    parentKey: 'Fryguy',
    value: {
      name: 'Autobomb',
      weakness: 'Flurry'
    }
  }

];

module.exports = Immutable.List(testCases);
