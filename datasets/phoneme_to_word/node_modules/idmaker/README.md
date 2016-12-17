idmaker
=======

This module generates random ids that can be used as DOM element ids.

Installation
------------

    npm install idmaker

Usage
-----

In the browser:

  var idmaker = createIdmaker();
  var li = document.createElement('li');
  li.id = idmaker.randomId(10);

In Node:

  var idmaker = require('idmaker');
  var id = idmaker.randomId(10);

[Here's an example.](http://jimkang.com/idmaker/example)

Tests
-----

Run tests with `npm test`. Run tests in the debugger with 'npm run-script dtest'.

License
-------

MIT.
