basic-subleveler
================

Handles boilerplate setup for one-level-deep LevelDB [sublevel](https://github.com/dominictarr/level-sublevel) databases:

- Creates levels using the names and namespaces you provide.
- Gives you access to the `close` method on the db, so you can avoid unnecessary locking.
- Provides a "readAllValuesFromSublevel" method.


Installation
------------

    npm install basic-subleveler

Usage
-----

    var basicSubleveler = require('basic-subleveler');
    var level = require('level');
    var queue = require('queue-async');

    var leveldb = level(
      __dirname + '/test.db',
      {
        valueEncoding: 'json'
      }
    );

    var db = basicSubleveler.setUpSubleveledDB({
      db: leveldb,
      sublevels: {
        meats: 'm',
        vegetables: 'v'
      }
    });

    // Will decorate the LevelDB instance with `meats` and `vegetables`
    // sublevels. You can do stuff with them.

    var q = queue();

    q.defer(db.vegetables.put, 'broccoli', 'tasty');
    q.defer(db.vegetables.put, 'tomato', 'juicy');
    q.defer(db.vegetables.put, 'arugula', 'elitist');

    q.awaitAll(readVegetables);

    function readVegetables(error) {
      if (error) {
        console.log(error);
      }
      else {
        db.vegetables.readAllValues(logVegetables);
      }
    }

    function logVegetables(error, vegetables) {      
      if (error) {
        console.log(error);
      }
      else {
        console.log(vegetables);
      }
    }

Output:

  [
    'tasty',
    'juicy',
    'elitist'
  ]

Tests
-----

Run tests with `make test`.

License
-------

The MIT License (MIT)

Copyright (c) 2015 Jim Kang

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
