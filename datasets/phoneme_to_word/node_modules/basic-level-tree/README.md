basic-level-tree
================

A basic LevelDB-backed key-value tree. You can add nodes to your tree, and they will persist to a database.

Installation
------------

    npm install basic-level-tree

Usage
-----

Assuming a tree that looked like this:

    {
      "value": {
        "name": "Wart",
        "weakness": "vegetables"
      },
      "children": [
        {
          "value": {
            "name": "Tryclyde",
            "weakness": "mushroom blocks"
          },
          "children": [
            {
              "value": {
                "name": "Cobrat",
                "weakness": "turnips"
              },
              "children": []
            },
            {
              "value": {
                "name": "Pokey",
                "weakness": "Pokey heads"
              },
              "children": []
            },
            {
              "value": {
                "name": "Panser",
                "weakness": "turtle shells"
              },
              "children": []
            }
          ]
        },
        {
          "value": {
            "name": "Fryguy",
            "weakness": "mushroom blocks"
          },
          "children": [
            {
              "value": {
                "name": "Flurry",
                "weakness": "carrots"
              },
              "children": []
            },
            {
              "value": {
                "name": "Autobomb",
                "weakness": "Flurry"
              },
              "children": []
            }
          ]
        }
      ]
    }

You can retrieve subtrees by specifying the path you want to take through the tree in an array:

    var level = require('level');
    var createLevelTree = require('basic-level-tree');

    var db = level(
      './a-tree.db',
      {
        valueEncoding: 'json'
      }
    );
    // Note: basic-level-tree will not work without valueEncoding set to 'json'.

    createLevelTree(
      {
        db: db,
        treeName: 'subcon'
      },
      readFromTree
    );

    function readFromTree(tree, done) {
      tree.getSubtreeAtPath(['Fryguy'], logSubtree);
    }

    function logSubtree(error, subtree) {
      if (error) {
        console.log(error);
      }
      else {
        console.log(JSON.stringify(subtree, null, '  ');
      }
    }

Output:

    {
      "value": {
        "name": "Fryguy",
        "weakness": "mushroom blocks"
      },
      "children": [
        {
          "value": {
            "name": "Flurry",
            "weakness": "carrots"
          },
          "children": []
        },
        {
          "value": {
            "name": "Autobomb",
            "weakness": "Flurry"
          },
          "children": []
        }
      ]
    }

Check out [add-tests.js](https://github.com/jimkang/basic-level-tree/blob/master/tests/add-tests.js#L111) for an example of how to populate the tree.

API
---

**createLevelTree(opts, done)** - Creates a level-tree and passes back the root node to you. opts object takes:

- `db`: An open [levelup](https://github.com/Level/levelup) database instance, commonly created by calling the [ctor](https://github.com/Level/levelup#ctor).
- `treeName`: The name of your tree. Internally, it uses this to prevent name clashes with other trees you may have stored in your database.
- `root`: The value of the root. If you've already created your tree, this is ignored. If you haven't, it uses this to create the root of your tree.

Every node in the tree has these methods:

**addChild(childValue, done)** - Adds a child to the node and passes it back to you via the `done` callback. The `childValue` can be an object, a string, or anything that can be serialized to JSON.

Right now, `basic-level-tree` does not have a convenient way to add a lot of nodes at once. I'm using it to build a tree as I parse data, one node at a time. However, if you need to add a lot of data at once, please create an issue or submit a pull request!

Every node in the tree has a `value` property. It is the `child` you passed to `addChild`.

**addChildIfNotThere(opts, done)** - opts include `value`, the value of the child you are potentially adding and `equalityFn`, a function that takes two values and tells `addChildIfNotThere` if they are equal. If `equalityFn` is not specified, it will default to `_.isEqual`, a deep equality predicate.

If it finds that child already exists as defined by `equalityFn`, then it will pass back that existing child and instead of adding anything. If it does not exist, then it calls `addChild`.

**getChildren(done)** - Passes back to you the child nodes of the current node.

**getChildAtPath(path, done)** - Traverses the tree starting at the current node along the specified path and returns the node at the end. In the example tree above, if you had the root node, which has a value with the name "Wart" and wanted to get its grandchild node with the value with the name "Pokey", you could call `getChildPath` using the path `['Tryclyde', 'Pokey']. The method will visit Wart's child Tryclyde, then visit Tryclyde's child Pokey.

**getSubtree(done)** - Returns a representation of the subtree that has the current node at its root. The representation is a plain JavaScript object that has nodes each containing a `value` and `children`. There are no methods in this representation.

**getSubtreeAtPath(done)** - On your behalf, this method calls `getChildAtPath`, then calls `getSubtree` on that child. It's used in the [usage example](https://github.com/jimkang/basic-level-tree#usage) at the top.

**save(done)** - This will persist any changes you make to the node. I recommend you only mess with the `value`, but hey, I'm not your dad. (Unless you are Anderson. Ohay guy!)

Right now, there's no way to delete nodes. If you want one, please create an issue or a pull request!

Tests
-----

Run tests with `make test`.

Contributing
------------

- Please add tests for anything you change and add commands to run them to the `test` target in the Makefile.
- Follow the existing style.
- Avoid prototypal inheritance so that methods can be passed around without needing `bind`, unless you have a very strong reason to use it.

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
