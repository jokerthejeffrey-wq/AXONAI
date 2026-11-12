import test from 'node:test';import assert from 'node:assert/strict';import {simpleArithmetic} from '../server/arithmetic.js';
test('basic arithmetic is exact',()=>{assert.equal(simpleArithmetic('100 - 100'),'0');assert.equal(simpleArithmetic('What is 100 minus 100?'),'0');assert.equal(simpleArithmetic('What is 13 plus 3?'),'16');assert.equal(simpleArithmetic('6 / 0'),'Division by zero is undefined.');});
test('no generic code execution',()=>{assert.equal(simpleArithmetic('process.exit(0)'),null);assert.equal(simpleArithmetic('How are you?'),null);});
