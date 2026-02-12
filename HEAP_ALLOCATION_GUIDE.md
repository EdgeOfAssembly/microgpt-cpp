# Heap Allocation Strategy - No Stack Temporaries

## Problem

The original implementation used operator overloads that returned `Value` by value:

```cpp
Value operator+(const Value& other) const {
    return Value(data + other.data, {this, &other}, {1.0, 1.0});  // Stack temporary!
}
```

When used like this:
```cpp
Value* result = storage.store(*a + *b);
```

The `*a + *b` creates a temporary Value on the stack. Even though we store it, the children pointers in that Value point to the stack temporary, which becomes invalid after the expression completes. This caused segmentation faults during backward pass.

## Solution: Factory Methods

All Value creation now goes through `ValueStorage` factory methods that guarantee heap allocation:

```cpp
class ValueStorage {
    // Factory method creates Value directly on heap
    Value* add(Value* a, Value* b) {
        double result = a->data + b->data;
        return store(Value(result, {a, b}, {1.0, 1.0}));
    }
};
```

## Usage Guide

### ✅ CORRECT - Use Factory Methods

```cpp
ValueStorage storage;

// Basic operations
Value* a = storage.constant(3.0);
Value* b = storage.constant(4.0);
Value* sum = storage.add(a, b);           // a + b
Value* prod = storage.mul(a, b);          // a * b
Value* diff = storage.sub(a, b);          // a - b
Value* quot = storage.div(a, b);          // a / b

// Unary operations
Value* neg_a = storage.neg(a);            // -a
Value* pow_a = storage.pow(a, 2.0);       // a^2
Value* log_a = storage.log(a);            // log(a)
Value* exp_a = storage.exp(a);            // exp(a)
Value* relu_a = storage.relu(a);          // relu(a)

// Complex expressions
Value* result = storage.div(
    storage.add(storage.mul(a, b), a),    // (a*b + a) / b
    b
);

// Backward pass - no dangling pointers!
result->backward();
```

### ❌ WRONG - Direct Operator Use (DEPRECATED)

```cpp
// These create stack temporaries and cause segfaults!
Value* bad1 = storage.store(*a + *b);      // Compiler warning
Value* bad2 = storage.store(*a * *b);      // Compiler warning
Value* bad3 = storage.store(*a / *b);      // Compiler warning
Value* bad4 = storage.store(-*a);          // Compiler warning
```

## Benefits

1. **No dangling pointers** - All Values live on heap with stable addresses
2. **Explicit ownership** - Clear that storage owns all Values
3. **Better debugging** - Each Value has a stable address for inspection
4. **Impossible to misuse** - Factory methods enforce correct pattern
5. **Compiler warnings** - Deprecated operators warn if used accidentally

## Performance

Factory methods have **zero overhead** compared to operators:
- Same number of allocations (all Values go in deque)
- No extra copies (move semantics)
- Slightly more verbose but much safer

## Migration from Operator Syntax

| Old (Operators) | New (Factory Methods) |
|----------------|----------------------|
| `storage.store(*a + *b)` | `storage.add(a, b)` |
| `storage.store(*a * *b)` | `storage.mul(a, b)` |
| `storage.store(*a - *b)` | `storage.sub(a, b)` |
| `storage.store(*a / *b)` | `storage.div(a, b)` |
| `storage.store(-*a)` | `storage.neg(a)` |
| `storage.store(a->pow(2))` | `storage.pow(a, 2.0)` |
| `storage.store(a->log())` | `storage.log(a)` |
| `storage.store(a->exp())` | `storage.exp(a)` |
| `storage.store(a->relu())` | `storage.relu(a)` |

## Why Not Smart Pointers?

We considered `std::unique_ptr` and `std::shared_ptr` but chose raw pointers with arena allocation because:

1. **Performance** - Zero overhead, no reference counting
2. **Simplicity** - Clear ownership model (storage owns everything)
3. **ML Framework Pattern** - Matches PyTorch, TensorFlow design
4. **Debugging** - Easier to inspect in debugger

The trade-off is manual lifetime management, but `ValueStorage` makes this safe and automatic.

## Memory Management

All Values are stored in `std::deque<Value>` which:
- Never invalidates pointers when growing
- Automatically cleans up when storage is destroyed
- Provides stable addresses for backward pass

Usage pattern:
```cpp
{
    ValueStorage storage;  // Create arena
    
    // All operations allocate in arena
    Value* result = /* ... computation ... */;
    
    // Backward pass uses stable pointers
    result->backward();
    
    // Optimizer reads gradients
    optimizer.step(params);
    
}  // Storage destroyed, all Values freed automatically
```

## Testing

Run tests to verify no stack temporaries:
```bash
./test_factory_methods   # Unit tests for factory methods
./test_train_small       # Integration test with training
./build/train            # Full 500-step training
```

All tests should complete without segmentation faults.
