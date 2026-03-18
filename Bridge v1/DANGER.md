# DANGER: Coordinate System Bug

## Critical Issue

In this version, there is a **coordinate system bug**:

- **"depth"** refers to the **in-plane** dimension (should be out-of-plane)
- **"thickness"** refers to the **out-of-plane** dimension (should be in-plane)

This is **NOT** as intended and is inverted from the correct convention.

## Version Status

Despite this bug, this is technically the **first and most minimalist stable version** of the bridge optimization system. All other functionality works correctly with this inverted coordinate system.

## Important Notice

When interpreting results or modifying the code, keep this coordinate swap in mind. Future versions should correct this naming convention to match what I mean.
