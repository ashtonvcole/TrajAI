# `traj_ai` Module

Functions and classes for objective-frame dynamics prediction.

## Quick Reference

- [`batch_geometry.py`](batch_geometry.py): Functions to perform geometric operations on batches of vectors, e.g., a coordinate frame rotation.
- [`losses.py`](losses.py): Pytorch modules to compute loss based on position or velocity discrepancies.
- [`relaters.py`](relaters.py): Pytorch modules to compute a relationship between an influencing and influenced particle, e.g., relative displacement.
- [`transcoders.py`](transcoders.py): Pytorch modules to transform a state into an objective one, i.e., indifferent to rotations and translations.
- [`updaters.py`](updaters.py):

## Common Tensor Conventions

### Default State

```
state = [x1[n], y1[n], ..., x1[n - num_past], y1[n - num_past],
         u1[n], v1[n], ..., u1[n - num_past], v1[n - num_past],
         additional, attributes, here, ...]
```

### Normal-Tangential Objective State

```
state_invariant = [xt[1], xn[1], ..., xt[num_past], xn[num_past],
                   v,
                   vt[1], vn[1], ..., vt[num_past], vn[num_past],
                   additional, attributes, here, ...]
```

### Relation
```
state1 = [x1[n], y1[n], ..., x1[n - num_past], y1[n - num_past],
          u1[n], v1[n], ..., u1[n - num_past], v1[n - num_past],
          additional, attributes, here, ...]

state2 = [x2[n], y2[n], ..., x2[n - num_past], y2[n - num_past],
          u2[n], v2[n], ..., u2[n - num_past], v2[n - num_past],
          additional, attributes, here, ...]

become

relation12 = [dxt12[n], dxn12[n], ..., dxt12[n - num_past], dxn12[n - num_past],
              dvt12[n], dvn12[n], ..., dvt12[n - num_past], dvn12[n - num_past],
              d12[n], ..., d12[n - num_past]]

dxt12[i]: The tangential component of the displacement of 2 from 1 at frame i.
dxn12[i]: The normal component of the displacement of 2 from 1 at frame i.
dvt12[i]: The tangential component of the velocity difference of 2 from 1 at frame i.
dvn12[i]: The normal component of the velocity difference of 2 from 1 at frame i.
(from the normal-tangential frame of 2, the influenced object)

```