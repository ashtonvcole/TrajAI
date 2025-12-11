# `gns_tools` Library

Functions and classes to build and train an arbitrary Graph Neural Simulator.

## Quick Reference

- [`gnn.py`](gnn.py): Classes to build a Graph Neural Network (GNN). The primary components of this are an encoder, message passing layers, and a decoder.
- [`gnn.py`](gnn.py): Classes and functions to build and rollout a Graph Neural Simulator (GNS). This is specifically a GNN which learns an update quantity, applied to the state with an updater function. The resulting GNS is a state transfer function, i.e., `X[n + 1] = GNS(X[n])`.
- [`mlp.py`](mlp.py): Generic neural networks which may be used as components to a GNN.
- [`trainers.py`](trainers.py): Generic training functions.