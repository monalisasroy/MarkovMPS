# Scope

The package generates data. A run resolves its parameters from three layers, the defaults, a TOML instance, and the command-line flags, computes the requested observable, and writes one tab-separated table and one figure under `results/`. The committed instances under `problems/` name the regimes treated in the documentation; any other point of parameter space is one edited copy or one flag away.

What is done with the data is the reader's own analysis. Classifying a regime, fitting an exponent, and producing a publication figure each need more than one data point and a judgment the package does not make, so none of them is a function of the package. The tables carry the values and the residual or standard error beside each, and everything downstream of that is left to the reader.
